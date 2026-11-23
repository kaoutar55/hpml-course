"""The same ideas on a real Hugging Face model (default Qwen/Qwen2.5-1.5B) and real text.

  torchrun --standalone --nproc_per_node=8 hf_train.py --mode fsdp
  torchrun --standalone --nproc_per_node=2 hf_train.py --mode tp      # transformers' built-in tensor-parallel plan

Modes: ddp | fsdp (FSDP2 on each decoder layer) | tp (from_pretrained(tp_plan="auto"), all GPUs in one TP group).
Needs the Hub (model weights) and, for real text, raw.githubusercontent.com. Without the text it falls back to random tokens.

Precision: all three modes keep FP32 master weights and compute in BF16, so their losses can be
compared. The first run loaded the TP model in BF16 and trained it in pure BF16 (BF16 weights and
BF16 Adam state); --tp-dtype bf16 reproduces that. With lr = 1e-5, a BF16 weight of size w only changes
when the update is larger than about w / 256, so most updates round away (see update_frac_changed).
"""
import argparse
import datetime
import json
import os
import time
import urllib.request

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM, AutoTokenizer

import benchmark_utils as bu
from hpml_parallel import MixedPrecisionPolicy, fully_shard

TEXT_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'


def load_tokens(tok, seq, n_seq):
    try:
        text = urllib.request.urlopen(TEXT_URL, timeout=20).read().decode()
        ids = tok(text, return_tensors='pt').input_ids[0]
        n = min(n_seq, (len(ids) - 1) // (seq + 1))
        return ids[: n * (seq + 1)].view(n, seq + 1), 'tiny shakespeare'
    except Exception as e:
        print('no text download, using random tokens:', e, flush=True)
        g = torch.Generator().manual_seed(1234)
        return torch.randint(0, tok.vocab_size, (n_seq, seq + 1), generator=g), 'random tokens'


def local(t):
    return t.to_local() if isinstance(t, DTensor) else t


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True, choices=['ddp', 'fsdp', 'tp'])
    p.add_argument('--name', default='Qwen/Qwen2.5-1.5B')
    p.add_argument('--seq', type=int, default=1024)
    p.add_argument('--global-batch', type=int, default=16)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--tp-dtype', default='fp32', choices=['fp32', 'bf16'],
                   help='tp mode weights: fp32 master + BF16 autocast (default), or pure BF16 (original lab)')
    p.add_argument('--tag', default='')
    p.add_argument('--out', default='results_hf.jsonl')
    a = p.parse_args()

    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=10))
    rank, W = dist.get_rank(), dist.get_world_size()
    dev = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(dev)
    say = (lambda *m: print(*m, flush=True)) if rank == 0 else (lambda *m: None)

    tok = AutoTokenizer.from_pretrained(a.name)
    data, source = load_tokens(tok, a.seq, 256)
    if a.mode == 'tp':
        # transformers reads the model's own TP plan (q, k, v, gate, up by columns; o, down by rows)
        # and builds a TP mesh over all W GPUs. Every GPU sees the same sequences (dp = 1).
        dtype = torch.float32 if a.tp_dtype == 'fp32' else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(a.name, torch_dtype=dtype, tp_plan='auto')
        dp, dp_rank = 1, 0
        amp = a.tp_dtype == 'fp32'
    else:
        model = AutoModelForCausalLM.from_pretrained(a.name, torch_dtype=torch.float32).to(dev)
        dp, dp_rank = W, rank
        if a.mode == 'ddp':
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index])
            amp = True
        else:
            mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
            for layer in model.model.layers:
                fully_shard(layer, mp_policy=mp)
            fully_shard(model, mp_policy=mp)
            amp = False       # FSDP's policy already casts the weights to BF16 for compute
    say(f'{a.name} | mode {a.mode} | {W} GPUs | data: {source}')
    # the TP plan leaves the norm weights as plain tensors next to DTensors; the fused (foreach)
    # AdamW cannot mix the two, so use the per-tensor loop only in that case.
    params = list(model.parameters())
    mixed = len({isinstance(q, DTensor) for q in params}) > 1
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, foreach=not mixed)
    per = a.global_batch // dp
    times, losses, frac_changed = [], [], None
    for i in range(a.steps):
        rows = data[[(i * a.global_batch + j) % len(data) for j in range(a.global_batch)]]
        ids = rows[dp_rank * per:(dp_rank + 1) * per].to(dev)
        if i == a.warmup:
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            loss = model(input_ids=ids, labels=ids).loss      # the model shifts the labels itself
        loss.backward()
        if i == 0:
            before = [local(q).detach().clone() for q in params]
        opt.step()
        if i == 0:
            # fraction of this GPU's weight elements that the first update changed at all
            changed = sum((local(q).detach() != b).sum().item() for q, b in zip(params, before))
            total = sum(b.numel() for b in before)
            frac_changed = changed / total
            del before
        opt.zero_grad(set_to_none=True)
        l = loss.detach().float()
        dist.all_reduce(l, op=dist.ReduceOp.AVG)
        torch.cuda.synchronize()
        t = torch.tensor([time.perf_counter() - t0], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        losses.append(round(l.item(), 4))
        if i >= a.warmup:
            times.append(t.item())
        say(f'step {i:3d}  loss {l.item():.4f}  {t.item() * 1e3:8.1f} ms')
    peak = torch.tensor([torch.cuda.max_memory_allocated()], device=dev, dtype=torch.float64)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    state = torch.tensor([torch.cuda.memory_allocated()], device=dev, dtype=torch.float64)
    dist.all_reduce(state, op=dist.ReduceOp.MAX)
    st = bu.time_stats(times)
    step_s = st['step_ms'] / 1e3
    res = dict(tag=a.tag, mode=a.mode, model=a.name, gpus=W, seq=a.seq, global_batch=a.global_batch,
               step_ms=st['step_ms'], tokens_per_s=round(a.global_batch * a.seq / step_s),
               peak_gb=round(peak.item() / 2**30, 2), losses=losses, data=source, torch=torch.__version__,
               status='ok', state_gb=round(state.item() / 2**30, 2), dp=dp, seqs_per_gpu=per,
               weights_dtype='bf16' if (a.mode == 'tp' and a.tp_dtype == 'bf16') else 'fp32',
               update_frac_changed=round(frac_changed, 4), **{k: v for k, v in st.items() if k != 'step_ms'})
    if rank == 0:
        print('RESULT ' + json.dumps(res), flush=True)
        with open(a.out, 'a') as f:
            f.write(json.dumps(res) + '\n')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
