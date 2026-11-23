"""HPML Lecture 8 multi-GPU lab: one training script, one parallelism per --mode.

  torchrun --standalone --nproc_per_node=8 train.py --mode fsdp --model 1B

Modes (W = number of GPUs launched by torchrun):
  ddp   every GPU holds the whole model and optimizer (Lecture 7)
  fsdp  FSDP2 over all W GPUs. --reshard 1: ZeRO-3, --reshard 0: ZeRO-2-like. --hsdp-shard k: HSDP, shard over k
  tp    tensor parallel over --tp GPUs (+ sequence parallel with --sp 1), FSDP2 over the other W / tp
  pp    pipeline over all W GPUs, --schedule gpipe | 1f1b | interleaved | zb
  cp    context parallel over --cp GPUs, FSDP2 over all W

Who sees which data: only data-parallel replicas get different sequences. The W GPUs form
dp = W / (tp * cp * pp) data-parallel groups; every GPU in a TP, CP or pipeline group works on the
same sequences. Each optimizer step uses --global-batch sequences in total, so every GPU processes
global_batch / dp sequences (CP then splits each of them into cp pieces along the sequence).

Rank 0 prints one line that starts with RESULT and appends the same JSON to --out.
Diagnostic options, off by default: --dtype fp32, --clip-grad, --vocab-parallel-loss, --mem-regions,
--grad-report, --profile. Without them a run uses the first-run settings (TP with dp = 1: --tp-dp1-autocast).
"""
import argparse
import contextlib
import datetime
import json
import os
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

import benchmark_utils as bu
import hpml_parallel as hp
from hpml_llm import Transformer, build_model, count_params, get_config
from profiling_utils import StepProfiler, code_revision, environment

HERE = os.path.dirname(os.path.abspath(__file__))


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True, choices=['ddp', 'fsdp', 'tp', 'pp', 'cp'])
    p.add_argument('--model', default='1B')
    p.add_argument('--seq', type=int, default=2048)
    p.add_argument('--global-batch', type=int, default=16, help='sequences per optimizer step, all GPUs together')
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--warmup', type=int, default=5, help='steps not timed')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--ac', default='none', choices=['none', 'full'])
    p.add_argument('--reshard', type=int, default=1)
    p.add_argument('--hsdp-shard', type=int, default=0)
    p.add_argument('--tp', type=int, default=1)
    p.add_argument('--sp', type=int, default=1)
    p.add_argument('--schedule', default='1f1b', choices=list(hp.SCHEDULES))
    p.add_argument('--microbatches', type=int, default=8)
    p.add_argument('--v', type=int, default=2, help='stages per GPU for interleaved schedules')
    p.add_argument('--cp', type=int, default=1)
    p.add_argument('--rotate', default='allgather', choices=['allgather', 'alltoall'])
    p.add_argument('--save', default='', help='distributed checkpoint folder to write after the last step')
    p.add_argument('--load', default='', help='distributed checkpoint folder to resume from')
    p.add_argument('--start-step', type=int, default=0, help='data position when resuming')
    p.add_argument('--peak-tflops', type=float, default=989.0, help='dense BF16 peak per GPU (H100 and H200 SXM: 989)')
    p.add_argument('--tag', default='')
    p.add_argument('--out', default='results.jsonl')
    # ---- diagnostic options used in the follow-up runs; all off by default
    p.add_argument('--seed', type=int, default=0, help='weight initialization seed')
    p.add_argument('--dtype', default='bf16', choices=['bf16', 'fp32'],
                   help='bf16: BF16 compute, FP32 master weights (original). fp32: FP32 everywhere, TF32 off')
    p.add_argument('--clip-grad', type=float, default=0.0, help='clip the global gradient norm (0 = off, original)')
    p.add_argument('--vocab-parallel-loss', type=int, default=0, help='tp mode: keep logits sharded over the vocabulary')
    p.add_argument('--tp-dp1-autocast', type=int, default=0,
                   help='tp mode with dp = 1: 1 = original lab path (no FSDP, autocast); 0 = FSDP2 over the 1-GPU dp mesh')
    p.add_argument('--mem-regions', default='', help='JSON file: memory at named points of the first timed step')
    p.add_argument('--mem-snapshot', type=int, default=0, help='with --mem-regions: group live memory by allocation site')
    p.add_argument('--grad-report', default='', help='JSON file: loss and per-parameter gradient norms of step 0')
    p.add_argument('--profile', type=int, default=0, help='1: record a short torch.profiler trace (off by default)')
    p.add_argument('--profile-ranks', default='0', help='comma-separated ranks to trace')
    p.add_argument('--profile-dir', default='profiles/run')
    p.add_argument('--profile-wait', type=int, default=2)
    p.add_argument('--profile-warmup', type=int, default=1)
    p.add_argument('--profile-active', type=int, default=2)
    p.add_argument('--profile-shapes', type=int, default=0)
    p.add_argument('--profile-memory', type=int, default=0)
    p.add_argument('--profile-stack', type=int, default=0)
    return p.parse_args(argv)


def batch_for_step(step, vocab, seq, global_batch, dp_rank, dp_size, device, pool=64):
    """Deterministic data: a fixed pool of 64 random sequences, cycled. The model memorizes them, so the
    loss falls, and every mode sees exactly the same tokens at the same step.
    Step k uses pool rows (16k ... 16k + 15) mod 64 for a global batch of 16, so the data repeats every
    4 steps. Data-parallel rank r takes its own contiguous slice of the global batch."""
    g = torch.Generator().manual_seed(1234)
    data = torch.randint(0, vocab, (pool, seq + 1), generator=g)
    idx = [(step * global_batch + i) % pool for i in range(global_batch)]
    per = global_batch // dp_size
    rows = data[idx[dp_rank * per:(dp_rank + 1) * per]].to(device)
    return rows[:, :-1].contiguous(), rows[:, 1:].contiguous()     # inputs and next-token targets


def all_max(x, device):
    t = torch.tensor([float(x)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def build(a, cfg, W, rank, dev):
    """Build the model on CPU, then apply the parallelism of --mode. Returns everything the step needs."""
    model = build_model(cfg, seed=a.seed)          # identical weights on every rank, built on CPU
    tp = a.tp if a.mode == 'tp' else 1
    cp = a.cp if a.mode == 'cp' else 1
    pp = W if a.mode == 'pp' else 1
    dp = W // (tp * cp * pp)
    assert dp * tp * cp * pp == W, 'the parallel degrees must multiply to the number of GPUs'
    assert a.global_batch % dp == 0, 'global batch must divide over the data-parallel GPUs'
    bf16 = a.dtype == 'bf16'
    pdt = torch.bfloat16 if bf16 else torch.float32
    s = SimpleNamespace(model=model, tp=tp, cp=cp, pp=pp, dp=dp, dp_rank=rank, amp=False, cp_mesh=None,
                        stages=None, sched=None, vpl=bool(a.vocab_parallel_loss) and a.mode == 'tp')

    if a.mode == 'ddp':
        model.to(dev)
        hp.apply_ac(model, a.ac)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index], broadcast_buffers=False)
        s.amp = bf16          # FP32 weights, BF16 compute: 4 + 4 + 8 = 16 bytes per parameter
    elif a.mode == 'fsdp':
        if a.hsdp_shard:
            mesh = init_device_mesh('cuda', (W // a.hsdp_shard, a.hsdp_shard), mesh_dim_names=('replicate', 'shard'))
        else:
            mesh = init_device_mesh('cuda', (W,), mesh_dim_names=('dp',))
        model.to(dev)
        hp.apply_ac(model, a.ac)
        hp.apply_fsdp(model, mesh, reshard_after_forward=bool(a.reshard), param_dtype=pdt)
    elif a.mode == 'tp':
        # Column-parallel wq/wk/wv split the heads, so every TP rank needs whole query and K, V heads.
        assert cfg.n_heads % tp == 0 and cfg.n_kv_heads % tp == 0, \
            f'--tp {tp} must divide n_heads = {cfg.n_heads} and n_kv_heads = {cfg.n_kv_heads}'
        # 2D mesh: rows are data-parallel groups, columns are TP groups. For W = 8, tp = 2:
        # [[0, 1], [2, 3], [4, 5], [6, 7]]. GPUs 0 and 1 split every matrix and see the same
        # sequences; GPUs 0, 2, 4, 6 hold the same TP shard and see different sequences.
        mesh = init_device_mesh('cuda', (dp, tp), mesh_dim_names=('dp', 'tp'))
        s.dp_rank = mesh['dp'].get_local_rank()
        model.to(dev)
        hp.apply_tp(model, mesh['tp'], sequence_parallel=bool(a.sp), vocab_parallel_loss=s.vpl)
        hp.apply_ac(model, a.ac)
        if dp > 1 or not a.tp_dp1_autocast:
            # FSDP2 over the dp sub-mesh, even when dp = 1 (nothing to shard, but the same BF16
            # mixed-precision policy as the dp > 1 rows: weights are cast to BF16 once per block and every
            # activation stays BF16).
            hp.apply_fsdp(model, mesh['dp'], reshard_after_forward=bool(a.reshard), param_dtype=pdt)
        else:
            # First-run behaviour (--tp-dp1-autocast 1): no FSDP, FP32 weights and autocast. The residual
            # stream stays FP32, and autocast casts the gathered input to BF16 separately for wq, wk, wv,
            # w1 and w3, saving five BF16 copies per layer for backward instead of one (Part C).
            s.amp = bf16
    elif a.mode == 'cp':
        mesh = init_device_mesh('cuda', (dp, cp), mesh_dim_names=('dp', 'cp'))
        s.cp_mesh, s.dp_rank = mesh['cp'], mesh['dp'].get_local_rank()
        fsdp_mesh = init_device_mesh('cuda', (W,), mesh_dim_names=('dp_cp',))   # FSDP shards over all W GPUs
        model.to(dev)
        hp.apply_ac(model, a.ac)
        hp.apply_fsdp(model, fsdp_mesh, reshard_after_forward=bool(a.reshard), param_dtype=pdt)
    elif a.mode == 'pp':
        assert a.ac == 'none', 'keep the pipeline demo simple: no activation checkpointing'
        pp_group = init_device_mesh('cuda', (W,), mesh_dim_names=('pp',)).get_group()
        s.stages, s.sched = hp.build_pipeline(model, rank, W, a.schedule, a.microbatches, a.v, dev, pp_group)
        s.dp_rank, s.amp = 0, bf16
    s.model = model
    s.inner = model.module if a.mode == 'ddp' else model
    s.params = [p for st in s.stages for p in st.submod.parameters()] if s.stages else list(model.parameters())
    # TP without SP leaves the norm weights as plain tensors next to DTensors; the fused (foreach)
    # AdamW cannot mix the two, so use the per-tensor loop only in that case.
    s.mixed = len({isinstance(p, DTensor) for p in s.params}) > 1
    s.opt = torch.optim.AdamW(s.params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1, foreach=not s.mixed)
    return s


def grad_norm_sq(params, pp):
    """Squared global gradient norm. A DTensor norm is reduced over its mesh by full_tensor(); under PP
    each rank holds different layers, so the per-rank sums are added with an all-reduce."""
    total = None
    for p in params:
        if p.grad is None:
            continue
        n = torch.linalg.vector_norm(p.grad.float())
        n = n.full_tensor() if isinstance(n, DTensor) else n
        total = n * n if total is None else total + n * n
    total = total if total is not None else torch.zeros((), device=torch.cuda.current_device())
    if pp > 1:
        dist.all_reduce(total)
    return total


def forward_backward(s, a, inp, tgt, rank, W, dev, prof, mem):
    """One forward and backward pass. Returns the loss averaged over all GPUs (a plain tensor)."""
    autocast = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if s.amp else contextlib.nullcontext
    if a.mode == 'pp':
        losses = []
        with prof.region('pipeline step'), autocast():
            if rank == 0:
                s.sched.step(inp)
            elif rank == W - 1:
                s.sched.step(target=tgt, losses=losses)
            else:
                s.sched.step()
        loss = torch.stack(losses).mean() if losses else torch.zeros((), device=dev)
        dist.broadcast(loss, src=W - 1)             # only the last stage computed the loss
        return loss
    ctx = hp.cp_context(s.cp_mesh, [inp, tgt, s.inner.freqs_cis], [1, 1, 0], {inp, tgt}, a.rotate) if s.cp > 1 \
        else contextlib.nullcontext()
    with ctx, hp.loss_context(s.vpl):
        with prof.region('forward'), autocast():
            logits = s.model(inp)                   # (b, s, V); a DTensor sharded over V with --vocab-parallel-loss
        mem.mark('after forward (logits exist)')
        with prof.region('loss'):
            loss = F.cross_entropy(logits.float().flatten(0, 1), tgt.flatten(0, 1))
        mem.mark('after loss (FP32 logits exist)')
        if a.mem_snapshot and mem.on:
            mem.snapshot_before_backward = bu.snapshot_summary()
        # logits stays referenced until the step returns, as in the first run (it is part of the peak)
        with prof.region('backward'):
            loss.backward()
        mem.mark('after backward')
    loss = loss.detach()
    loss = loss.full_tensor() if isinstance(loss, DTensor) else loss
    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
    return loss


def optimizer_step(s, a, step, prof):
    with prof.region('optimizer'):
        if a.clip_grad > 0:
            norm = grad_norm_sq(s.params, s.pp).sqrt()
            scale = torch.clamp(a.clip_grad / (norm + 1e-6), max=1.0)
            for p in s.params:
                if p.grad is not None:
                    p.grad.mul_(scale)
        lr_scale = min(1.0, (step + 1) / 5)          # 5 warm-up steps for the learning rate
        for gp in s.opt.param_groups:
            gp['lr'] = a.lr * lr_scale
        s.opt.step()
        s.opt.zero_grad(set_to_none=True)


def named_norms(s, use_grad):
    """Per-parameter L2 norms (of the gradient or of the weight), as Python floats keyed by name."""
    out = {}
    named = [(n, p) for st in s.stages for n, p in st.submod.named_parameters()] if s.stages \
        else list(s.inner.named_parameters())
    for n, p in named:
        t = p.grad if use_grad else p.data
        if t is None:
            continue
        v = torch.linalg.vector_norm(t.float())
        v = v.full_tensor() if isinstance(v, DTensor) else v
        out[n.replace('_checkpoint_wrapped_module.', '').replace('_fsdp_wrapped_module.', '')] = v.item()
    return out


def main(argv=None):
    a = get_args(argv)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=5))
    rank, W = dist.get_rank(), dist.get_world_size()
    dev = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(dev)
    say = (lambda *m: print(*m, flush=True)) if rank == 0 else (lambda *m: None)
    if a.dtype == 'fp32':
        torch.backends.cuda.matmul.allow_tf32 = False      # true FP32 matmuls for correctness tests
        torch.backends.cudnn.allow_tf32 = False
    if a.profile and a.out == 'results.jsonl':
        a.out = 'results_profiled.jsonl'                  # never mix profiled timings into the benchmark file

    cfg = get_config(a.model, a.seq)
    s = build(a, cfg, W, rank, dev)
    with torch.device('meta'):                          # count on an unsharded copy that uses no memory
        n_all, n_flop = count_params(Transformer(cfg))
    say(f'model {a.model}: {n_all / 1e9:.2f}B parameters, {cfg.n_layers} layers, h = {cfg.dim}, seq {a.seq}, {W} GPUs')
    torch.cuda.empty_cache()

    if a.load:
        assert a.mode == 'fsdp', 'the checkpoint demo uses FSDP'
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
        msd, osd = get_state_dict(s.model, s.opt)
        dcp.load({'model': msd, 'optim': osd}, checkpoint_id=a.load)
        set_state_dict(s.model, s.opt, model_state_dict=msd, optim_state_dict=osd)
        say(f'resumed from {a.load} on {W} GPUs')

    files = [os.path.join(HERE, f) for f in ('train.py', 'hpml_llm.py', 'hpml_parallel.py', 'benchmark_utils.py')]
    rev = code_revision(files)
    meta = dict(mode=a.mode, model=a.model, gpus=W, dp=s.dp, tp=s.tp, pp=s.pp, cp=s.cp, seq=a.seq,
                global_batch=a.global_batch, dtype=a.dtype, schedule=a.schedule if a.mode == 'pp' else None,
                microbatches=a.microbatches if a.mode == 'pp' else None, steps=a.steps, tag=a.tag, code_rev=rev)
    prof = StepProfiler(enable=a.profile, rank=rank, ranks=[int(r) for r in a.profile_ranks.split(',')],
                        out_dir=a.profile_dir, wait=a.profile_wait, warmup=a.profile_warmup, active=a.profile_active,
                        record_shapes=bool(a.profile_shapes), profile_memory=bool(a.profile_memory),
                        with_stack=bool(a.profile_stack), metadata=meta)
    mem = bu.MemoryRegions(s.inner, enable=bool(a.mem_regions) and a.mode != 'pp')
    mem.snapshot_before_backward = None
    if a.mem_snapshot and a.mem_regions:
        bu.start_history()

    tokens = a.global_batch * a.seq
    times, losses, status, oom_snapshot, grad_rep = [], [], 'ok', None, None
    try:
        with prof:
            for i in range(a.steps):
                step = a.start_step + i
                inp, tgt = batch_for_step(step, cfg.vocab, a.seq, a.global_batch, s.dp_rank, s.dp, dev)
                if i == a.warmup:
                    torch.cuda.reset_peak_memory_stats()
                    if a.mem_regions:
                        mem.start()
                torch.cuda.synchronize()
                dist.barrier()
                t0 = time.perf_counter()
                loss = forward_backward(s, a, inp, tgt, rank, W, dev, prof, mem)
                if i == 0 and a.grad_report:
                    grad_rep = dict(loss=loss.item(), grad_norm=grad_norm_sq(s.params, s.pp).sqrt().item(),
                                    grads=named_norms(s, use_grad=True))
                optimizer_step(s, a, step, prof)
                if i == 0 and a.grad_report:
                    grad_rep['weights_after_step'] = named_norms(s, use_grad=False)
                loss = loss.item()                       # .item() waits for the GPU
                torch.cuda.synchronize()
                dt = all_max(time.perf_counter() - t0, dev)   # the step takes as long as the slowest GPU
                if mem.on:
                    mem.mark('after optimizer step')
                    mem.points.append(('peak during step', bu.gb(torch.cuda.max_memory_allocated())))
                    mem.stop()
                losses.append(round(loss, 4))
                if i >= a.warmup:
                    times.append(dt)
                say(f'step {step:3d}  loss {loss:.4f}  {dt * 1e3:8.1f} ms')
                prof.step()
    except torch.cuda.OutOfMemoryError as e:
        status = 'OOM'
        if a.mem_snapshot and a.mem_regions:
            oom_snapshot = bu.snapshot_summary()          # taken while the failed step's tensors are still alive
        say(f'OUT OF MEMORY: {str(e).splitlines()[0]}')
    if a.mem_snapshot and a.mem_regions:
        bu.stop_history()

    peak = all_max(torch.cuda.max_memory_allocated(), dev)
    peak_res = all_max(torch.cuda.max_memory_reserved(), dev)
    state = all_max(torch.cuda.memory_allocated(), dev)    # weights + optimizer state between steps
    st = bu.time_stats(times)
    step_s = st['step_ms'] / 1e3
    flops_per_token = 6 * n_flop + 12 * cfg.n_layers * cfg.dim * a.seq   # matmuls + attention scores
    mfu = flops_per_token * tokens / step_s / (W * a.peak_tflops * 1e12) if times else float('nan')
    # The 12*L*h*s attention term counts the full s x s score matrix. With a causal mask the fused kernel
    # skips the upper triangle, so it does about half that work: 6*L*h*s. mfu keeps the definition
    # used in the first run, so the two stay comparable; mfu_causal counts only the work a causal kernel does.
    flops_causal = 6 * n_flop + 6 * cfg.n_layers * cfg.dim * a.seq
    mfu_causal = flops_causal * tokens / step_s / (W * a.peak_tflops * 1e12) if times else float('nan')

    if a.save and status == 'ok':
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict
        msd, osd = get_state_dict(s.model, s.opt)
        dcp.save({'model': msd, 'optim': osd}, checkpoint_id=a.save)
        say(f'saved {a.save}: every rank wrote its own shards')

    seqs_per_gpu = a.global_batch // s.dp
    res = dict(tag=a.tag, mode=a.mode, model=a.model, params_b=round(n_all / 1e9, 3), gpus=W, dp=s.dp, tp=s.tp,
               pp=s.pp, cp=s.cp, sp=a.sp if a.mode == 'tp' else None, reshard=a.reshard, hsdp_shard=a.hsdp_shard,
               ac=a.ac, schedule=a.schedule if a.mode == 'pp' else None,
               microbatches=a.microbatches if a.mode == 'pp' else None, seq=a.seq, global_batch=a.global_batch,
               step_ms=st['step_ms'],
               tokens_per_s=round(tokens / step_s) if times else float('nan'),
               mfu=round(mfu, 4), mfu_causal=round(mfu_causal, 4), peak_gb=bu.gb(peak), state_gb=bu.gb(state), losses=losses, status=status,
               torch=torch.__version__, gpu=torch.cuda.get_device_name(dev),
               # ---- added fields
               **{k: v for k, v in st.items() if k != 'step_ms'},
               peak_reserved_gb=bu.gb(peak_res), tokens_per_step=tokens, seqs_per_gpu=seqs_per_gpu,
               tokens_per_gpu_per_step=seqs_per_gpu * a.seq // s.cp,
               tokens_per_s_per_gpu=round(tokens / step_s / W) if times else float('nan'),
               dtype=a.dtype, clip_grad=a.clip_grad, vocab_parallel_loss=int(s.vpl), tp_dp1_autocast=a.tp_dp1_autocast if a.mode == 'tp' else None, lr=a.lr, seed=a.seed,
               warmup=a.warmup, steps=a.steps, profiled=bool(a.profile), code_rev=rev, nccl=environment().get('nccl'))
    if rank == 0:
        print('RESULT ' + json.dumps(res), flush=True)
        with open(a.out, 'a') as f:
            f.write(json.dumps(res) + '\n')
        if a.mem_regions:
            bu.write_json(a.mem_regions, dict(run=res, points=mem.points, snapshot_before_backward=mem.snapshot_before_backward,
                                              oom_snapshot=oom_snapshot))
        if grad_rep is not None:
            bu.write_json(a.grad_report, dict(run=res, **grad_rep))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
