"""Controlled numerical check: does each parallel layout compute the same step as one GPU?

  torchrun --standalone --nproc_per_node=8 numerical_validation.py --model 350M --dtype fp32 \
      --modes ddp,fsdp,fsdp_z2,hsdp,tp2,tp2_nosp,tp4,tp8,tp8_nosp,tp2_vpl,cp2 --out nv_fp32.json

Every rank first runs a reference: the unsharded model on one GPU, the whole global batch, the same
weights (same seed), the same targets and the same AdamW step as train.py. Then, for each mode, it
builds that layout with train.build(), runs ONE step with train.forward_backward() and
train.optimizer_step(), gathers every gradient and every updated weight back to full tensors
(DTensor.full_tensor()), and compares them with the reference by parameter name.

Reported per mode:
  loss_abs_diff           |loss - loss_ref|
  grad_rel_l2             ||g - g_ref|| / ||g_ref|| over all parameters together
  grad_rel_l2_worst       the same per parameter, worst parameter (name included)
  update_rel_l2           ||dw - dw_ref|| / ||dw_ref||, dw = weight change of the first AdamW step
  update_sign_flip_frac   fraction of elements whose first-step update has the opposite sign
                          (Adam's first step is about lr * sign(g), so tiny gradient differences near
                          zero flip whole updates; this is why update differences look large)
With --dtype bf16 the reference is computed twice, in FP32 and in BF16 autocast, and the FP32-vs-BF16
gap of the reference itself is reported as the rounding floor that parallel layouts are compared to.
"""
import argparse
import datetime
import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

import benchmark_utils as bu
import train as T
from hpml_llm import build_model, get_config
from profiling_utils import StepProfiler, code_revision, environment

MODES = {
    'ddp': dict(mode='ddp'),
    'fsdp': dict(mode='fsdp'),
    'fsdp_z2': dict(mode='fsdp', reshard=0),
    'hsdp': dict(mode='fsdp', hsdp_shard=4),
    'tp2': dict(mode='tp', tp=2), 'tp4': dict(mode='tp', tp=4), 'tp8': dict(mode='tp', tp=8),
    'tp2_nosp': dict(mode='tp', tp=2, sp=0), 'tp8_nosp': dict(mode='tp', tp=8, sp=0),
    'tp2_vpl': dict(mode='tp', tp=2, vocab_parallel_loss=1), 'tp8_vpl': dict(mode='tp', tp=8, vocab_parallel_loss=1),
    'tp8_autocast': dict(mode='tp', tp=8, tp_dp1_autocast=1), 'tp8_nosp_autocast': dict(mode='tp', tp=8, sp=0, tp_dp1_autocast=1),
    'cp2': dict(mode='cp', cp=2), 'cp8': dict(mode='cp', cp=8),
}


def full(t):
    return t.full_tensor() if isinstance(t, torch.distributed.tensor.DTensor) else t


def clean(name):
    return name.replace('_checkpoint_wrapped_module.', '').replace('module.', '', 1) if name.startswith('module.') \
        else name.replace('_checkpoint_wrapped_module.', '')


def reference(cfg, a, dev, bf16):
    """One step of the unsharded model on the full global batch. Returns loss, grads, updates (on CPU)."""
    m = build_model(cfg, seed=a.seed).to(dev)
    inp, tgt = T.batch_for_step(0, cfg.vocab, a.seq, a.global_batch, 0, 1, dev)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
        logits = m(inp)
    loss = F.cross_entropy(logits.float().flatten(0, 1), tgt.flatten(0, 1))
    loss.backward()
    before = {n: p.detach().float().cpu().clone() for n, p in m.named_parameters()}
    grads = {n: p.grad.detach().float().cpu().clone() for n, p in m.named_parameters()}
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr / 5, betas=(0.9, 0.95), weight_decay=0.1)   # step 0 of the warm-up
    opt.step()
    upd = {n: p.detach().float().cpu() - before[n] for n, p in m.named_parameters()}
    out = dict(loss=loss.item(), grads=grads, upd=upd)
    del m, opt, logits
    torch.cuda.empty_cache()
    return out


def compare(ref, loss, grads, upd):
    num = den = unum = uden = 0.0
    flips = total = 0
    worst = (0.0, None)
    for n, g_ref in ref['grads'].items():
        g = grads[n]
        d2, r2 = (g - g_ref).pow(2).sum().item(), g_ref.pow(2).sum().item()
        num, den = num + d2, den + r2
        rel = (d2 / r2) ** 0.5 if r2 > 0 else 0.0
        if rel > worst[0]:
            worst = (rel, n)
        u, u_ref = upd[n], ref['upd'][n]
        unum, uden = unum + (u - u_ref).pow(2).sum().item(), uden + u_ref.pow(2).sum().item()
        nz = (u_ref != 0) & (u != 0)
        flips += ((torch.sign(u) != torch.sign(u_ref)) & nz).sum().item()
        total += nz.sum().item()
    has_bad = any(not torch.isfinite(g).all() for g in grads.values())
    return dict(loss=loss, loss_abs_diff=abs(loss - ref['loss']), grad_rel_l2=(num / den) ** 0.5,
                grad_rel_l2_worst=worst[0], grad_worst_param=worst[1], update_rel_l2=(unum / uden) ** 0.5,
                update_sign_flip_frac=flips / max(total, 1), nonfinite_grads=bool(has_bad))


def run_mode(name, cfg, a, W, rank, dev):
    args = T.get_args(['--mode', 'ddp'])                     # train.py defaults, then this mode's settings
    for k, v in dict(model=a.model, seq=a.seq, global_batch=a.global_batch, lr=a.lr, dtype=a.dtype, seed=a.seed,
                     **{k: v for k, v in MODES[name].items()}).items():
        setattr(args, k, v)
    s = T.build(args, cfg, W, rank, dev)
    names = [n for n, _ in s.inner.named_parameters()]
    before = {clean(n): full(p.detach()).float().cpu().clone() for n, p in s.inner.named_parameters()}
    inp, tgt = T.batch_for_step(0, cfg.vocab, a.seq, a.global_batch, s.dp_rank, s.dp, dev)
    prof = StepProfiler(enable=False)
    mem = bu.MemoryRegions(s.inner, enable=False)
    loss = T.forward_backward(s, args, inp, tgt, rank, W, dev, prof, mem).item()
    grads = {clean(n): full(p.grad.detach()).float().cpu().clone() for n, p in s.inner.named_parameters()}
    T.optimizer_step(s, args, 0, prof)
    upd = {clean(n): full(p.detach()).float().cpu() - before[clean(n)] for n, p in s.inner.named_parameters()}
    del s
    torch.cuda.empty_cache()
    return loss, grads, upd, names


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='350M')
    p.add_argument('--seq', type=int, default=512)
    p.add_argument('--global-batch', type=int, default=8)
    p.add_argument('--dtype', default='fp32', choices=['fp32', 'bf16'])
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--modes', default='ddp,fsdp,tp2')
    p.add_argument('--out', default='numerical_validation.json')
    a = p.parse_args()
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=10))
    rank, W = dist.get_rank(), dist.get_world_size()
    dev = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(dev)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = get_config(a.model, a.seq)

    ref = reference(cfg, a, dev, bf16=(a.dtype == 'bf16'))
    result = dict(config=vars(a), world_size=W, environment=environment(),
                  code_rev=code_revision([T.__file__, T.hp.__file__, os.path.abspath(__file__)]), modes={})
    if a.dtype == 'bf16':                                    # rounding floor: the reference against itself in FP32
        ref32 = reference(cfg, a, dev, bf16=False)
        result['bf16_reference_vs_fp32_reference'] = compare(ref32, ref['loss'], ref['grads'], ref['upd'])
        del ref32
    result['reference_loss'] = ref['loss']
    for name in a.modes.split(','):
        try:
            loss, grads, upd, _ = run_mode(name, cfg, a, W, rank, dev)
            result['modes'][name] = compare(ref, loss, grads, upd)
        except Exception as e:                               # keep going; record the failure as data
            result['modes'][name] = dict(error=f'{type(e).__name__}: {str(e).splitlines()[0][:300]}')
            torch.cuda.empty_cache()
        if rank == 0:
            r = result['modes'][name]
            print(name, {k: (f'{v:.3e}' if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        dist.barrier()
    if rank == 0:
        bu.write_json(a.out, result)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
