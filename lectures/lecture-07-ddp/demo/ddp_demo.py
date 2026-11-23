# ddp_demo.py: PyTorch DDP on 2 GPUs.
# Run on a machine with 2 GPUs:   torchrun --standalone --nproc_per_node=2 ddp_demo.py --part all
# Run without GPUs (mechanics only, timings are not meaningful):
#                                 torchrun --standalone --nproc_per_node=2 ddp_demo.py --cpu --width 256 --layers 4 --batch 32
import argparse
import json
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.nn.parallel import DistributedDataParallel as DDP


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["scaling", "buckets", "nosync", "trace", "all"])
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--batch", type=int, default=256, help="samples per GPU per step")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--buckets", default="1,5,25,100,400", help="bucket_cap_mb values to sweep")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--out", default="results")
    ap.add_argument("--cpu", action="store_true", help="gloo on CPU, for machines without 2 GPUs")
    return ap.parse_args()


# ---------------------------------------------------------------- setup
def setup(args):
    if args.cpu:
        dist.init_process_group("gloo")
        return torch.device("cpu"), None
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)                 # without this every rank lands on GPU 0
    dist.init_process_group("nccl")
    torch.backends.cuda.matmul.allow_tf32 = True
    return torch.device("cuda", local), local


def make_model(args, dev):
    torch.manual_seed(0)                         # same initial weights on every rank and every run
    layers = []
    for _ in range(args.layers):
        layers += [nn.Linear(args.width, args.width), nn.GELU()]
    layers.append(nn.Linear(args.width, 10))
    return nn.Sequential(*layers).to(dev)


def make_batch(args, dev, rank, i=0):
    g = torch.Generator().manual_seed(1000 * rank + i)   # different data on every rank
    x = torch.randn(args.batch, args.width, generator=g)
    y = torch.randint(0, 10, (args.batch,), generator=g)
    return x.to(dev), y.to(dev)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)              # GPU work is asynchronous: wait before reading the clock


def local_time(step, args, dev):
    """Average seconds per step on this rank only (no collectives)."""
    for _ in range(args.warmup):
        step()
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        step()
    sync(dev)
    return (time.perf_counter() - t0) / args.steps


def global_time(step, args, dev):
    """Average seconds per step, taking the slowest rank (that rank sets the pace)."""
    for _ in range(args.warmup):
        step()
    sync(dev)
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        step()
    sync(dev)
    t = torch.tensor([(time.perf_counter() - t0) / args.steps], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def wrap(model, local, **kw):
    return DDP(model, device_ids=[local] if local is not None else None, **kw)


def counting_hook(state, bucket):
    """Same all-reduce as DDP's default, plus a counter of bucket all-reduces."""
    state["n"] += 1
    return default_hooks.allreduce_hook(None, bucket)


def free(dev):
    if dev.type == "cuda":
        torch.cuda.empty_cache()


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


# ---------------------------------------------------------------- part 1: scaling
def part_scaling(args, dev, local, rank, p, res):
    x, y = make_batch(args, dev, rank)

    # 1 GPU: rank 0 trains alone, the other ranks wait
    t1 = 0.0
    if rank == 0:
        m = make_model(args, dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

        def step1():
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(x), y).backward()
            opt.step()
        t1 = local_time(step1, args, dev)
        del m, opt
        free(dev)
    t = torch.tensor([t1], device=dev)
    dist.broadcast(t, src=0)
    t1 = t.item()

    # The all-reduce alone, on a buffer as large as the gradient
    nparams = args.layers * (args.width * args.width + args.width) + args.width * 10 + 10
    flat = torch.zeros(nparams, device=dev)
    t_ar = global_time(lambda: dist.all_reduce(flat), args, dev)
    del flat
    free(dev)

    # p GPUs with DDP, same per-GPU batch (weak scaling)
    m = wrap(make_model(args, dev), local)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

    def stepN():
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(m(x), y).backward()      # DDP all-reduces the buckets during this call
        opt.step()
    tN = global_time(stepN, args, dev)
    del m, opt
    free(dev)

    exposed = max(tN - t1, 0.0)
    r = dict(params_M=nparams / 1e6, grad_MB=nparams * 4 / 1e6, t1_ms=t1 * 1e3, t_allreduce_ms=t_ar * 1e3,
             tN_ms=tN * 1e3, exposed_ms=exposed * 1e3,
             samples_per_s_1=args.batch / t1, samples_per_s_N=p * args.batch / tN, efficiency=t1 / tN)
    res["scaling"] = r
    log(rank, f"\n== Scaling: {nparams / 1e6:.0f}M parameters, {nparams * 4 / 1e6:.0f} MB of FP32 gradients, batch {args.batch} per GPU")
    log(rank, f"1 GPU step                      {r['t1_ms']:8.1f} ms   {r['samples_per_s_1']:9.0f} samples/s")
    log(rank, f"{p} GPUs, DDP step                {r['tN_ms']:8.1f} ms   {r['samples_per_s_N']:9.0f} samples/s")
    log(rank, f"all-reduce of the gradient alone {r['t_allreduce_ms']:8.1f} ms")
    log(rank, f"DDP overhead (DDP step - 1 GPU)  {r['exposed_ms']:8.1f} ms   exposed communication, bucket copies and SMs shared with NCCL")
    log(rank, "  (part B4 splits this with a profiler trace)")
    log(rank, f"scaling efficiency E = t1 / tN   {r['efficiency']:8.2f}")


# ---------------------------------------------------------------- part 2: bucket size
def part_buckets(args, dev, local, rank, p, res):
    x, y = make_batch(args, dev, rank)
    rows = []
    for cap in [float(c) for c in args.buckets.split(",")]:
        # Count bucket all-reduces in one step (after DDP's first-iteration bucket rebuild)
        m = wrap(make_model(args, dev), local, bucket_cap_mb=cap)
        state = {"n": 0}
        m.register_comm_hook(state, counting_hook)
        for _ in range(2):
            state["n"] = 0
            m.zero_grad(set_to_none=True)
            F.cross_entropy(m(x), y).backward()
        n_buckets = state["n"]
        del m
        free(dev)

        # Time full training steps with DDP's default (C++) all-reduce
        m = wrap(make_model(args, dev), local, bucket_cap_mb=cap)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

        def step():
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(x), y).backward()
            opt.step()
        t = global_time(step, args, dev)
        del m, opt
        free(dev)
        rows.append(dict(bucket_cap_mb=cap, all_reduces_per_step=n_buckets, step_ms=t * 1e3))
    res["buckets"] = rows
    log(rank, "\n== Bucket size")
    log(rank, f"{'bucket_cap_mb':>14} {'all-reduces/step':>17} {'step time':>11}")
    for r in rows:
        log(rank, f"{r['bucket_cap_mb']:>14.0f} {r['all_reduces_per_step']:>17d} {r['step_ms']:>8.1f} ms")


# ---------------------------------------------------------------- part 3: no_sync
def part_nosync(args, dev, local, rank, p, res):
    batches = [make_batch(args, dev, rank, i) for i in range(args.accum)]

    def run(use_no_sync, timed):
        m = wrap(make_model(args, dev), local)
        state = {"n": 0}
        m.register_comm_hook(state, counting_hook)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

        def accumulate():
            for i, (x, y) in enumerate(batches):
                last = i == len(batches) - 1
                ctx = m.no_sync() if (use_no_sync and not last) else nullcontext()
                with ctx:                                       # forward AND backward inside no_sync
                    (F.cross_entropy(m(x), y) / args.accum).backward()

        def step():
            accumulate()
            opt.step()
            opt.zero_grad(set_to_none=True)

        if timed:
            t = global_time(step, args, dev)
            per_step = state["n"] / (args.warmup + args.steps)
            del m, opt
            free(dev)
            return t, per_step
        accumulate()                                            # one window, no optimizer step
        g = torch.cat([q.grad.flatten() for q in m.parameters()]).float()
        del m, opt
        free(dev)
        return g

    t_plain, n_plain = run(False, True)
    t_ns, n_ns = run(True, True)
    g_plain, g_ns = run(False, False), run(True, False)
    rel = ((g_plain - g_ns).norm() / g_plain.norm()).item()
    r = dict(accum=args.accum, plain_ms=t_plain * 1e3, plain_allreduces=n_plain,
             no_sync_ms=t_ns * 1e3, no_sync_allreduces=n_ns, grad_rel_diff=rel)
    res["nosync"] = r
    log(rank, f"\n== Gradient accumulation over {args.accum} micro-batches")
    log(rank, f"plain DDP     {r['plain_ms']:8.1f} ms per optimizer step, {r['plain_allreduces']:.0f} bucket all-reduces")
    log(rank, f"with no_sync  {r['no_sync_ms']:8.1f} ms per optimizer step, {r['no_sync_allreduces']:.0f} bucket all-reduces")
    log(rank, f"same gradients? relative difference {rel:.1e} (rounding only)")


# ---------------------------------------------------------------- part 4: profiler trace
def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def overlap(a, b):
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            tot += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def analyze_trace(path, out, title):
    with open(path) as f:
        ev = json.load(f)["traceEvents"]
    X = [e for e in ev if e.get("ph") == "X" and "ts" in e and "dur" in e]
    # GPU-side annotations mark where each step really ran on the GPU (the CPU runs ahead).
    fwd = sorted((e for e in X if e.get("cat") == "gpu_user_annotation" and e["name"] == "DistributedDataParallel.forward"),
                 key=lambda e: e["ts"])
    opt = sorted((e for e in X if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("Optimizer.step")),
                 key=lambda e: e["ts"])
    if not fwd or not opt:
        return None
    start = fwd[-1]["ts"]                                     # last profiled step on the GPU
    opt_last = [e for e in opt if e["ts"] >= start][-1]
    end = opt_last["ts"] + opt_last["dur"]
    kern = [e for e in X if e.get("cat") == "kernel" and start <= e["ts"] and e["ts"] + e["dur"] <= end]
    if not kern:
        return None
    rel = lambda e: (e["ts"] - start, e["ts"] - start + e["dur"])
    comm = merge([rel(e) for e in kern if "nccl" in e["name"].lower()])
    comp = merge([rel(e) for e in kern if "nccl" not in e["name"].lower()])
    comm_t, comp_t = sum(e - s for s, e in comm), sum(e - s for s, e in comp)
    hidden = overlap(comm, comp)
    f_end = fwd[-1]["dur"]
    o_start = opt_last["ts"] - start
    phases = [("forward", 0, f_end), ("backward", f_end, o_start), ("optimizer", o_start, end - start)]
    r = dict(step_ms=(end - start) / 1e3, forward_ms=f_end / 1e3, backward_ms=(o_start - f_end) / 1e3,
             optimizer_ms=opt_last["dur"] / 1e3, compute_ms=comp_t / 1e3, nccl_ms=comm_t / 1e3,
             nccl_kernels=sum(1 for e in kern if "nccl" in e["name"].lower()),
             nccl_hidden_ms=hidden / 1e3, nccl_exposed_ms=(comm_t - hidden) / 1e3)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.broken_barh([(s / 1e3, (e - s) / 1e3) for s, e in comp], (20, 8), color="#0F62FE")
    ax.broken_barh([(s / 1e3, (e - s) / 1e3) for s, e in comm], (8, 8), color="#DA1E28")
    for name, s, e in phases:
        ax.broken_barh([(s / 1e3, (e - s) / 1e3)], (32, 6), color="#E0E0E0", edgecolor="#8D8D8D")
        ax.text((s + e) / 2e3, 35, name, ha="center", va="center", fontsize=9)
    ax.set_yticks([12, 24, 35])
    ax.set_yticklabels(["NCCL kernels", "compute kernels", "phase"])
    ax.set_xlabel("time in one DDP step (ms), rank 0")
    ax.set_ylim(4, 40)
    ax.set_title(f"{title}: {r['nccl_kernels']} all-reduces, NCCL {r['nccl_ms']:.1f} ms, {r['nccl_hidden_ms']:.1f} ms under "
                 f"compute and {r['nccl_exposed_ms']:.1f} ms exposed", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "trace_step.png"), dpi=150)
    plt.close(fig)
    return r


def part_trace(args, dev, local, rank, p, res):
    from torch.profiler import ProfilerActivity, profile, record_function, schedule
    x, y = make_batch(args, dev, rank)
    m = wrap(make_model(args, dev), local)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
    path = os.path.join(args.out, "trace_rank0.json")

    def save(prof):
        if rank == 0:
            prof.export_chrome_trace(path)        # open in chrome://tracing or ui.perfetto.dev

    with profile(activities=acts, schedule=schedule(wait=1, warmup=3, active=2, repeat=1), on_trace_ready=save) as prof:
        for _ in range(6):
            with record_function("forward"):
                loss = F.cross_entropy(m(x), y)
            with record_function("backward"):
                loss.backward()
            with record_function("optimizer"):
                opt.step()
                opt.zero_grad(set_to_none=True)
            prof.step()
    sync(dev)
    dist.barrier()
    del m, opt
    free(dev)
    if rank == 0:
        name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"
        r = analyze_trace(path, args.out, f"{p} x {name}")
        res["trace"] = r
        if r is None:
            print("\n== Trace: no GPU kernels recorded (CPU run). The trace file still shows the CPU side.")
        else:
            print("\n== Profiler trace of one DDP step, rank 0")
            print(f"step {r['step_ms']:.1f} ms on the GPU: forward {r['forward_ms']:.1f}, backward {r['backward_ms']:.1f}, optimizer {r['optimizer_ms']:.1f} ms")
            print(f"{r['nccl_kernels']} NCCL all-reduce kernels, {r['nccl_ms']:.1f} ms in total; compute kernels {r['compute_ms']:.1f} ms")
            print(f"NCCL under compute (hidden) {r['nccl_hidden_ms']:.1f} ms, NCCL alone (exposed) {r['nccl_exposed_ms']:.1f} ms")
            print(f"figure: {os.path.join(args.out, 'trace_step.png')}, trace: {path}")


# ---------------------------------------------------------------- main
def main():
    args = parse()
    dev, local = setup(args)
    rank, p = dist.get_rank(), dist.get_world_size()
    os.makedirs(args.out, exist_ok=True)
    res = dict(world_size=p, torch=torch.__version__, args=vars(args))
    if dev.type == "cuda":
        v = torch.cuda.nccl.version()
        res.update(gpu=torch.cuda.get_device_name(dev), cuda=torch.version.cuda,
                   nccl=".".join(map(str, v)) if isinstance(v, tuple) else str(v))
    log(rank, f"world size {p}, {res.get('gpu', 'CPU (gloo)')}, torch {torch.__version__}"
              + (f", CUDA {res['cuda']}, NCCL {res['nccl']}" if dev.type == "cuda" else ""))

    parts = ["scaling", "buckets", "nosync", "trace"] if args.part == "all" else [args.part]
    for part in parts:
        globals()["part_" + part](args, dev, local, rank, p, res)

    if rank == 0:
        with open(os.path.join(args.out, f"results_{args.part}.json"), "w") as f:
            json.dump(res, f, indent=2)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
