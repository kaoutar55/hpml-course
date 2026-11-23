# alpha_beta.py: measure t(n) for all-reduce and fit t = a + n / b.
# Run: torchrun --standalone --nproc_per_node=4 alpha_beta.py            (CPU, gloo)
#      torchrun --standalone --nproc_per_node=2 alpha_beta.py --cuda     (GPUs, nccl)
import argparse
import time
import torch
import torch.distributed as dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()
    if args.cuda:
        import os
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dist.init_process_group("nccl")
        dev = torch.device("cuda", local)
    else:
        dist.init_process_group("gloo")
        dev = torch.device("cpu")
    rank, p = dist.get_rank(), dist.get_world_size()

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    sizes = [2 ** k for k in range(10, 28, 2)]      # 1 KiB .. 128 MiB
    rows = []
    for nbytes in sizes:
        x = torch.ones(nbytes // 4, device=dev)
        reps = 20 if nbytes <= 2 ** 22 else 5
        for _ in range(3):                         # warm-up
            dist.all_reduce(x)
        sync(); dist.barrier()
        t0 = time.perf_counter()
        for _ in range(reps):
            dist.all_reduce(x)
        sync()
        t = torch.tensor([(time.perf_counter() - t0) / reps], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)  # the slowest rank defines the time
        rows.append((nbytes, t.item()))

    if rank == 0:
        backend = "nccl" if args.cuda else "gloo (CPU)"
        print(f"all-reduce on p = {p}, {backend}")
        print(f"{'size':>10} {'time':>10} {'algbw GB/s':>11} {'busbw GB/s':>11}")
        for n, t in rows:
            algbw = n / t / 1e9
            print(f"{n / 1024:>8.0f} KiB {t * 1e3:>8.3f} ms {algbw:>11.2f} {algbw * 2 * (p - 1) / p:>11.2f}")
        # Least-squares fit of t = a + n / b
        n_ = torch.tensor([r[0] for r in rows], dtype=torch.float64)
        t_ = torch.tensor([r[1] for r in rows], dtype=torch.float64)
        A = torch.stack([torch.ones_like(n_), n_], 1) / t_.unsqueeze(1)   # relative error, so small sizes count
        coef = torch.linalg.lstsq(A, torch.ones_like(t_).unsqueeze(1)).solution.squeeze()
        a, inv_b = coef[0].item(), coef[1].item()
        print(f"\nfit: a (latency) = {a * 1e6:.1f} us, b (bandwidth) = {1 / inv_b / 1e9:.2f} GB/s")
        print(f"crossover n* = a * b = {a / inv_b / 1e3:.0f} KB")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
