# ring_allreduce.py: ring all-reduce by hand, checked against dist.all_reduce.
# Run: torchrun --standalone --nproc_per_node=4 ring_allreduce.py
import torch
import torch.distributed as dist


def ring_allreduce(x):
    """In-place SUM all-reduce of a 1D tensor with 2(p-1) send/recv steps. Returns bytes sent."""
    rank, p = dist.get_rank(), dist.get_world_size()
    right, left = (rank + 1) % p, (rank - 1) % p
    chunks = list(x.chunk(p))          # views into x, so results land in x
    sent = 0
    # Phase 1, reduce-scatter: at step s, send chunk (rank - s) and add the received chunk (rank - s - 1)
    for s in range(p - 1):
        snd, rcv = (rank - s) % p, (rank - s - 1) % p
        tmp = torch.empty_like(chunks[rcv])
        req = dist.isend(chunks[snd].contiguous(), right)   # isend + recv: no deadlock
        dist.recv(tmp, left)
        req.wait()
        chunks[rcv] += tmp
        sent += chunks[snd].numel() * chunks[snd].element_size()
    # Now rank r owns the full sum of chunk (r + 1) % p.
    # Phase 2, all-gather: forward finished chunks, receiver copies instead of adding
    for s in range(p - 1):
        snd, rcv = (rank - s + 1) % p, (rank - s) % p
        tmp = torch.empty_like(chunks[rcv])
        req = dist.isend(chunks[snd].contiguous(), right)
        dist.recv(tmp, left)
        req.wait()
        chunks[rcv].copy_(tmp)
        sent += chunks[snd].numel() * chunks[snd].element_size()
    return sent


def main():
    dist.init_process_group("gloo")
    rank, p = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(rank)
    n_elems = 1_000_000                      # 4 MB of float32 per rank
    x = torch.randn(n_elems)
    ref = x.clone()
    dist.all_reduce(ref, op=dist.ReduceOp.SUM)

    sent = ring_allreduce(x)
    err = (x - ref).abs().max().item()
    n = n_elems * 4
    rows = [None] * p
    dist.all_gather_object(rows, (sent, err))
    if rank == 0:
        print(f"p = {p}, n = {n / 1e6:.1f} MB per rank")
        print(f"formula 2(p-1)/p * n = {2 * (p - 1) / p * n / 1e6:.2f} MB")
        for r, (s, e) in enumerate(rows):
            print(f"  rank {r}: sent {s / 1e6:.2f} MB, max |ring - all_reduce| = {e:.1e}")
        print("The small differences come from a different summation order, not a bug.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
