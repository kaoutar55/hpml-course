# collectives.py: the collectives from the lecture, on CPU with the gloo backend.
# Run: torchrun --standalone --nproc_per_node=4 collectives.py
import torch
import torch.distributed as dist


def show(title, t):
    """Gather every rank's tensor on rank 0 and print them in rank order."""
    rows = [None] * dist.get_world_size()
    dist.all_gather_object(rows, t.tolist())
    if dist.get_rank() == 0:
        print(f"\n{title}")
        for r, row in enumerate(rows):
            print(f"  rank {r}: {row}")


def main():
    dist.init_process_group("gloo")
    rank, p = dist.get_rank(), dist.get_world_size()

    # Same data as the simulation lab: rank k holds (k+1) * [1, 2, 3, 4]
    def start():
        return torch.arange(1, p + 1, dtype=torch.float32) * (rank + 1)

    x = start()
    show("before (every collective starts from this)", x)

    x = start(); dist.broadcast(x, src=0)
    show("broadcast from rank 0: everyone gets rank 0's vector", x)

    x = start(); dist.reduce(x, dst=0, op=dist.ReduceOp.SUM)
    show("reduce to rank 0: only rank 0's result is defined (others keep partial values)", x)

    x = start(); dist.all_reduce(x, op=dist.ReduceOp.SUM)
    show("all-reduce SUM: every rank gets the chunk-by-chunk sum", x)

    x = start(); dist.all_reduce(x, op=dist.ReduceOp.SUM); x /= p
    show("all-reduce SUM then divide by p: the average, which is what DDP needs", x)

    # reduce-scatter: rank k keeps chunk k of the sum (one number per rank here)
    x = start()
    out = torch.zeros(1)
    try:
        dist.reduce_scatter(out, list(x.chunk(p)), op=dist.ReduceOp.SUM)
        how = "dist.reduce_scatter"
    except (RuntimeError, NotImplementedError):
        # Older gloo builds do not implement reduce_scatter. Same result, more bytes.
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        out = x.chunk(p)[rank].clone()
        how = "all_reduce + slice (this gloo build has no reduce_scatter)"
    show(f"reduce-scatter ({how}): rank k keeps chunk k of the sum", out)

    # all-gather: every rank contributes its chunk, everyone gets all chunks
    parts = [torch.zeros(1) for _ in range(p)]
    dist.all_gather(parts, out)
    show("all-gather of those chunks: back to the full all-reduce result", torch.cat(parts))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
