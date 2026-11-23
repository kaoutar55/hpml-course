"""Measurement helpers for the Lecture 8 lab: step-time statistics and GPU memory attribution.

Timing method (the same in every run of this lab): torch.cuda.synchronize() and a barrier before each
step, synchronize after it, and the slowest rank's time (all-reduce MAX). A step takes as long as
its slowest GPU. Here we also keep every timed step, so a run reports the median and spread, not
only the mean.
"""
import collections
import json
import math
import os

import torch


def time_stats(times_s):
    """Summary of the timed (post-warm-up) steps, in milliseconds."""
    if not times_s:
        nan = float('nan')
        return dict(n_timed=0, step_ms=nan, step_ms_median=nan, step_ms_p10=nan, step_ms_p90=nan,
                    step_ms_min=nan, step_ms_max=nan, step_ms_std=nan)
    ms = sorted(t * 1e3 for t in times_s)
    n = len(ms)

    def q(p):                                   # linear interpolation between order statistics
        x = p * (n - 1)
        lo, hi = math.floor(x), math.ceil(x)
        return ms[lo] + (ms[hi] - ms[lo]) * (x - lo)

    mean = sum(ms) / n
    std = (sum((m - mean) ** 2 for m in ms) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return dict(n_timed=n, step_ms=round(mean, 1), step_ms_median=round(q(0.5), 1), step_ms_p10=round(q(0.1), 1),
                step_ms_p90=round(q(0.9), 1), step_ms_min=round(ms[0], 1), step_ms_max=round(ms[-1], 1),
                step_ms_std=round(std, 2))


def gb(x):
    return round(x / 2**30, 3)


class MemoryRegions:
    """Records torch.cuda.memory_allocated() at named points of ONE training step.

    Forward hooks on the embedding, each block, the final norm and the output layer show how the
    memory held for backward (saved activations) grows through the forward pass. mark() adds
    points such as 'after loss' and 'after backward'. Numbers are per GPU, in GiB, for this rank.
    Under FSDP the numbers include gathered weights that are alive at that moment."""

    def __init__(self, model, enable=False):
        self.points, self.on, self.handles = [], False, []
        if not enable:
            return
        for name, mod in model.named_modules():
            short = name.replace('_checkpoint_wrapped_module.', '')
            if short in ('tok_embeddings', 'norm', 'output') or (short.startswith('layers.') and short.count('.') == 1):
                self.handles.append(mod.register_forward_hook(self._hook(short)))

    def _hook(self, name):
        def f(mod, inp, out):
            if self.on:
                self.points.append((f'fwd {name}', gb(torch.cuda.memory_allocated())))
        return f

    def start(self):
        self.on, self.points = True, []
        torch.cuda.synchronize()
        self.points.append(('step start', gb(torch.cuda.memory_allocated())))

    def mark(self, name):
        if self.on:
            self.points.append((name, gb(torch.cuda.memory_allocated())))

    def stop(self):
        self.on = False
        for h in self.handles:
            h.remove()
        self.handles = []


# ---------------------------------------------------------------- allocator snapshots
def start_history(max_entries=200000):
    """Record the Python stack of every CUDA allocation (for diagnostics only: slow)."""
    torch.cuda.memory._record_memory_history(max_entries=max_entries, stacks='python')


def stop_history():
    torch.cuda.memory._record_memory_history(enabled=None)


OUR_FILES = ('hpml_llm.py', 'hpml_parallel.py', 'train.py', 'hf_train.py')


def _site(frames):
    """Name an allocation by the first frame in the lab's own files, plus the nearest torch frame."""
    ours = next((f for f in frames if os.path.basename(f.get('filename', '')) in OUR_FILES), None)
    torch_fn = next((f.get('name') for f in frames if 'torch' in f.get('filename', '')), '?')
    if ours is None:
        return f'(no lab frame) {torch_fn}'
    return f"{os.path.basename(ours['filename'])}:{ours['line']} {ours.get('name', '')} | {torch_fn}"


def snapshot_summary(top=25):
    """Group the live (allocated) blocks of the current allocator state by allocation site.
    Returns the total, the reserved size, and the largest groups in GiB."""
    snap = torch.cuda.memory._snapshot()
    groups = collections.defaultdict(lambda: [0, 0])
    total = reserved = 0
    for seg in snap['segments']:
        if seg.get('device', 0) != torch.cuda.current_device():
            continue
        reserved += seg['total_size']
        for b in seg['blocks']:
            if b['state'] != 'active_allocated':
                continue
            total += b['size']
            g = groups[_site(b.get('frames', []))]
            g[0] += b['size']
            g[1] += 1
    rows = sorted(groups.items(), key=lambda kv: -kv[1][0])[:top]
    return dict(allocated_gb=gb(total), reserved_gb=gb(reserved),
                top=[dict(site=k, gb=gb(v[0]), blocks=v[1]) for k, v in rows])


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)
