"""Read torch.profiler Chrome traces (.json or .json.gz) and summarize what the GPU did.

Kernel classes, from the kernel name:
  nccl       collective or point-to-point communication kernels (AllGather, ReduceScatter, AllReduce, SendRecv)
  matmul     GEMM kernels (cuBLAS, CUTLASS, nvjet)
  attention  fused attention kernels (flash / memory-efficient / cuDNN SDPA)
  other      everything else: elementwise, norms, softmax, optimizer, copies

Time is measured with interval unions, never by adding kernel durations, because kernels on
different streams overlap. For a window of the trace:
  compute_busy_ms   time when at least one non-NCCL kernel was running
  nccl_busy_ms      time when at least one NCCL kernel was running
  overlap_ms        time when both were running
  exposed_nccl_ms   nccl_busy - overlap: communication that nothing else was hiding
  idle_ms           window - (compute_busy union nccl_busy)
Limits: an NCCL kernel's duration includes the time it waits for the other GPUs to arrive, so it is
an upper bound on transfer time, not a bandwidth measurement. Bytes moved are not in the trace.
"""
import gzip
import json
import re
from collections import Counter, defaultdict


def load(path):
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt') as f:
        return json.load(f)


def kernel_class(name):
    n = name.lower()
    if 'nccl' in n:
        return 'nccl'
    if any(k in n for k in ('flash', 'fmha', 'attention', 'sdpa', 'cudnn_generated_fort_native_sdpa')):
        return 'attention'
    if any(k in n for k in ('gemm', 'cutlass', 'nvjet', 'xmma', 'cublas', 'sm90_', 'matmul')):
        return 'matmul'
    return 'other'


def nccl_op(name):
    for k in ('AllGather', 'ReduceScatter', 'AllReduce', 'SendRecv', 'Broadcast', 'AllToAll', 'Send', 'Recv'):
        if k.lower() in name.lower():
            return k
    return 'nccl other'


def _union(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _length(iv):
    return sum(e - s for s, e in iv)


def _intersect(a, b):
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            out.append([s, e])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def events(trace):
    ev = trace['traceEvents'] if isinstance(trace, dict) else trace
    kernels = [e for e in ev if e.get('ph') == 'X' and e.get('cat') == 'kernel']
    annos = [e for e in ev if e.get('ph') == 'X' and e.get('cat') == 'gpu_user_annotation']
    return kernels, annos


def step_windows(annos):
    """GPU-side windows of the profiled steps, from the ProfilerStep#N annotations."""
    w = [(a['ts'], a['ts'] + a['dur'], a['name']) for a in annos if a['name'].startswith('ProfilerStep#')]
    return sorted(w)


def summarize(trace, window=None):
    kernels, annos = events(trace)
    if window is None:
        steps = step_windows(annos)
        if steps:
            window = (steps[0][0], steps[-1][1])
        else:
            window = (min(k['ts'] for k in kernels), max(k['ts'] + k['dur'] for k in kernels))
    t0, t1 = window
    ks = [k for k in kernels if k['ts'] < t1 and k['ts'] + k['dur'] > t0]
    clip = lambda k: [max(k['ts'], t0), min(k['ts'] + k['dur'], t1)]
    by_class = defaultdict(list)
    for k in ks:
        by_class[kernel_class(k['name'])].append(clip(k))
    comp = _union([iv for c, v in by_class.items() if c != 'nccl' for iv in v])
    comm = _union(by_class.get('nccl', []))
    ov = _intersect(comp, comm)
    busy = _union(comp + comm)
    win = t1 - t0
    nccl_counts = Counter(nccl_op(k['name']) for k in ks if kernel_class(k['name']) == 'nccl')
    regions = defaultdict(float)
    for a in annos:
        if not a['name'].startswith('ProfilerStep#') and a['ts'] < t1 and a['ts'] + a['dur'] > t0:
            regions[a['name']] += (min(a['ts'] + a['dur'], t1) - max(a['ts'], t0)) / 1e3
    n_steps = max(1, len(step_windows(annos)))
    ms = lambda us: round(us / 1e3, 3)
    return dict(
        window_ms=ms(win), n_steps=n_steps, n_kernels=len(ks),
        kernel_sum_ms={c: ms(_length(_union(v))) for c, v in sorted(by_class.items())},
        compute_busy_ms=ms(_length(comp)), nccl_busy_ms=ms(_length(comm)), overlap_ms=ms(_length(ov)),
        exposed_nccl_ms=ms(_length(comm) - _length(ov)), idle_ms=ms(win - _length(busy)),
        idle_frac=round((win - _length(busy)) / win, 3) if win else None,
        nccl_kernels=dict(nccl_counts), streams=len({(k['pid'], k['tid']) for k in ks}),
        gpu_regions_ms={k: round(v, 2) for k, v in sorted(regions.items())})


GRAYS = {'matmul': '0.15', 'attention': '0.40', 'other': '0.65', 'nccl': '0.85'}
HATCH = {'nccl': '////'}


def plot_timeline(trace, ax, window=None, title='', max_ms=None, label_regions=True):
    """Draw one GPU's kernels per stream, in grayscale, from the real trace timestamps."""
    kernels, annos = events(trace)
    if window is None:
        steps = step_windows(annos)
        window = (steps[0][0], steps[0][1]) if steps else (min(k['ts'] for k in kernels),
                                                            max(k['ts'] + k['dur'] for k in kernels))
    t0, t1 = window
    if max_ms:
        t1 = min(t1, t0 + max_ms * 1e3)
    ks = [k for k in kernels if k['ts'] < t1 and k['ts'] + k['dur'] > t0]
    streams = sorted({k['tid'] for k in ks}, key=lambda s: (str(s)))
    row = {s: i for i, s in enumerate(streams)}
    for c in ('matmul', 'attention', 'other', 'nccl'):
        for s in streams:
            iv = [(max(k['ts'], t0) / 1e3 - t0 / 1e3, (min(k['ts'] + k['dur'], t1) - max(k['ts'], t0)) / 1e3)
                  for k in ks if k['tid'] == s and kernel_class(k['name']) == c]
            if iv:
                ax.broken_barh(iv, (row[s] + 0.1, 0.8), facecolors=GRAYS[c], edgecolor='0.0' if c == 'nccl' else 'none',
                               linewidth=0.2, hatch=HATCH.get(c), label=c if s == streams[0] or c == 'nccl' else None)
    labels = [f'stream {s}' for s in streams]
    if label_regions:
        y = len(streams)
        for a in annos:
            if a['name'].startswith('ProfilerStep#') or not (a['ts'] < t1 and a['ts'] + a['dur'] > t0):
                continue
            s0, d = max(a['ts'], t0) / 1e3 - t0 / 1e3, (min(a['ts'] + a['dur'], t1) - max(a['ts'], t0)) / 1e3
            ax.broken_barh([(s0, d)], (y + 0.1, 0.8), facecolors='white', edgecolor='0.0', linewidth=0.6)
            if d > (t1 - t0) / 1e3 * 0.06:
                ax.text(s0 + d / 2, y + 0.5, a['name'], ha='center', va='center', fontsize=7)
        labels.append('regions')
    ax.set_yticks([i + 0.5 for i in range(len(labels))])
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0, (t1 - t0) / 1e3)
    ax.set_xlabel('time within the step (ms)')
    ax.set_title(title, fontsize=9)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=GRAYS[c], edgecolor='0.0' if c == 'nccl' else 'none', hatch=HATCH.get(c), label=c)
               for c in ('matmul', 'attention', 'other', 'nccl')]
    ax.legend(handles=handles, fontsize=7, loc='upper right', ncol=4, frameon=False, bbox_to_anchor=(1.0, 1.32))
