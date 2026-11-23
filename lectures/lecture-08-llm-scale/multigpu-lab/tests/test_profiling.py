"""Profiling must be off by default, honor the rank selection, write Chrome traces when asked,
and the trace analysis must use interval unions (no double counting of overlapping kernels)."""
import glob
import gzip
import json
import os

import torch

from _helpers import SRC  # noqa: F401  (puts src/ on sys.path)
import benchmark_utils as bu
import trace_analysis as ta
from profiling_utils import StepProfiler


def test_disabled_by_default_is_a_no_op(tmp_path):
    prof = StepProfiler(out_dir=str(tmp_path / 'p'))
    assert not prof.on and not prof.enabled_somewhere
    with prof:
        with prof.region('forward'):
            x = torch.ones(4) * 2
        prof.step()
    assert x.sum().item() == 8
    assert not os.path.exists(tmp_path / 'p')          # nothing written, not even the folder


def test_unselected_rank_records_nothing(tmp_path):
    prof = StepProfiler(enable=True, rank=3, ranks=[0], out_dir=str(tmp_path / 'p'))
    assert not prof.on
    with prof:
        prof.step()
    assert not glob.glob(str(tmp_path / 'p' / '*'))


def test_enabled_writes_gzip_chrome_trace_and_manifest(tmp_path):
    out = tmp_path / 'A' / 'fsdp'
    prof = StepProfiler(enable=True, rank=0, ranks=[0], out_dir=str(out), wait=1, warmup=1, active=1,
                        metadata=dict(mode='test'))
    w = torch.randn(64, 64)
    with prof:
        for _ in range(4):
            with prof.region('forward'):
                y = (w @ w).sum()
            prof.step()
    trace = out / 'rank0.json.gz'
    assert trace.exists()
    with gzip.open(trace, 'rt') as f:
        data = json.load(f)
    assert 'traceEvents' in data and any(e.get('name') == 'forward' for e in data['traceEvents'])
    man = json.load(open(out / 'manifest_rank0.json'))
    assert man['mode'] == 'test' and man['profiler']['active'] == 1 and man['trace'].endswith('.json.gz')


def test_trace_budget_skips_extra_traces(tmp_path):
    for i in range(2):
        d = tmp_path / 'A' / f'old{i}'
        d.mkdir(parents=True)
        (d / 'rank0.json.gz').write_bytes(b'x')
    prof = StepProfiler(enable=True, rank=0, ranks=[0], out_dir=str(tmp_path / 'A' / 'new'), max_traces=2)
    assert not prof.on and 'skipped' in prof.note


def test_interval_union_does_not_double_count():
    k = lambda ts, dur, name, tid=7: dict(ph='X', cat='kernel', ts=ts, dur=dur, name=name, pid=0, tid=tid)
    trace = dict(traceEvents=[
        dict(ph='X', cat='gpu_user_annotation', ts=0, dur=100, name='ProfilerStep#3', pid=0, tid=7),
        k(0, 40, 'sm90_xmma_gemm_bf16'), k(20, 40, 'ampere_gemm', tid=8),        # two matmuls overlap: busy 0..60
        k(50, 30, 'ncclDevKernel_AllGather_RING_LL', tid=9),                     # comm 50..80, overlaps 50..60
        k(90, 5, 'vectorized_elementwise_kernel'),
    ])
    s = ta.summarize(trace)
    assert s['window_ms'] == 0.1
    assert s['compute_busy_ms'] == 0.065          # 0..60 plus 90..95
    assert s['nccl_busy_ms'] == 0.03
    assert s['overlap_ms'] == 0.01
    assert s['exposed_nccl_ms'] == 0.02
    assert s['idle_ms'] == 0.015                  # 80..90 and 95..100
    assert s['nccl_kernels'] == {'AllGather': 1}


def test_time_stats_median_and_spread():
    st = bu.time_stats([0.010, 0.012, 0.011, 0.030])
    assert st['n_timed'] == 4 and st['step_ms_median'] == 11.5 and st['step_ms_max'] == 30.0
    assert bu.time_stats([])['n_timed'] == 0
