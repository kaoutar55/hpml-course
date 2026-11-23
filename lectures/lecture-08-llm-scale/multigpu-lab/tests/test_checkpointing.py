"""Distributed checkpointing (Part F).

1. Save after 3 steps on 2 GPUs and resume on 2 GPUs: the resumed steps must match an uninterrupted
   run, because weights, Adam state, the learning-rate schedule position (--start-step) and the data
   position are all restored. Same GPU count, so the same reduction order.
2. Resume the same checkpoint on 1 GPU (resharding 2 -> 1): it must load and continue close to the
   uninterrupted run. Exact equality is not expected: one GPU sums the gradient in a different order.
The model has no dropout, so there is no RNG state to restore."""
import json

from _helpers import needs_gpus, result_line, torchrun

COMMON = ['--mode', 'fsdp', '--model', 'tiny', '--seq', '128', '--global-batch', '8', '--warmup', '1']


@needs_gpus(2)
def test_save_resume_matches_uninterrupted_run(tmp_path):
    ck = str(tmp_path / 'ck')
    rc, so, se = torchrun('train.py', COMMON + ['--steps', '6', '--out', str(tmp_path / 'r.jsonl')], gpus=2)
    assert rc == 0, se[-3000:]
    ref = result_line(so)['losses']
    rc, so, se = torchrun('train.py', COMMON + ['--steps', '3', '--save', ck, '--out', str(tmp_path / 'r.jsonl')], gpus=2)
    assert rc == 0, se[-3000:]
    first = result_line(so)['losses']
    rc, so, se = torchrun('train.py', COMMON + ['--steps', '3', '--load', ck, '--start-step', '3',
                                                '--out', str(tmp_path / 'r.jsonl')], gpus=2)
    assert rc == 0, se[-3000:]
    resumed = result_line(so)['losses']
    assert max(abs(x - y) for x, y in zip(first, ref[:3])) < 1e-4, (first, ref[:3])   # run-to-run
    assert max(abs(x - y) for x, y in zip(resumed, ref[3:])) < 2e-3, (resumed, ref[3:])

    rc, so, se = torchrun('train.py', COMMON + ['--steps', '3', '--load', ck, '--start-step', '3',
                                                '--out', str(tmp_path / 'r.jsonl')], gpus=1)
    assert rc == 0, se[-3000:]
    one_gpu = result_line(so)['losses']
    assert max(abs(x - y) for x, y in zip(one_gpu, ref[3:])) < 5e-2, (one_gpu, ref[3:])
