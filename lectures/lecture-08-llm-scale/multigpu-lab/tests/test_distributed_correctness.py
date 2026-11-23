"""Data layout and data-parallel correctness.

CPU: the data split. Data-parallel ranks get disjoint slices that together are the global batch,
every rank of a TP/CP/PP group gets the same slice, and the data is the same in every run.
GPU: DDP, FSDP2 (ZeRO-3 and ZeRO-2-like) and HSDP reproduce the one-GPU reference step in FP32."""
import json

import torch

from _helpers import needs_gpus, torchrun
import train as T


def test_dp_slices_cover_the_global_batch_exactly_once():
    vocab, seq, gbatch = 100, 16, 8
    full, _ = T.batch_for_step(3, vocab, seq, gbatch, 0, 1, 'cpu')
    for dp in (1, 2, 4, 8):
        parts = [T.batch_for_step(3, vocab, seq, gbatch, r, dp, 'cpu')[0] for r in range(dp)]
        assert all(p.shape == (gbatch // dp, seq) for p in parts)
        assert torch.equal(torch.cat(parts), full)


def test_data_is_deterministic_and_targets_are_shifted_inputs():
    a_in, a_tg = T.batch_for_step(5, 50, 8, 4, 1, 2, 'cpu')
    b_in, b_tg = T.batch_for_step(5, 50, 8, 4, 1, 2, 'cpu')
    assert torch.equal(a_in, b_in) and torch.equal(a_tg, b_tg)
    assert torch.equal(a_in[:, 1:], a_tg[:, :-1])


def test_pool_repeats_every_four_steps_at_global_batch_16():
    # The default data cycles a pool of 64 sequences: with 16 per step, step k and k + 4 are identical.
    # This is why the loss of the 8B runs drops sharply at step 4 (first repeat of seen data).
    x0, _ = T.batch_for_step(0, 32000, 8, 16, 0, 1, 'cpu')
    x4, _ = T.batch_for_step(4, 32000, 8, 16, 0, 1, 'cpu')
    x1, _ = T.batch_for_step(1, 32000, 8, 16, 0, 1, 'cpu')
    assert torch.equal(x0, x4) and not torch.equal(x0, x1)


def test_default_flags_reproduce_the_original_run_settings():
    a = T.get_args(['--mode', 'fsdp'])
    assert (a.dtype, a.clip_grad, a.vocab_parallel_loss, a.profile, a.mem_regions, a.grad_report) == \
        ('bf16', 0.0, 0, 0, '', '')
    # one deliberate change of default: TP with dp = 1 now uses FSDP2's BF16 policy (first run: --tp-dp1-autocast 1)
    assert a.tp_dp1_autocast == 0


@needs_gpus(4)
def test_data_parallel_layouts_match_reference_fp32(tmp_path):
    out = tmp_path / 'nv.json'
    rc, so, se = torchrun('numerical_validation.py', ['--model', 'tiny', '--seq', '128', '--global-batch', '8',
                          '--dtype', 'fp32', '--modes', 'ddp,fsdp,fsdp_z2,hsdp', '--out', str(out)], gpus=4)
    assert rc == 0, se[-3000:]
    r = json.load(open(out))['modes']
    for name, m in r.items():
        assert 'error' not in m, (name, m)
        assert m['loss_abs_diff'] < 1e-5 and m['grad_rel_l2'] < 1e-5, (name, m)
