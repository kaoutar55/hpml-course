"""Tensor parallelism computes the same function as the unsharded model.

CPU tests (2 processes, gloo, FP32): the TP plans of hpml_parallel.apply_tp, with and without sequence
parallelism and with the vocabulary-parallel loss, reproduce the reference loss and every gradient.
GPU tests (torchrun): the same check through train.py's real code path, plus TP8 layouts."""
import json
import os
import tempfile

import pytest
import torch
import torch.multiprocessing as mp

from _helpers import SRC, needs_gpus, torchrun


def _tp_worker(rank, world, init_file, sp, vpl, q):
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor

    import hpml_parallel as hp
    from hpml_llm import build_model, get_config
    dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank, world_size=world)
    try:
        cfg = get_config('tiny', 32)
        g = torch.Generator().manual_seed(1)
        inp = torch.randint(0, cfg.vocab, (2, 32), generator=g)
        tgt = torch.randint(0, cfg.vocab, (2, 32), generator=g)

        ref = build_model(cfg)
        loss_ref = F.cross_entropy(ref(inp).flatten(0, 1), tgt.flatten(0, 1))
        loss_ref.backward()

        m = build_model(cfg)
        mesh = init_device_mesh('cpu', (world,), mesh_dim_names=('tp',))
        hp.apply_tp(m, mesh, sequence_parallel=sp, vocab_parallel_loss=vpl)
        with hp.loss_context(vpl):
            logits = m(inp)
            if not vpl:
                assert not isinstance(logits, DTensor) and logits.shape == (2, 32, cfg.vocab)
            loss = F.cross_entropy(logits.flatten(0, 1), tgt.flatten(0, 1))
            loss.backward()
        loss = loss.full_tensor() if isinstance(loss, DTensor) else loss
        worst = 0.0
        refp = dict(ref.named_parameters())
        for n, p in m.named_parameters():
            gp = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
            rel = ((gp - refp[n].grad).norm() / refp[n].grad.norm()).item()
            worst = max(worst, rel)
        if rank == 0:
            q.put((abs(loss.item() - loss_ref.item()), worst))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('sp,vpl', [(True, False), (False, False), (True, True), (False, True)])
def test_tp2_matches_reference_on_cpu(sp, vpl):
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    with tempfile.TemporaryDirectory() as d:
        init = os.path.join(d, 'init')
        procs = [ctx.Process(target=_tp_worker, args=(r, 2, init, sp, vpl, q)) for r in range(2)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(300)
        assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    dloss, worst = q.get(timeout=5)
    assert dloss < 1e-5, dloss
    assert worst < 1e-4, worst          # FP32, different summation order only


@needs_gpus(2)
def test_train_py_tp_layouts_match_reference_fp32(tmp_path):
    out = tmp_path / 'nv.json'
    rc, so, se = torchrun('numerical_validation.py', ['--model', 'tiny', '--seq', '128', '--global-batch', '4',
                          '--dtype', 'fp32', '--modes', 'tp2,tp2_nosp,tp2_vpl,cp2', '--out', str(out)], gpus=2)
    assert rc == 0, se[-3000:]
    r = json.load(open(out))['modes']
    for name, m in r.items():
        assert 'error' not in m, (name, m)
        assert m['loss_abs_diff'] < 1e-5 and m['grad_rel_l2'] < 1e-5, (name, m)


@needs_gpus(8)
def test_tp8_layouts_match_reference_fp32(tmp_path):
    out = tmp_path / 'nv8.json'
    # 350M has 16 query heads and 8 K, V heads, so it splits over 8 TP ranks (tiny has only 2 K, V heads)
    rc, so, se = torchrun('numerical_validation.py', ['--model', '350M', '--seq', '128', '--global-batch', '8',
                          '--dtype', 'fp32', '--modes', 'tp8,tp8_nosp', '--out', str(out)], gpus=8)
    assert rc == 0, se[-3000:]
    for name, m in json.load(open(out))['modes'].items():
        assert 'error' not in m, (name, m)
        assert m['grad_rel_l2'] < 1e-5, (name, m)
