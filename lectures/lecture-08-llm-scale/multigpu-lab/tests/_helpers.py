"""Shared helpers for the Lecture 8 tests. Run from the multigpu-lab folder:  python -m pytest tests -q"""
import json
import os
import subprocess
import sys

import pytest
import torch

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, SRC)


def n_gpus():
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def needs_gpus(k):
    return pytest.mark.skipif(n_gpus() < k, reason=f'needs {k} GPUs, found {n_gpus()}')


def torchrun(script, args, gpus, timeout=900):
    """Run a src/ script under torchrun and return (returncode, stdout, stderr)."""
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={gpus}',
           os.path.join(SRC, script), *args]
    env = {**os.environ, 'OMP_NUM_THREADS': '2'}
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=SRC, env=env)
    return p.returncode, p.stdout, p.stderr


def result_line(stdout):
    for line in stdout.splitlines():
        if line.startswith('RESULT '):
            return json.loads(line[7:])
    return None
