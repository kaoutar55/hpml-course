"""Opt-in, selective profiling for the Lecture 8 lab.

Profiling is OFF unless a run passes enable=True (train.py: --profile 1). When it is on:
  * only the ranks in `ranks` record (rank 0 by default; pass several ranks only when you need to see
    how ranks wait for each other, for example pipeline bubbles);
  * the torch.profiler schedule skips `wait` steps, warms up for `warmup` steps and records `active`
    steps (1 to 3 is plenty: one H100 step of an 8B model is tens of thousands of kernels);
  * each recorded rank writes one Chrome-format trace, <out_dir>/rank<r>.json.gz, plus manifest.json
    with the run's metadata. Open the trace at https://ui.perfetto.dev (drag and drop the file).

Profiling changes what it measures: the CPU side records every op launch, the CUDA side uses CUPTI,
and record_shapes / profile_memory / with_stack add more overhead and much larger files. Step times
measured during a profiled run are NOT benchmark numbers; train.py writes them to a separate file and
marks them profiled=True.
"""
import contextlib
import datetime
import glob
import gzip
import json
import os
import shutil

import torch


def code_revision(paths):
    """SHA-256 of the source files that define a run, so results can be tied to the exact code."""
    import hashlib
    h = hashlib.sha256()
    for p in sorted(paths):
        with open(p, 'rb') as f:
            h.update(os.path.basename(p).encode() + b'\0' + f.read())
    return h.hexdigest()[:16]


def environment():
    env = dict(torch=torch.__version__, cuda=torch.version.cuda)
    try:
        env['nccl'] = '.'.join(map(str, torch.cuda.nccl.version()))
    except Exception:
        env['nccl'] = None
    if torch.cuda.is_available():
        env['gpu'] = torch.cuda.get_device_name()
        env['gpu_mem_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    return env


class StepProfiler:
    """Wrap a training loop:

        prof = StepProfiler(enable=args.profile, rank=rank, ranks=[0], out_dir='traces/C/tp2')
        with prof:
            for step in range(n):
                with prof.region('forward'):
                    ...
                prof.step()          # once per training step, after the optimizer update

    When disabled, every method is a no-op and region() returns a null context, so the training loop
    is unchanged."""

    def __init__(self, enable=False, rank=0, ranks=(0,), out_dir='profiles/run', wait=1, warmup=1, active=2,
                 record_shapes=False, profile_memory=False, with_stack=False, max_trace_mb=300, max_traces=16,
                 metadata=None):
        self.rank = rank
        self.on = bool(enable) and rank in set(ranks)
        self.enabled_somewhere = bool(enable)
        self.out_dir, self.max_trace_mb, self.max_traces = out_dir, max_trace_mb, max_traces
        self.cfg = dict(wait=wait, warmup=warmup, active=active, record_shapes=record_shapes,
                        profile_memory=profile_memory, with_stack=with_stack, ranks=sorted(set(ranks)))
        self.metadata = dict(metadata or {})
        self.prof, self.trace_path, self.note = None, None, None
        if self.on:
            os.makedirs(out_dir, exist_ok=True)
            n_existing = len(glob.glob(os.path.join(out_dir, '..', '*', 'rank*.json.gz')))
            if n_existing >= max_traces:
                # a budget, so a loop over many configurations cannot fill the disk by accident
                self.on, self.note = False, f'skipped: {n_existing} traces already under {os.path.dirname(out_dir)}'

    # ---- context manager
    def __enter__(self):
        if not self.on:
            return self
        from torch.profiler import ProfilerActivity, profile, schedule
        self.prof = profile(
            activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else []),
            schedule=schedule(wait=self.cfg['wait'], warmup=self.cfg['warmup'], active=self.cfg['active'], repeat=1),
            on_trace_ready=self._export,
            record_shapes=self.cfg['record_shapes'], profile_memory=self.cfg['profile_memory'],
            with_stack=self.cfg['with_stack'])
        self.prof.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.prof is not None:
            try:
                self.prof.__exit__(exc_type, exc, tb)     # stops CUPTI even if the step raised (for example OOM)
            finally:
                self.prof = None
        if self.on:
            self._write_manifest(error=None if exc_type is None else f'{exc_type.__name__}: {exc}')
        return False

    def step(self):
        if self.prof is not None:
            self.prof.step()

    def region(self, name):
        """A named range that shows up on the CPU timeline and, as gpu_user_annotation, on the GPU timeline."""
        if not self.enabled_somewhere:
            return contextlib.nullcontext()
        return torch.profiler.record_function(name)

    # ---- output
    def _export(self, prof):
        raw = os.path.join(self.out_dir, f'rank{self.rank}.json')
        prof.export_chrome_trace(raw)
        gz = raw + '.gz'
        with open(raw, 'rb') as f, gzip.open(gz, 'wb', compresslevel=6) as g:
            shutil.copyfileobj(f, g)
        os.remove(raw)
        mb = os.path.getsize(gz) / 2**20
        if mb > self.max_trace_mb:
            os.remove(gz)
            self.note = f'trace was {mb:.0f} MB compressed, over the {self.max_trace_mb} MB budget, deleted'
        else:
            self.trace_path = gz

    def _write_manifest(self, error=None):
        man = dict(self.metadata, profiler=self.cfg, rank=self.rank, trace=self.trace_path, note=self.note,
                   error=error, written_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   environment=environment(),
                   viewer='open the .json.gz trace at https://ui.perfetto.dev')
        with open(os.path.join(self.out_dir, f'manifest_rank{self.rank}.json'), 'w') as f:
            json.dump(man, f, indent=1)
