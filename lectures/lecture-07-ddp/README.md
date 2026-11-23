# Lecture 7: distributed data parallelism

`demo/` holds the demo notebook, an executed copy and the scripts the notebook writes and runs:

- `HPML_2026_L7_Demo.ipynb`: the notebook to run. Part A runs the five collectives, a ring all-reduce written by hand and a fit of the alpha-beta cost model. Part B trains with DDP on two GPUs: scaling, bucket size, `no_sync` and a profiler trace of one step.
- `HPML_2026_L7_Demo_executed.ipynb`: the same notebook run on 2 H100 GPUs. All outputs saved, nothing to run.
- `collectives.py`, `ring_allreduce.py`, `alpha_beta.py`, `ddp_demo.py`: the scripts the notebook writes with `%%writefile`. Each one starts with the command that runs it.
- `modal_ddp.py`: an optional way to run the demo on 2 H100 GPUs on Modal, a paid cloud service. You do not need it for the course.

**Run first.** Open the demo notebook and run Part A from the top. The browser simulation in [../../simulations/lecture-7-ddp/](../../simulations/lecture-7-ddp/) shows the same collectives step by step.

**Hardware.** Part A runs on CPU, with 4 processes talking through the `gloo` backend, so it works on Colab. Part B needs 2 GPUs. Colab gives one, so use Kaggle's free "GPU T4 x2" accelerator for Part B ([setup](../../docs/getting-started.md#2-kaggle-two-t4-gpus)). On Colab, Part B falls back to a small CPU version: the bucket counts and `no_sync` behaviour are still correct, but ignore the timings.
