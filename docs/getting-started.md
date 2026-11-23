# Getting started

Every notebook in this course runs on a free service. Lectures 3 and 4 and most of the Lecture 7 and 8 demos run on Google Colab. The parts that need two GPUs run on Kaggle. The Lecture 8 multi-GPU lab needs an 8-GPU node, so for that one you read the executed copy. The table in the [main README](../README.md#open-the-notebooks-in-colab) says which notebook needs what.

## 1. Google Colab: one GPU

Colab gives you a free notebook with one GPU, usually an NVIDIA T4. Use it for Lectures 3 and 4, Part A of the Lecture 7 demo, and Part A and B1 of the Lecture 8 demo.

1. Open a notebook with its "Open in Colab" badge. The badges only work when this repository is public; if a badge shows an error, download the `.ipynb` file from GitHub and upload it in Colab with File, Upload notebook.
2. Choose Runtime, Change runtime type, select T4 GPU and save. Do this before you run anything, because changing the runtime type restarts the notebook and clears its state.
3. Check that you got a GPU. Run this in a cell:

   ```
   !nvidia-smi
   ```

   It should print a table with one Tesla T4. If it says the command failed or no devices were found, the runtime is still CPU-only.
4. Run the cells in order, from the top.

**What a free T4 can do.** It has 16 GB of memory (about 15 GB usable), FP16 tensor cores and enough speed for every single-GPU part of the course, including `torch.compile` and the profiler.

**What it cannot do.**
- Two-GPU parts. Colab gives one GPU, and DDP or FSDP across GPUs needs at least two. Use Kaggle for those.
- BF16. The T4 has no BF16 support, so the course notebooks use FP16 autocast or FP32 on it. The ideas are the same; the bytes per value change.
- Long sessions. Colab disconnects idle notebooks and limits free GPU time. When it disconnects, the files you wrote and the variables you set are gone, so run the notebook again from the top.
- Numbers that match the saved outputs. The saved copies were measured on A100 or H100 GPUs. Expect different times on a T4; the direction of each result should be the same.

## 2. Kaggle: two T4 GPUs

Kaggle notebooks offer a free "GPU T4 x2" accelerator: two T4 GPUs in one machine. Use it for Part B of the Lecture 7 demo and B2 of the Lecture 8 demo.

1. Sign in to https://www.kaggle.com and verify your phone number in your account settings. Kaggle only turns on GPUs and internet access for verified accounts.
2. Create a new notebook (Create, New Notebook). Then use File, Import Notebook to upload the `.ipynb` file you downloaded from this repository.
3. Open the notebook's settings panel on the right. Set Accelerator to "GPU T4 x2".
4. In the same panel, turn Internet on. The notebooks install packages and some download data, which fails without it.
5. Check the setup with the snippet in section 3. It should report 2 GPUs.

Kaggle limits GPU time per week; the remaining quota is shown in the settings panel. Turn the accelerator off, or stop the session, when you are done.

The two T4s are connected over PCIe, not NVLink. Communication between them is much slower than on the H100 nodes behind the saved outputs, which is why the DDP and FSDP timings look different.

## 3. Check your setup

Run this cell after you choose the runtime. It tells you the PyTorch version, how many GPUs the notebook sees and whether `torchrun` is on the path.

```python
import shutil
import torch

print("PyTorch", torch.__version__)
print("CUDA available:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}, {p.total_memory / 2**30:.1f} GiB")
print("torchrun:", shutil.which("torchrun") or "not found, use python -m torch.distributed.run")
```

On Colab with a T4 you should see one GPU. On Kaggle with "GPU T4 x2" you should see two.

## 4. Troubleshooting

**No GPU found.** `torch.cuda.is_available()` prints `False`, or `nvidia-smi` fails. The runtime is CPU-only, which is the default. On Colab choose Runtime, Change runtime type, T4 GPU; on Kaggle set the Accelerator in the settings panel. If Colab says no GPU is available, you have reached the free usage limit for now: try again later, or use Kaggle.

**CUDA out of memory.** The GPU does not have room for the next allocation. This happens when tensors from earlier cells are still held, or when a run is too large for 15 GB. Restart the runtime and run the notebook from the top, so that only what the current cell needs is in memory. If it still fails, lower the batch size or sequence length flags of that cell; the notebooks name them.

**The notebook restarts or disconnects.** Colab and Kaggle restart a session when it uses all of the machine's CPU memory, when it is idle too long, or when you change the runtime type. All variables are lost, and after a disconnect the files the notebook wrote are lost too. Run the notebook again from the top, because later cells depend on what earlier cells created.

**`torchrun: command not found`.** The `torchrun` command is a small launcher that pip installs next to PyTorch, and in some environments its folder is not on the shell's path. Run the same launcher through Python instead: replace `torchrun` with `python -m torch.distributed.run`, keeping every other argument.

**NCCL error on a machine with one GPU**, for example "Duplicate GPU detected". NCCL, the GPU communication library behind DDP and FSDP, needs one GPU per process. Launching 2 processes with `--nproc_per_node=2` on a single GPU puts both on the same device, and NCCL refuses. Run that part on Kaggle with "GPU T4 x2". In the Lecture 7 demo, Part B also falls back to a CPU version with the `gloo` backend when it finds fewer than two GPUs; its timings mean nothing, but the mechanics are the same.

## 5. Simulations

Open `simulations/index.html` in a browser. They need no install and work offline.

## 6. Your own machine

You need Python 3.10 or newer and a recent PyTorch with CUDA. The notebooks install anything else they need in their first cells. Check your install with:

```
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
```
