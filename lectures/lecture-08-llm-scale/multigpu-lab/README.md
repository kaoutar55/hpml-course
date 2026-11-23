# Lecture 8 multi-GPU lab

One Llama-style model trained on 8 GPUs with each technique from the lecture: DDP, FSDP2, activation checkpointing, tensor and sequence parallelism, pipeline schedules, context parallelism, distributed checkpointing, a Hugging Face model, and TorchTitan.

- `HPML_2026_L8_MultiGPU_Lab.ipynb`: the notebook to run. It needs one node with 8 H100 or H200 GPUs and about 75 minutes with profiling off.
- `HPML_2026_L8_MultiGPU_Lab_executed.ipynb`: the same notebook with every output, measured on 8 H100 GPUs. Read this if you have no such node.
- `src/`: the training script and helpers the notebook writes and runs.
- `tests/`: unit tests for the code in `src/`. Run them from this folder with `python -m pytest tests -q`. The data, profiling and CPU tensor-parallel tests run without a GPU; the others are skipped unless the machine has 2, 4 or 8 GPUs.

**Read first.** Open the executed copy and read the intro, including "How to read the numbers". Then read the section "When Theory and Measurement Disagree" at the end: it explains where the measured results differ from the textbook formulas and why. After that, read the parts in order.

**Hardware.** Running the notebook needs one node with 8 H100 or H200 GPUs (Part 0 needs 2, Part D needs 4). It does not run on Colab or Kaggle. Reading the executed copy needs nothing.
