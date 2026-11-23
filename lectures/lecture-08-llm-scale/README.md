# Lecture 8: LLM-scale parallelism

Two folders, for two different kinds of hardware:

- `demo/HPML_2026_L8_Demo.ipynb`: the demo notebook. Part A shows each technique on CPU processes: a ZeRO memory calculator, DTensor placements, tensor parallelism of one MLP block, FSDP2 and a pipeline-schedule simulator. B1 measures activation checkpointing on one GPU. B2 compares DDP and FSDP2 on two GPUs.
- `multigpu-lab/`: a lab that trains one Llama-style model on 8 GPUs with every technique from the lecture, with an executed copy to read. Its [README](multigpu-lab/README.md) has the details.

**Run first.** Start with the demo notebook, Part A. Then read the executed copy of the multi-GPU lab, starting with the section "When Theory and Measurement Disagree" at the end.

**Hardware.**

| Material | Needs |
|---|---|
| Demo, Part A | CPU (Colab works) |
| Demo, B1 | one GPU (Colab's free T4) |
| Demo, B2 | 2 GPUs (Kaggle's free "GPU T4 x2") |
| Multi-GPU lab | one node with 8 H100 or H200 GPUs; read the executed copy if you have none |

The T4 has no BF16, so the demo's GPU parts use FP16 autocast (B1) or FP32 (B2). The browser simulations for this lecture are in [../../simulations/lecture-8-llm-scale/](../../simulations/lecture-8-llm-scale/).
