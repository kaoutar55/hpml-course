# Lectures

Each folder holds the notebooks and code for one lecture, with a README that says what is inside, what to open first and what hardware it needs. The descriptions of each notebook are in the [main README](../README.md#open-the-notebooks-in-colab).

The Colab badges only work when this repository is public. If a badge shows an error, download the notebook from GitHub and upload it to Colab (File, Upload notebook).

| Lecture | Notebook | Needs | Colab |
|---|---|---|---|
| Lecture 3: runtime and profiling | [Live profiling demo](lecture-03-runtime-and-profiling/HPML_2026_L3_Live_Profiling_Demo.ipynb) | nothing to read (outputs saved); one GPU (Colab T4) to re-run | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaoutar55/hpml-course/blob/main/lectures/lecture-03-runtime-and-profiling/HPML_2026_L3_Live_Profiling_Demo.ipynb) |
| Lecture 4: torch.compile | [Demo notebook](lecture-04-torch-compile/HPML_2026_L4_torch_compile_Demo.ipynb), [executed on an A100](lecture-04-torch-compile/HPML_2026_L4_torch_compile_Demo_executed_A100.ipynb) | CPU works; one GPU (Colab T4) for meaningful timings | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaoutar55/hpml-course/blob/main/lectures/lecture-04-torch-compile/HPML_2026_L4_torch_compile_Demo.ipynb) |
| Lecture 7: distributed data parallelism | [Demo notebook](lecture-07-ddp/demo/HPML_2026_L7_Demo.ipynb), [executed copy](lecture-07-ddp/demo/HPML_2026_L7_Demo_executed.ipynb) | Part A: CPU. Part B: 2 GPUs (Kaggle T4 x2). Colab runs Part A only | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaoutar55/hpml-course/blob/main/lectures/lecture-07-ddp/demo/HPML_2026_L7_Demo.ipynb) |
| Lecture 8: LLM-scale parallelism | [Demo notebook](lecture-08-llm-scale/demo/HPML_2026_L8_Demo.ipynb) | Part A: CPU. B1: one GPU (Colab T4). B2: 2 GPUs (Kaggle T4 x2) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaoutar55/hpml-course/blob/main/lectures/lecture-08-llm-scale/demo/HPML_2026_L8_Demo.ipynb) |
| Lecture 8: multi-GPU lab | [Notebook](lecture-08-llm-scale/multigpu-lab/HPML_2026_L8_MultiGPU_Lab.ipynb), [executed copy](lecture-08-llm-scale/multigpu-lab/HPML_2026_L8_MultiGPU_Lab_executed.ipynb) | one node with 8 H100 or H200 GPUs; read the executed copy if you have none | not on Colab |

## Other lectures

Lectures 1, 2, 5 and 6, and the rest of the schedule, are on the course page: https://kaoutarelmaghraoui.com/hpml/. They do not have notebooks in this repository yet.
