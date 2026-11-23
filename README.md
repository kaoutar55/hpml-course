# High-Performance Machine Learning (HPML)

Course materials for COMS E6998, High-Performance Machine Learning, a graduate course at Columbia University taught by Dr. Kaoutar El Maghraoui in Fall 2026. The course is about making machine learning fast: how GPUs and PyTorch run your code, how to profile it and speed it up, and how to scale training across many GPUs.

This repository holds the notebooks, lab code and browser simulations used in the lectures. The course page has the schedule, the slides and live copies of the simulations: https://kaoutarelmaghraoui.com/hpml/

## Start here

1. Pick a notebook from the table below and open it with its "Open in Colab" badge.
2. If the "Needs" column asks for a GPU, switch the runtime before you run anything: Runtime, Change runtime type, T4 GPU. Changing it later restarts the notebook and clears what you ran.
3. Run the cells in order, from the top. Later cells use files and variables that earlier cells create.

The Colab badges only work when this repository is public. If a badge shows an error, download the notebook from GitHub and upload it to Colab instead (File, Upload notebook).

For setup details, Kaggle's two-GPU option and fixes for common errors, read [docs/getting-started.md](docs/getting-started.md).

## Open the notebooks in Colab

| Lecture | Notebook | What you do and what you learn | Needs | Open |
|---|---|---|---|---|
| 3: Runtime and profiling | [Live profiling demo](lectures/lecture-03-runtime-and-profiling/HPML_2026_L3_Live_Profiling_Demo.ipynb) | Read real PyTorch profiler traces: eager against compiled ResNet-18, a launch-bound loop, hidden syncs from `.item()`, the input pipeline, three AdamW implementations and a memory snapshot. You learn to tell from a trace where the time goes. | None to read: outputs and traces are saved. One GPU (Colab T4) to re-run | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaoutar55/hpml-course/blob/main/lectures/lecture-03-runtime-and-profiling/HPML_2026_L3_Live_Profiling_Demo.ipynb) |
| 4: torch.compile | [torch.compile demo](lectures/lecture-04-torch-compile/HPML_2026_L4_torch_compile_Demo.ipynb) | Run one small function, GELU(xW + b), through eager mode and `torch.compile`, then look at the captured FX graph, the kernels Inductor generates, guards and recompiles, graph breaks and dynamic shapes. You learn when compiling pays off: pointwise chains fuse well, a large matrix multiply gains little, and the first call's compile time must be paid back. | Runs on CPU. One GPU (Colab T4) for meaningful timings and the CUDA-only section | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaoutar55/hpml-course/blob/main/lectures/lecture-04-torch-compile/HPML_2026_L4_torch_compile_Demo.ipynb) |
| 4: torch.compile | [torch.compile demo, executed on an A100](lectures/lecture-04-torch-compile/HPML_2026_L4_torch_compile_Demo_executed_A100.ipynb) | The same notebook run on an A100. All outputs saved, nothing to run. Compare your timings with it. | Nothing | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaoutar55/hpml-course/blob/main/lectures/lecture-04-torch-compile/HPML_2026_L4_torch_compile_Demo_executed_A100.ipynb) |
| 7: Distributed data parallelism | [Lecture 7 demo](lectures/lecture-07-ddp/demo/HPML_2026_L7_Demo.ipynb) | Part A: run the five collectives, write ring all-reduce by hand and fit the alpha-beta cost model to measured times. Part B: DDP scaling from one GPU to two, bucket size, `no_sync` for gradient accumulation, and a profiler trace of one DDP step. You learn what DDP communicates and how much of it hides behind backward. | Part A: CPU (Colab works). Part B: 2 GPUs (Kaggle T4 x2) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaoutar55/hpml-course/blob/main/lectures/lecture-07-ddp/demo/HPML_2026_L7_Demo.ipynb) |
| 7: Distributed data parallelism | [Lecture 7 demo, executed](lectures/lecture-07-ddp/demo/HPML_2026_L7_Demo_executed.ipynb) | The same notebook run on 2 H100 GPUs. All outputs saved, nothing to run. | Nothing | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaoutar55/hpml-course/blob/main/lectures/lecture-07-ddp/demo/HPML_2026_L7_Demo_executed.ipynb) |
| 8: LLM-scale parallelism | [Lecture 8 demo](lectures/lecture-08-llm-scale/demo/HPML_2026_L8_Demo.ipynb) | Part A on CPU: a ZeRO memory calculator, DTensor placements, tensor parallelism of one MLP block, FSDP2 on CPU and a pipeline-schedule simulator. B1: activation checkpointing, memory against time. B2: DDP against FSDP2. You learn what each technique stores per GPU and what it communicates. | Part A: CPU (Colab works). B1: one GPU (Colab T4). B2: 2 GPUs (Kaggle T4 x2) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kaoutar55/hpml-course/blob/main/lectures/lecture-08-llm-scale/demo/HPML_2026_L8_Demo.ipynb) |
| 8: LLM-scale parallelism | [Multi-GPU lab](lectures/lecture-08-llm-scale/multigpu-lab/HPML_2026_L8_MultiGPU_Lab.ipynb) | Train one Llama-style model on 8 GPUs with each technique from the lecture (DDP, FSDP2, activation checkpointing, tensor, pipeline and context parallelism, distributed checkpointing) and measure memory and step time. Not for Colab. | One node with 8 H100 or H200 GPUs, about 75 minutes | not on Colab |
| 8: LLM-scale parallelism | [Multi-GPU lab, executed](lectures/lecture-08-llm-scale/multigpu-lab/HPML_2026_L8_MultiGPU_Lab_executed.ipynb) | The lab run on 8 H100 GPUs. All outputs saved, nothing to run. Read it on GitHub if you have no such node. | Nothing | not on Colab |

Colab gives you one GPU, so in the Lecture 7 demo only Part A runs there, and in the Lecture 8 demo Part A and B1 run there. The two-GPU parts need Kaggle's free "GPU T4 x2" accelerator; [docs/getting-started.md](docs/getting-started.md) shows how to turn it on.

The browser simulations in [simulations/](simulations/) need no install and no GPU. Open `simulations/index.html` in a browser.

## What each notebook assumes you know

- All notebooks assume you can read Python and a basic PyTorch training loop.
- The Lecture 4 demo builds on Lecture 3: timing GPU code with a synchronize and a median, and reading profiler tables.
- The Lecture 8 demo and lab build on Lecture 7: the collectives (all-reduce, reduce-scatter, all-gather) and how DDP works.
- Each notebook names the lecture topic of every section, so keep the slides open next to it.

## What is here

| Folder | What it holds |
|---|---|
| [lectures/](lectures/) | One folder per lecture with notebooks, lab code and a README that says what to open first and what hardware it needs |
| [simulations/](simulations/) | Interactive browser simulations used in the lectures. They work offline |
| [docs/](docs/) | The [setup guide](docs/getting-started.md) and a [reading list](docs/resources.md) |

## Get help or report a mistake

If a notebook does not run, or you find a mistake in a notebook or a slide, open an issue on this repository's GitHub Issues page. There are two templates, one for each case. Include the notebook name, the cell and the full error text, so the problem can be reproduced.

## License

- Code (Python files, notebook code cells, simulation JavaScript): MIT, see [LICENSE](LICENSE).
- Teaching content (notebook text, documentation, figures, simulation text): CC BY 4.0, see [LICENSE-CONTENT.md](LICENSE-CONTENT.md).
- Third-party notices: [NOTICE.md](NOTICE.md).

Please credit the course when you reuse it. [CITATION.cff](CITATION.cff) gives the citation.
