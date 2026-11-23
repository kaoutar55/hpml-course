# Lecture 3: Reading real PyTorch traces

- `HPML_2026_L3_Live_Profiling_Demo.ipynb`: the notebook. Every cell has its output saved, from a run on an A100, so you can read it without a GPU.
- `traces/*.json.gz`: one profiler trace per run in the notebook. Go to https://ui.perfetto.dev and drag a file onto the page. Perfetto opens `.json.gz` directly, so do not unzip it.
- `mem.pickle`: a CUDA memory snapshot from section 7. Drag it onto https://pytorch.org/memory_viz.

**Read first.** Open the notebook and read it from the top. Then open `traces/launch_bound_loop.json.gz` next to `traces/launch_bound_vectorized.json.gz`: a comb of tiny kernels against a couple of large ones is the easiest difference to see.

**Hardware.** You need no GPU to read it. To re-run it, use one GPU: Colab's free T4 is enough. A re-run writes plain `.json` files into `traces/`. Your numbers will differ from the saved A100 run; the shapes in the traces will not.
