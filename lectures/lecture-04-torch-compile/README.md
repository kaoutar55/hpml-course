# Lecture 4: torch.compile

This folder has one notebook in two copies:

- `HPML_2026_L4_torch_compile_Demo.ipynb`: the notebook to run. It takes one small function, GELU(xW + b), from eager mode through `torch.compile`: benchmarking, the first call against later calls, the FX graph TorchDynamo captures, the kernels Inductor generates, guards and recompiles, graph breaks, a pointwise chain that fuses well, a profiler comparison and the break-even number of calls. It already contains outputs from an A100 run; running it replaces them with yours.
- `HPML_2026_L4_torch_compile_Demo_executed_A100.ipynb`: the same notebook run on an A100. All outputs saved, nothing to run. Use it to compare your numbers.

**Run first.** Open the demo notebook and run it from the top. To see a true cold compile, restart the runtime and run the cell that clears the Inductor cache before any cell that imports torch; the notebook explains why.

**Hardware.** The notebook runs on CPU, but the timings and the profiler section only mean something on a GPU. Colab's free T4 is enough. Section 11 (`mode="reduce-overhead"`) is a CUDA-only, optional demo. On a T4 your times will differ from the A100 copy; the direction of each result stays the same.
