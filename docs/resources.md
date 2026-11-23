# Resources

## Tools

- [Perfetto trace viewer](https://ui.perfetto.dev): use this to open the PyTorch profiler traces from Lecture 3 and the multi-GPU lab. Drag a `.json` or `.json.gz` file onto the page.
- [PyTorch memory visualizer](https://pytorch.org/memory_viz): use this to open memory snapshots such as `mem.pickle` from Lecture 3 and see which allocations are alive at each moment.
- [PyTorch documentation](https://pytorch.org/docs/stable/): read this for the exact behaviour of `torch.profiler`, `torch.compile`, DDP, FSDP2 and DTensor when a notebook result surprises you.
- [Triton documentation](https://triton-lang.org): read this to understand the kernels `torch.compile` generates in Lecture 4, and to write your own.
- [TorchTitan](https://github.com/pytorch/torchtitan): read this for a PyTorch-native trainer that combines FSDP2, tensor, pipeline and context parallelism the way the Lecture 8 lab does, at production scale.
- [Google Colab](https://colab.research.google.com): use this for a free single GPU; see the [setup guide](getting-started.md).
- [Modal](https://modal.com): a paid cloud service for renting GPUs by the second. Optional; `lectures/lecture-07-ddp/demo/modal_ddp.py` uses it to run the Lecture 7 demo on 2 H100 GPUs.

## Papers behind Lectures 7 and 8

- Rajbhandari et al., ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (arXiv [1910.02054](https://arxiv.org/abs/1910.02054)). Read this for the 16 bytes per parameter of training state and the three ZeRO stages that FSDP implements.
- Shoeybi et al., Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (arXiv [1909.08053](https://arxiv.org/abs/1909.08053)). Read this for why tensor parallelism pairs a column-parallel layer with a row-parallel one.
- Korthikanti et al., Reducing Activation Recomputation in Large Transformer Models (arXiv [2205.05198](https://arxiv.org/abs/2205.05198)). Read this for the activation-memory formula, sequence parallelism and selective checkpointing.
- Huang et al., GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism (arXiv [1811.06965](https://arxiv.org/abs/1811.06965)). Read this for micro-batches and the pipeline bubble formula.
- Liu et al., Ring Attention with Blockwise Transformers for Near-Infinite Context (arXiv [2310.01889](https://arxiv.org/abs/2310.01889)). Read this for how context parallelism passes keys and values around a ring of GPUs.
- Dao et al., FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (arXiv [2205.14135](https://arxiv.org/abs/2205.14135)). Read this for why a fused attention kernel never stores the full s × s score matrix, which changes the memory estimates in Lecture 8.
- Llama Team, The Llama 3 Herd of Models (arXiv [2407.21783](https://arxiv.org/abs/2407.21783)). Read this for how one production run combined data, tensor, pipeline and context parallelism, and why it used an all-gather for context parallelism.
