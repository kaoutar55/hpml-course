"""The parallelism recipes from Lecture 8, each one a short function.

Order matters, and it is the order TorchTitan uses:
  1. tensor (+ sequence) parallelism   apply_tp
  2. activation checkpointing          apply_ac
  3. FSDP2 (ZeRO-3, ZeRO-2-like, HSDP)  apply_fsdp
Pipeline parallelism builds its own stage modules (build_pipeline), and context parallelism
is a context manager around forward and backward (cp_context).

The defaults give the setup of the first run. Two switches, used in the follow-up runs, are off
unless asked for: apply_fsdp(param_dtype=torch.float32) for FP32 correctness tests, and
apply_tp(vocab_parallel_loss=True), which keeps the logits sharded over the vocabulary.
"""
import contextlib
import copy

import torch
import torch.nn.functional as F
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (ColwiseParallel, PrepareModuleInput, RowwiseParallel,
                                               SequenceParallel, parallelize_module)

try:
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard              # PyTorch 2.6 and later
except ImportError:
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard  # 2.4, 2.5


# ---------------------------------------------------------------- FSDP2 (lecture: ZeRO and FSDP2)
def apply_fsdp(model, mesh, reshard_after_forward=True, param_dtype=torch.bfloat16):
    """One FSDP unit per transformer block, then the root (embeddings, final norm, output).

    fully_shard replaces every parameter with a DTensor sharded on dim 0 over the mesh: each GPU keeps
    1/N of the FP32 weights, gradients and Adam state. Before a unit runs forward, FSDP all-gathers
    its weights (cast to param_dtype, so the all-gather moves BF16); after backward it reduce-scatters
    the gradients (in reduce_dtype = FP32) so each GPU ends with the gradient of its own shard.

    reshard_after_forward=True  : ZeRO-3. Free the gathered weights after forward, gather again in backward.
    reshard_after_forward=False : keep each gathered block from its forward until its backward.
        We call this "ZeRO-2-like", not ZeRO-2: gradients and optimizer state are sharded as in ZeRO-2,
        but the weights are still sharded between steps and gathered once per step. Literal ZeRO-2
        (DeepSpeed stage 2) keeps a full weight replica on every GPU for the whole run.
    A 2D mesh (replicate, shard) gives HSDP: shard inside a group, replicate across groups."""
    mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32)
    for block in model.layers.values():
        fully_shard(block, mesh=mesh, mp_policy=mp, reshard_after_forward=reshard_after_forward)
    fully_shard(model, mesh=mesh, mp_policy=mp, reshard_after_forward=reshard_after_forward)
    return model


# ---------------------------------------------------------------- tensor parallelism (lecture: tensor and sequence parallelism)
def apply_tp(model, tp_mesh, sequence_parallel=True, vocab_parallel_loss=False):
    """Megatron split of every block: wq, wk, wv, w1, w3 by columns; wo, w2 by rows.

    ColwiseParallel: weight (out, in) is Shard(0) over the t GPUs, so each GPU computes out/t of the
        output features. Its input must be Replicate (the full activation on every TP GPU).
    RowwiseParallel: weight is Shard(1); each GPU multiplies its slice of the input features and
        produces a partial sum of the full output, which is all-reduced (or reduce-scattered with SP).
    SequenceParallel: the norms run on a sequence shard, Shard(1) of (b, s, h).
    PrepareModuleInput: before attention and the MLP, all-gather the sequence shards back to the full
        (b, s, h) activation, because the column-split matmuls need every token.

    With sequence parallelism the all-reduce after wo and w2 becomes a reduce-scatter and the one
    before wq and w1 an all-gather: same bytes, but the residual stream and norms are split by t.
    Without it the residual stream is replicated on every TP GPU.

    All t GPUs of a TP group work on the SAME sequences. With a fixed global batch, raising t lowers
    the data-parallel degree, so each GPU processes more sequences (see Part C).

    vocab_parallel_loss=False (default): the output layer all-gathers the logits, so every TP GPU
        holds the full (b, s, V) logits. True: keep them sharded over V (Shard(-1)) and compute the
        cross-entropy with torch.distributed.tensor.parallel.loss_parallel()."""
    if vocab_parallel_loss:
        out_style = ColwiseParallel(input_layouts=Shard(1) if sequence_parallel else Replicate(),
                                    output_layouts=Shard(-1), use_local_output=False)
    elif sequence_parallel:
        out_style = ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate())
    else:
        out_style = ColwiseParallel(output_layouts=Replicate())
    if sequence_parallel:
        parallelize_module(model, tp_mesh, {
            'tok_embeddings': RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
            'norm': SequenceParallel(),
            'output': out_style,
        })
    else:
        parallelize_module(model, tp_mesh, {
            'tok_embeddings': RowwiseParallel(input_layouts=Replicate()),
            'output': out_style,
        })
    for block in model.layers.values():
        plan = {
            'attention.wq': ColwiseParallel(), 'attention.wk': ColwiseParallel(), 'attention.wv': ColwiseParallel(),
            'feed_forward.w1': ColwiseParallel(), 'feed_forward.w3': ColwiseParallel(),
        }
        if sequence_parallel:
            plan.update({
                'attention_norm': SequenceParallel(),
                'attention': PrepareModuleInput(input_layouts=(Shard(1), None), desired_input_layouts=(Replicate(), None)),
                'attention.wo': RowwiseParallel(output_layouts=Shard(1)),
                'ffn_norm': SequenceParallel(),
                'feed_forward': PrepareModuleInput(input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),)),
                'feed_forward.w2': RowwiseParallel(output_layouts=Shard(1)),
            })
        else:
            plan.update({'attention.wo': RowwiseParallel(), 'feed_forward.w2': RowwiseParallel()})
        parallelize_module(block, tp_mesh, plan)
    return model


def loss_context(vocab_parallel_loss):
    """Context for the cross-entropy and its backward. loss_parallel() lets F.cross_entropy take logits
    that are sharded over the vocabulary: it computes the softmax max and sum with all-reduces over the
    TP group instead of gathering (b, s, V)."""
    if not vocab_parallel_loss:
        return contextlib.nullcontext()
    from torch.distributed.tensor.parallel import loss_parallel
    return loss_parallel()


# ---------------------------------------------------------------- activation checkpointing (lecture: activation checkpointing)
def apply_ac(model, mode):
    """'full': keep only each block's input, rerun the block's forward during backward.
    The embedding, final norm, output layer and loss are outside the blocks and are not recomputed."""
    if mode == 'none':
        return model
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    for name in list(model.layers.keys()):
        model.layers[name] = checkpoint_wrapper(model.layers[name], preserve_rng_state=False)
    return model


# ---------------------------------------------------------------- pipeline parallelism (lecture: pipeline parallelism)
SCHEDULES = {'gpipe': 'ScheduleGPipe', '1f1b': 'Schedule1F1B',
             'interleaved': 'ScheduleInterleaved1F1B', 'zb': 'ScheduleInterleavedZeroBubble'}


def pipeline_loss(pred, target):
    return F.cross_entropy(pred.float().flatten(0, 1), target.flatten(0, 1))


def build_pipeline(model, pp_rank, pp_size, schedule, n_microbatches, v, device, group):
    """Cut the CPU model into stages and keep the ones this rank owns.
    gpipe / 1f1b: one stage per rank. interleaved / zb: v stages per rank, rank r owns stages r, r + p, r + 2p, ...

    Layers are split evenly, but the stages are not equal work: the first stage also has the
    embedding (a cheap lookup) and the last stage also has the final norm, the h x V output matmul
    and the loss. For the 1B model the output layer has about 1.5 blocks' worth of weights."""
    from torch.distributed import pipelining
    multi = schedule in ('interleaved', 'zb')
    n_stages = pp_size * (v if multi else 1)
    L = len(model.layers)
    assert L % n_stages == 0, f'{L} layers do not split into {n_stages} equal stages'
    per = L // n_stages
    stages = []
    for sid in range(pp_rank, n_stages, pp_size):
        m = copy.deepcopy(model)
        keep = set(range(sid * per, (sid + 1) * per))
        for k in list(m.layers.keys()):
            if int(k) not in keep:
                del m.layers[k]
        if sid > 0:
            m.tok_embeddings = None
        if sid < n_stages - 1:
            m.norm, m.output = None, None
        m.to(device)
        stages.append(pipelining.PipelineStage(m, sid, n_stages, device, group=group))
    cls = getattr(pipelining, SCHEDULES[schedule])
    sched = cls(stages if multi else stages[0], n_microbatches=n_microbatches, loss_fn=pipeline_loss)
    return stages, sched


# ---------------------------------------------------------------- context parallelism (lecture: context parallelism)
def cp_context(cp_mesh, buffers, seq_dims, no_restore, rotate='allgather'):
    """Shards each buffer along its sequence dimension for the duration of the block, and swaps
    scaled_dot_product_attention for a ring version. rotate='allgather' is what Llama 3 did
    (gather K and V once per layer), 'alltoall' passes K and V blocks around the ring.
    Each sequence is cut into 2c chunks and GPU i gets chunks i and 2c - 1 - i, so every GPU has
    the same amount of causal-attention work."""
    try:
        from torch.distributed.tensor.experimental import context_parallel
    except ImportError as e:
        raise SystemExit(f'context parallelism needs PyTorch 2.7 or later: {e}')
    try:                                      # the private module moved between releases
        from torch.distributed.tensor.experimental._attention import set_rotate_method
    except ImportError:
        from torch.distributed.tensor.experimental._context_parallel._attention import set_rotate_method
    from torch.nn.attention import SDPBackend, sdpa_kernel
    set_rotate_method(rotate)
    stack = contextlib.ExitStack()
    stack.enter_context(sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]))
    stack.enter_context(context_parallel(cp_mesh, buffers=buffers, buffer_seq_dims=seq_dims, no_restore_buffers=no_restore))
    return stack
