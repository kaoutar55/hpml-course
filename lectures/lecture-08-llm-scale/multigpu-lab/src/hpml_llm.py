"""A small Llama-style decoder for the HPML Lecture 8 multi-GPU lab.

Every module that the lecture splits has a plain name, so the parallelism code can point at it:
  tok_embeddings, layers["0"].attention.{wq,wk,wv,wo}, layers["0"].feed_forward.{w1,w2,w3},
  layers["0"].attention_norm, layers["0"].ffn_norm, norm, output.

Shapes used in the comments: b = sequences on this GPU, s = tokens per sequence on this GPU,
h = model width (dim), V = vocabulary size (32000 for every size here).
The model code is the same in the first run and in the follow-up runs, so their measurements can be compared.
"""
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 32000
    dim: int = 2048
    n_layers: int = 16
    n_heads: int = 32
    n_kv_heads: int = 8      # grouped-query attention: 4 query heads share one K, V head
    hidden: int = 5632       # SwiGLU hidden size
    max_seq: int = 2048
    rope_theta: float = 500000.0
    eps: float = 1e-5


# Parameter counts are printed by the training script; these are the rough sizes.
# All sizes use a 32K vocabulary, so the output layer is h x 32000 (Llama 3 itself uses 128K).
CONFIGS = {
    'tiny': Config(vocab=4096, dim=256, n_layers=4, n_heads=4, n_kv_heads=2, hidden=704),   # smoke test
    '350M': Config(dim=1024, n_layers=24, n_heads=16, n_kv_heads=8, hidden=2816),
    '1B': Config(dim=2048, n_layers=16, n_heads=32, n_kv_heads=8, hidden=5632),            # about 0.85B
    '3B': Config(dim=3072, n_layers=28, n_heads=24, n_kv_heads=8, hidden=8192),
    '8B': Config(dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, hidden=14336),           # Llama 3 8B shape, 32K vocab
}


def get_config(name, seq):
    return replace(CONFIGS[name], max_seq=seq)


def rope_table(head_dim, seq, theta, device=None):
    # One complex rotation per (position, pair of channels): shape (seq, head_dim / 2), complex64.
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angles = torch.outer(torch.arange(seq, device=device).float(), freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x, freqs):
    # x: (b, s, heads, head_dim). freqs holds one row per position of the LOCAL sequence,
    # which is what context parallelism hands us after it shards the buffer.
    # The rotation runs in FP32 (x.float()) and the result is cast back to x's dtype.
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    f = freqs[: x.shape[1]].view(1, x.shape[1], 1, -1)
    return torch.view_as_real(xc * f).flatten(3).type_as(x)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.head_dim = c.dim // c.n_heads
        self.wq = nn.Linear(c.dim, c.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(c.dim, c.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(c.dim, c.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(c.n_heads * self.head_dim, c.dim, bias=False)

    def forward(self, x, freqs):
        b, s, _ = x.shape
        # -1 instead of n_heads: under tensor parallelism each GPU only has its own heads
        # (n_heads / t query heads and n_kv_heads / t K, V heads).
        q = self.wq(x).view(b, s, -1, self.head_dim)
        k = self.wk(x).view(b, s, -1, self.head_dim)
        v = self.wv(x).view(b, s, -1, self.head_dim)
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))          # (b, heads, s, head_dim)
        rep = q.shape[1] // k.shape[1]
        if rep > 1:                                                # expand K, V to one copy per query head
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        # Causal attention. In BF16 on H100 this dispatches to a fused (flash-style) kernel that
        # does not store the s x s score matrix; it keeps O(s) statistics and recomputes in backward.
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(o.transpose(1, 2).reshape(b, s, -1))


class FeedForward(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w1 = nn.Linear(c.dim, c.hidden, bias=False)
        self.w3 = nn.Linear(c.dim, c.hidden, bias=False)
        self.w2 = nn.Linear(c.hidden, c.dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))           # SwiGLU


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attention_norm = nn.RMSNorm(c.dim, eps=c.eps)
        self.attention = Attention(c)
        self.ffn_norm = nn.RMSNorm(c.dim, eps=c.eps)
        self.feed_forward = FeedForward(c)

    def forward(self, x, freqs):
        h = x + self.attention(self.attention_norm(x), freqs)     # residual stream: (b, s, h)
        return h + self.feed_forward(self.ffn_norm(h))


class Transformer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.tok_embeddings = nn.Embedding(c.vocab, c.dim)
        self.layers = nn.ModuleDict({str(i): Block(c) for i in range(c.n_layers)})   # dict: pipeline stages delete keys
        self.norm = nn.RMSNorm(c.dim, eps=c.eps)
        self.output = nn.Linear(c.dim, c.vocab, bias=False)                          # logits: (b, s, V)
        self.register_buffer('freqs_cis', rope_table(c.dim // c.n_heads, c.max_seq, c.rope_theta), persistent=False)

    def forward(self, tokens):
        # In a pipeline, stage 0 gets token ids and later stages get hidden states.
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)
        h = self.norm(h) if self.norm is not None else h
        return self.output(h) if self.output is not None else h

    @torch.no_grad()
    def init_weights(self, seed=0):
        """Same weights on every GPU and in every mode: CPU generator, fixed order."""
        g = torch.Generator().manual_seed(seed)
        for name, p in sorted(self.named_parameters()):
            if 'norm' in name:
                p.fill_(1.0)
            else:
                std = 0.02 / (2 * self.c.n_layers) ** 0.5 if name.endswith(('wo.weight', 'w2.weight')) else 0.02
                p.copy_(torch.empty(p.shape).normal_(0.0, std, generator=g))
        self.freqs_cis = rope_table(self.c.dim // self.c.n_heads, self.c.max_seq, self.c.rope_theta, self.freqs_cis.device)


def build_model(cfg, seed=0):
    """Allocate on the meta device (no memory), then materialize on CPU and fill deterministically."""
    with torch.device('meta'):
        m = Transformer(cfg)
    m.to_empty(device='cpu')
    m.init_weights(seed)
    return m


def count_params(model):
    n = sum(p.numel() for p in model.parameters())
    emb = model.c.vocab * model.c.dim
    return n, n - emb          # all, and without the embedding table (a lookup has no FLOPs)
