"""Building blocks: RMSNorm, rotary embeddings (RoPE), grouped-query attention with a KV cache, SwiGLU."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """x / sqrt(mean(x^2) + eps) * weight. No mean subtraction and no bias, unlike LayerNorm.
    Computed in float32 for stability, returned in the input dtype."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x32 = x.float()
        out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight


def precompute_rope(head_dim, max_len, theta=10000.0):
    """cos and sin of the angles m * theta_i, shape (max_len, head_dim // 2),
    with theta_i = theta^(-2i / head_dim) for the i-th pair of dimensions."""
    inv_freq = theta ** (-torch.arange(0, head_dim, 2).float() / head_dim)   # (head_dim // 2,)
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)            # (max_len, head_dim // 2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin, start_pos=0):
    """Rotate each interleaved pair (x[2i], x[2i+1]) by the angle of its absolute position.

    x: (B, H, T, head_dim). start_pos is the position of the first token (used with the KV cache).
    Same pairing convention as mistral-inference. After the rotation, the dot product q_m . k_n
    only depends on m - n, which is how RoPE encodes relative positions.
    """
    T = x.size(2)
    c = cos[start_pos:start_pos + T].to(x.dtype)      # (T, head_dim // 2), broadcast over B and H
    s = sin[start_pos:start_pos + T].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]               # even and odd dimensions
    out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return out.flatten(-2)                            # back to (..., head_dim), pairs interleaved again


class KVCache:
    """Keys and values of one layer, preallocated: (max_batch, n_kv_head, max_len, head_dim)."""

    def __init__(self, max_batch, n_kv_head, max_len, head_dim, device="cpu", dtype=torch.float32):
        self.k = torch.zeros(max_batch, n_kv_head, max_len, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros(max_batch, n_kv_head, max_len, head_dim, device=device, dtype=dtype)

    def update(self, k, v, start_pos):
        """Write k, v (B, n_kv, T, hd) at positions start_pos ... start_pos + T - 1,
        then return all cached keys and values from position 0 to start_pos + T - 1."""
        B, T = k.size(0), k.size(2)
        self.k[:B, :, start_pos:start_pos + T] = k
        self.v[:B, :, start_pos:start_pos + T] = v
        return self.k[:B, :, :start_pos + T], self.v[:B, :, :start_pos + T]


class Attention(nn.Module):
    """Causal multi-head attention with grouped-query attention (GQA) and RoPE.

    GQA: n_head query heads share n_kv_head key/value heads (n_head // n_kv_head queries per group).
    n_kv_head = n_head is standard multi-head attention, n_kv_head = 1 is multi-query attention.
    Fewer K/V heads means a smaller KV cache at generation time.
    """

    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.n_kv_head, self.head_dim = cfg.n_head, cfg.n_kv_head, cfg.head_dim
        self.dropout, self.flash = cfg.dropout, cfg.flash
        self.use_rope = cfg.use_rope
        self.wq = nn.Linear(cfg.n_embd, cfg.n_head * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.n_embd, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.n_embd, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_head * cfg.head_dim, cfg.n_embd, bias=False)

    def forward(self, x, cos, sin, start_pos=0, cache=None):
        """x: (B, T, n_embd) for positions start_pos ... start_pos + T - 1. Returns (B, T, n_embd)."""
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)       # (B, H, T, hd)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)    # (B, Hkv, T, hd)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        if self.use_rope:
            q, k = apply_rope(q, cos, sin, start_pos), apply_rope(k, cos, sin, start_pos)
        if cache is not None:
            k, v = cache.update(k, v, start_pos)        # now covers positions 0 ... start_pos + T - 1

        # GQA: repeat each K/V head so every query head has its own copy
        rep = self.n_head // self.n_kv_head
        if rep > 1:
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)

        p_drop = self.dropout if self.training else 0.0
        if self.flash and cache is None and start_pos == 0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p_drop)
        else:
            S = k.size(2)                                                # number of keys
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, S)
            # query t sits at absolute position start_pos + t and may only see keys 0 ... start_pos + t
            q_pos = torch.arange(start_pos, start_pos + T, device=x.device)[:, None]
            k_pos = torch.arange(S, device=x.device)[None, :]
            att = att.masked_fill(k_pos > q_pos, float("-inf"))
            att = F.dropout(att.softmax(dim=-1), p=p_drop, training=self.training)
            y = att @ v                                                  # (B, H, T, hd)
        y = y.transpose(1, 2).reshape(B, T, self.n_head * self.head_dim)
        return F.dropout(self.wo(y), p=p_drop, training=self.training)


class FeedForward(nn.Module):
    """SwiGLU feed-forward: w2(silu(w1 x) * w3 x), three linear layers without bias."""

    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.n_embd, cfg.ffn_dim, bias=False)
        self.w3 = nn.Linear(cfg.n_embd, cfg.ffn_dim, bias=False)
        self.w2 = nn.Linear(cfg.ffn_dim, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
