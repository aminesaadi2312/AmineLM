"""Séances M3 (RMSNorm, RoPE, attention), M4 (FFN SwiGLU) et M6 (KV cache)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """x / √(mean(x²) + ε) · weight, calculée en float32, sortie au dtype de x."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        raise NotImplementedError("séance M3 : à compléter")

    def forward(self, x):
        raise NotImplementedError("séance M3 : à compléter")


def precompute_rope(head_dim, max_len, theta=10000.0):
    """cos, sin de shape (max_len, head_dim // 2) pour les angles m · θ_i, θ_i = theta^(−2i / head_dim)."""
    raise NotImplementedError("séance M3 : à compléter")


def apply_rope(x, cos, sin, start_pos=0):
    """x : (B, H, T, head_dim). Fait tourner chaque paire entrelacée (x[2i], x[2i+1]) de l'angle
    de sa position absolue start_pos + t (convention de mistral-inference)."""
    raise NotImplementedError("séance M3 : à compléter")


class KVCache:
    """Cache des keys et values d'une couche : tenseurs (max_batch, n_kv_head, max_len, head_dim)."""

    def __init__(self, max_batch, n_kv_head, max_len, head_dim, device="cpu", dtype=torch.float32):
        self.k = torch.zeros(max_batch, n_kv_head, max_len, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros(max_batch, n_kv_head, max_len, head_dim, device=device, dtype=dtype)

    def update(self, k, v, start_pos):
        """Écrit k, v (B, n_kv, T, hd) aux positions start_pos … start_pos + T − 1,
        puis renvoie toutes les keys et values de 0 à start_pos + T − 1."""
        raise NotImplementedError("séance M6 : à compléter")


class Attention(nn.Module):
    """Attention causale multi-têtes avec GQA et RoPE. Attributs : wq, wk, wv, wo (sans biais)."""

    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.n_kv_head, self.head_dim = cfg.n_head, cfg.n_kv_head, cfg.head_dim
        self.dropout, self.flash = cfg.dropout, cfg.flash
        raise NotImplementedError("séance M3 : à compléter")

    def forward(self, x, cos, sin, start_pos=0, cache=None):
        """x : (B, T, n_embd) aux positions start_pos … start_pos + T − 1 → (B, T, n_embd)."""
        B, T, _ = x.shape
        raise NotImplementedError("séance M3 : à compléter")
        if cache is not None:
            raise NotImplementedError("séance M6 : à compléter")
        raise NotImplementedError("séance M3 : à compléter")
        if self.flash and cache is None and start_pos == 0:
            raise NotImplementedError("séance M7 : à compléter")
        else:
            raise NotImplementedError("séance M3 : à compléter")
        raise NotImplementedError("séance M3 : à compléter")


class FeedForward(nn.Module):
    """SwiGLU : w2(silu(w1 x) ⊙ w3 x), trois nn.Linear sans biais."""

    def __init__(self, cfg):
        super().__init__()
        raise NotImplementedError("séance M4 : à compléter")

    def forward(self, x):
        raise NotImplementedError("séance M4 : à compléter")
