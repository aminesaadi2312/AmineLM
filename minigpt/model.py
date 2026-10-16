"""Séances M4 (bloc, modèle), M5 (optimiseur) et M6 (génération avec KV cache)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig
from .layers import RMSNorm, Attention, FeedForward, KVCache, precompute_rope
from .sampling import sample_next


class Block(nn.Module):
    """Bloc pre-norm (noms de mistral-inference) : attention_norm, attention, ffn_norm, feed_forward."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        raise NotImplementedError("séance M4 : à compléter")

    def forward(self, x, cos, sin, start_pos=0, cache=None):
        raise NotImplementedError("séance M4 : à compléter")


class GPT(nn.Module):
    """tok_embeddings → layers → norm → lm_head (poids liés si cfg.tie_weights)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        raise NotImplementedError("séance M4 : à compléter")

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_embeddings.weight.numel()
        return n

    def forward(self, idx, targets=None, start_pos=0, caches=None):
        """idx : (B, T) → (logits (B, T, V), loss ou None). caches : liste de KVCache (une par couche) ou None."""
        raise NotImplementedError("séance M4 : à compléter")

    def configure_optimizer(self, weight_decay, lr, betas=(0.9, 0.95), device_type="cpu"):
        """AdamW avec deux groupes : weight decay sur les paramètres de dimension ≥ 2, rien sur les autres."""
        raise NotImplementedError("séance M5 : à compléter")

    def new_caches(self, batch_size, device=None, dtype=None):
        """Une KVCache par couche, de longueur block_size."""
        raise NotImplementedError("séance M6 : à compléter")

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, use_cache=True, generator=None):
        """Ajoute max_new_tokens tokens à idx (B, T). Avec use_cache : un prefill, puis un seul token par appel."""
        raise NotImplementedError("séance M6 : à compléter")
