"""Séance M6 · Échantillonnage : température, top-k, top-p."""
import torch


def filter_logits(logits, top_k=None, top_p=None):
    """logits : (B, V). Top-k puis top-p (sur la distribution après top-k) ; les tokens retirés valent −inf.
    Top-p garde un token si la masse des tokens plus probables que lui est < top_p."""
    raise NotImplementedError("séance M6 : à compléter")


def sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    """logits : (B, V) → ids (B,). temperature = 0 → argmax. Ordre : température → top-k → top-p."""
    raise NotImplementedError("séance M6 : à compléter")
