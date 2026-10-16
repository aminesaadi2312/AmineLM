"""Séance M5 · Outils d'entraînement : schedule du learning rate et évaluation."""
import math
import torch

from .data import get_batch


def get_lr(step, warmup, total, lr_max, lr_min):
    """Warmup linéaire lr_max·(step+1)/warmup, puis décroissance cosinus jusqu'à lr_min à step = total ; lr_min ensuite."""
    raise NotImplementedError("séance M5 : à compléter")


@torch.no_grad()
def estimate_loss(model, splits, eval_iters, block_size, batch_size, device="cpu", generator=None):
    """splits : {"train": data, "val": data}. Perte moyenne sur eval_iters batches par split,
    modèle en eval() pendant le calcul puis remis dans son mode d'origine."""
    raise NotImplementedError("séance M5 : à compléter")
