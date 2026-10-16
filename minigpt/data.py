"""Séance M2 · Données : fichiers binaires uint16 lus par memmap, et batches (x, y) décalés d'un cran."""
import numpy as np
import torch


def load_split(path):
    """Ouvre un fichier .bin d'ids uint16 sans le charger en RAM."""
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, block_size, batch_size, device="cpu", generator=None):
    """data : tableau 1D d'ids (numpy ou memmap).
    Renvoie x, y de shape (batch_size, block_size), int64, sur device, avec y[:, t] = data[i + t + 1].
    Les positions de départ sont tirées uniformément dans [0, len(data) − block_size − 1]."""
    raise NotImplementedError("séance M2 : à compléter")
