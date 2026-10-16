"""Séance M1 · Tokenizers : caractère (utilisé pour l'entraînement) et BPE au niveau octet (comme Tekken)."""
import json


class CharTokenizer:
    """Un id par caractère distinct du texte, dans l'ordre trié."""

    def __init__(self, chars):
        raise NotImplementedError("séance M1 : à compléter")

    @classmethod
    def from_text(cls, text):
        return cls(set(text))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        raise NotImplementedError("séance M1 : à compléter")

    def decode(self, ids) -> str:
        raise NotImplementedError("séance M1 : à compléter")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "char", "chars": self.chars}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["chars"])


def get_stats(ids, counts=None):
    """Compte les paires adjacentes : [1, 2, 3, 1, 2] → {(1, 2): 2, (2, 3): 1, (3, 1): 1}."""
    raise NotImplementedError("séance M1 : à compléter")


def merge(ids, pair, idx):
    """Remplace chaque occurrence de pair (de gauche à droite, sans chevauchement) par idx."""
    raise NotImplementedError("séance M1 : à compléter")


class BPETokenizer:
    """BPE au niveau octet : 256 octets de base, puis vocab_size − 256 fusions."""

    def __init__(self):
        self.merges = {}                                   # (a, b) → nouvel id, dans l'ordre de création
        self.vocab = {i: bytes([i]) for i in range(256)}   # id → octets

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text, vocab_size):
        raise NotImplementedError("séance M1 : à compléter")

    def encode(self, text) -> list[int]:
        raise NotImplementedError("séance M1 : à compléter")

    def decode(self, ids) -> str:
        raise NotImplementedError("séance M1 : à compléter")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "bpe", "merges": [[a, b, i] for (a, b), i in self.merges.items()]}, f)

    @classmethod
    def load(cls, path):
        tok = cls()
        with open(path, encoding="utf-8") as f:
            for a, b, i in json.load(f)["merges"]:
                tok.merges[(a, b)] = i
                tok.vocab[i] = tok.vocab[a] + tok.vocab[b]
        return tok
