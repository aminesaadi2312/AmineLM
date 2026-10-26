"""Séance M1 · Tokenizers : caractère (utilisé pour l'entraînement) et BPE au niveau octet (comme Tekken)."""
import json


class CharTokenizer:
    """Un id par caractère distinct du texte, dans l'ordre trié."""

    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.stoi = {c : i for i, c in enumerate(self.chars)}
        self.itos = {i : c for i, c in enumerate(self.chars)}
    @classmethod
    def from_text(cls, text):
        return cls(set(text))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [ self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "char", "chars": self.chars}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["chars"])


def get_stats(ids, counts=None):
    """Compte les paires adjacentes : [1, 2, 3, 1, 2] → {(1, 2): 2, (2, 3): 1, (3, 1): 1}."""
    counts = {} if counts is None else counts
    for i,j in zip(ids, ids[1:]):
        counts[(i,j)] = 1 + counts.get((i,j), 0) 
    return counts


def merge(ids, pair, idx):
    """Remplace chaque occurrence de pair (de gauche à droite, sans chevauchement) par idx."""
    out, i = [], 0
    while i < len(ids):
        if i+1 < len(ids) and ids[i] == pair[0] and ids[i+1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """BPE au niveau octet : 256 octets de base, puis vocab_size − 256 fusions."""

    def __init__(self):
        self.merges = {}                                   # (a, b) → nouvel id, dans l'ordre de création
        self.vocab = {i: bytes([i]) for i in range(256)}   # id → octets

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text, vocab_size):
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))
        for i in range(vocab_size - 256):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)               # à égalité : la première rencontrée
            idx = 256 + i
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
        return self

    def encode(self, text) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) > 1:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids) -> str:
        return b"".join(self.vocab[idx] for idx in ids).decode("utf-8", errors="replace")

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
