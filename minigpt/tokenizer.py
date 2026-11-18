"""Tokenizers: a character-level one (used for training) and a byte-level BPE (the idea behind Tekken, Mistral's tokenizer)."""
import json


class CharTokenizer:
    """One id per distinct character, in sorted order."""

    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        return cls(set(text))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

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
    """Count adjacent pairs: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}."""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """Replace every occurrence of pair (left to right, no overlap) with the new id idx."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-level BPE: the 256 possible bytes, then vocab_size - 256 learned merges."""

    def __init__(self):
        self.merges = {}                                   # (a, b) -> new id, in creation order
        self.vocab = {i: bytes([i]) for i in range(256)}   # id -> bytes

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
            pair = max(stats, key=stats.get)               # most frequent pair (first one seen on ties)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
        return self

    def encode(self, text) -> list[int]:
        """Start from bytes and apply merges in the order they were learned (earliest first)."""
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
