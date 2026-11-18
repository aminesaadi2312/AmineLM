"""Sampling the next token: temperature, top-k and top-p (nucleus)."""
import torch


def filter_logits(logits, top_k=None, top_p=None):
    """logits: (B, V). Apply top-k, then top-p on what is left; removed tokens get -inf.

    Top-p keeps a token if the probability mass of the tokens ranked above it is below top_p,
    so the most likely token is always kept.
    """
    logits = logits.clone()
    if top_k is not None and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits[logits < kth] = float("-inf")
    if top_p is not None and top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
        probs = sorted_logits.softmax(dim=-1)
        mass_before = probs.cumsum(dim=-1) - probs          # mass of the strictly better tokens
        sorted_logits[mass_before >= top_p] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)
    return logits


def sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    """logits: (B, V) -> ids (B,). temperature = 0 means greedy decoding (argmax).
    Order: temperature, then top-k, then top-p."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = filter_logits(logits / temperature, top_k, top_p)
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
