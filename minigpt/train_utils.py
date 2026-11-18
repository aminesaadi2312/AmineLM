"""Training helpers: learning-rate schedule and loss estimation."""
import math
import torch

from .data import get_batch


def get_lr(step, warmup, total, lr_max, lr_min):
    """Linear warmup to lr_max over `warmup` steps, then cosine decay to lr_min at step `total`."""
    if step < warmup:
        return lr_max * (step + 1) / warmup
    if step >= total:
        return lr_min
    progress = (step - warmup) / (total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, splits, eval_iters, block_size, batch_size, device="cpu", generator=None):
    """Average loss over eval_iters random batches for each split ({"train": data, "val": data}).
    The model is put in eval mode (no dropout) and restored to its previous mode afterwards."""
    was_training = model.training
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device, generator)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train(was_training)
    return out
