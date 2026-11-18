"""pytest tests/test_train.py"""
import math
import numpy as np
import torch
from minigpt import GPTConfig
from minigpt.model import GPT
from minigpt.data import get_batch
from minigpt.train_utils import get_lr, estimate_loss

CFG = GPTConfig(vocab_size=12, block_size=16, n_layer=2, n_head=4, n_kv_head=2, n_embd=32, dropout=0.2)


def test_get_lr():
    assert math.isclose(get_lr(0, 10, 100, 1e-3, 1e-4), 1e-4)
    assert math.isclose(get_lr(9, 10, 100, 1e-3, 1e-4), 1e-3)
    assert math.isclose(get_lr(10, 10, 100, 1e-3, 1e-4), 1e-3)
    assert math.isclose(get_lr(55, 10, 100, 1e-3, 1e-4), 1e-4 + 0.5 * 9e-4 * (1 + math.cos(math.pi * 0.5)))
    assert math.isclose(get_lr(100, 10, 100, 1e-3, 1e-4), 1e-4)


def test_optimizer_groups():
    m = GPT(CFG)
    opt = m.configure_optimizer(0.1, 1e-3)
    assert isinstance(opt, torch.optim.AdamW)
    for g in opt.param_groups:
        for p in g["params"]:
            assert g["weight_decay"] == (0.1 if p.dim() >= 2 else 0.0)
    assert sum(len(g["params"]) for g in opt.param_groups) == len(list(m.parameters()))


def test_estimate_loss_modes():
    m = GPT(CFG)
    m.train()
    data = np.random.randint(0, 12, 2000).astype(np.uint16)
    seen = []
    h = m.register_forward_hook(lambda mod, i, o: seen.append((mod.training, torch.is_grad_enabled())))
    out = estimate_loss(m, {"train": data, "val": data}, 3, 16, 4)
    h.remove()
    assert set(out) == {"train", "val"}
    assert all(not t and not g for t, g in seen), "evaluation in eval() mode and without gradients"
    assert m.training, "the model must go back to train mode"


def test_training_lowers_the_loss():
    torch.manual_seed(0)
    data = (np.arange(20000) % 12).astype(np.uint16)          # periodic sequence: easy to learn
    m = GPT(GPTConfig(vocab_size=12, block_size=16, n_layer=2, n_head=4, n_kv_head=2, n_embd=32))
    opt = m.configure_optimizer(0.1, 3e-3)
    first = None
    for step in range(80):
        for g in opt.param_groups:
            g["lr"] = get_lr(step, 10, 80, 3e-3, 3e-4)
        x, y = get_batch(data, 16, 16)
        _, loss = m(x, y)
        first = first or loss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    assert loss.item() < 0.3 * first
