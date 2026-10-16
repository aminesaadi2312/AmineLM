"""Séance M4 · pytest tests/test_model.py"""
import math
import torch
import torch.nn.functional as F
from minigpt import GPTConfig
from minigpt.layers import FeedForward
from minigpt.model import GPT, Block
from minigpt.layers import precompute_rope

CFG = GPTConfig(vocab_size=50, block_size=32, n_layer=2, n_head=4, n_kv_head=2, n_embd=64)


def test_swiglu():
    m = FeedForward(CFG)
    x = torch.randn(2, 5, 64)
    assert torch.allclose(m(x), m.w2(F.silu(m.w1(x)) * m.w3(x)), atol=1e-6)
    assert CFG.ffn_dim == 192


def test_bloc_pre_norm():
    torch.manual_seed(0)
    b = Block(CFG).eval()
    cos, sin = precompute_rope(CFG.head_dim, 32)
    x = torch.randn(2, 8, 64)
    h = x + b.attention(b.attention_norm(x), cos, sin)
    assert torch.allclose(b(x, cos, sin), h + b.feed_forward(b.ffn_norm(h)), atol=1e-5)


def test_forward_formes_et_loss_none():
    m = GPT(CFG)
    logits, loss = m(torch.randint(0, 50, (3, 20)))
    assert logits.shape == (3, 20, 50) and loss is None


def test_perte_initiale_proche_de_ln_V():
    torch.manual_seed(0)
    m = GPT(CFG)
    _, loss = m(torch.randint(0, 50, (8, 32)), torch.randint(0, 50, (8, 32)))
    assert abs(loss.item() - math.log(50)) < 0.3, f"perte initiale {loss.item():.2f}, attendu ≈ {math.log(50):.2f} (init std 0,02 ?)"


def test_causal():
    m = GPT(CFG).eval()
    idx = torch.randint(0, 50, (2, 16))
    idx2 = idx.clone()
    idx2[:, 9:] = torch.randint(0, 50, (2, 7))
    assert torch.allclose(m(idx)[0][:, :9], m(idx2)[0][:, :9], atol=1e-5)


def test_weight_tying_et_parametres():
    m = GPT(CFG)
    assert m.lm_head.weight is m.tok_embeddings.weight
    D, F_, L, V, kv = 64, 192, 2, 50, 2 * 16
    per_layer = D * D + 2 * D * kv + D * D + 3 * D * F_ + 2 * D
    assert m.num_params() == L * per_layer + V * D + D


def test_surapprend_un_batch():
    torch.manual_seed(0)
    m = GPT(CFG)
    idx = torch.randint(0, 50, (4, 17))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(60):
        _, loss = m(idx[:, :-1], idx[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 1.0, f"le modèle n'apprend pas un batch fixe ({loss.item():.2f})"
