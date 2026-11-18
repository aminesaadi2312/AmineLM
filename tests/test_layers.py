"""pytest tests/test_layers.py"""
import math
import torch
import torch.nn.functional as F
from minigpt import GPTConfig
from minigpt.layers import RMSNorm, precompute_rope, apply_rope, Attention

CFG = GPTConfig(vocab_size=32, block_size=32, n_layer=2, n_head=4, n_kv_head=2, n_embd=32)


def test_rmsnorm_formula_and_dtype():
    m = RMSNorm(16)
    assert torch.equal(m.weight.data, torch.ones(16))
    with torch.no_grad():
        m.weight.normal_()
    x = torch.randn(3, 5, 16) * 4 + 2
    ref = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * m.weight
    assert torch.allclose(m(x), ref, atol=1e-5)
    xb = x.to(torch.bfloat16)
    assert m.to(torch.bfloat16)(xb).dtype == torch.bfloat16


def test_rope_frequencies():
    cos, sin = precompute_rope(8, 10)
    assert cos.shape == (10, 4)
    inv = torch.tensor([10000 ** (-2 * i / 8) for i in range(4)])
    assert torch.allclose(cos, torch.cos(torch.arange(10)[:, None] * inv), atol=1e-6)
    assert torch.allclose(sin, torch.sin(torch.arange(10)[:, None] * inv), atol=1e-6)


def test_rope_rotates_interleaved_pairs():
    cos, sin = precompute_rope(4, 3)
    x = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]] * 3]])          # (1, 1, 3, 4)
    out = apply_rope(x, cos, sin)
    # position 1, pair 0: angle of 1 rad -> (cos 1, sin 1)
    assert torch.allclose(out[0, 0, 1, :2], torch.tensor([math.cos(1.0), math.sin(1.0)]), atol=1e-6)
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5), "a rotation keeps the norm"


def test_rope_relative_and_start_pos():
    cos, sin = precompute_rope(16, 16)
    q, k = torch.randn(16), torch.randn(16)
    Q = apply_rope(q.expand(1, 1, 16, 16).clone(), cos, sin)[0, 0]
    K = apply_rope(k.expand(1, 1, 16, 16).clone(), cos, sin)[0, 0]
    assert abs((Q[3] @ K[1]).item() - (Q[12] @ K[10]).item()) < 1e-4, "the score only depends on m - n"
    x = torch.randn(2, 3, 10, 16)
    full = apply_rope(x, cos, sin)
    assert torch.allclose(apply_rope(x[:, :, 4:6], cos, sin, start_pos=4), full[:, :, 4:6], atol=1e-6)


def _ref_attention(m, x, cos, sin):
    B, T, _ = x.shape
    H, Hkv, hd = CFG.n_head, CFG.n_kv_head, CFG.head_dim
    q = m.wq(x).view(B, T, H, hd).transpose(1, 2)
    k = m.wk(x).view(B, T, Hkv, hd).transpose(1, 2)
    v = m.wv(x).view(B, T, Hkv, hd).transpose(1, 2)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    k, v = k.repeat_interleave(H // Hkv, 1), v.repeat_interleave(H // Hkv, 1)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return m.wo(y.transpose(1, 2).reshape(B, T, -1))


def test_attention_gqa_shapes():
    m = Attention(CFG)
    assert m.wk.out_features == CFG.n_kv_head * CFG.head_dim, "wk projects to n_kv_head * head_dim"
    assert m.wq.bias is None and m.wo.bias is None


def test_attention_matches_reference():
    torch.manual_seed(0)
    m = Attention(CFG).eval()
    cos, sin = precompute_rope(CFG.head_dim, 32)
    x = torch.randn(2, 9, CFG.n_embd)
    assert torch.allclose(m(x, cos, sin), _ref_attention(m, x, cos, sin), atol=1e-5)


def test_attention_is_causal():
    m = Attention(CFG).eval()
    cos, sin = precompute_rope(CFG.head_dim, 32)
    x = torch.randn(2, 10, CFG.n_embd)
    x2 = x.clone()
    x2[:, 6:] = torch.randn_like(x2[:, 6:])
    assert torch.allclose(m(x, cos, sin)[:, :6], m(x2, cos, sin)[:, :6], atol=1e-6), "the future must not leak"
