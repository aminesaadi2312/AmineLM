"""Séance M6 · pytest tests/test_generate.py"""
import torch
from minigpt import GPTConfig
from minigpt.model import GPT
from minigpt.layers import KVCache
from minigpt.sampling import filter_logits, sample_next

CFG = GPTConfig(vocab_size=30, block_size=48, n_layer=2, n_head=4, n_kv_head=2, n_embd=32)


def test_filter_top_k_et_top_p():
    lg = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0]])
    assert torch.isinf(filter_logits(lg, top_k=2)).sum() == 3
    p = lg.softmax(-1)[0]                                  # 0.64, 0.24, 0.09, …
    kept = ~torch.isinf(filter_logits(lg, top_p=0.7))[0]
    assert kept.tolist() == [True, True, False, False, False], "top-p : on garde jusqu'à franchir 0,7"
    assert torch.equal(filter_logits(lg), lg)


def test_sample_next():
    lg = torch.randn(4, 10)
    assert torch.equal(sample_next(lg, temperature=0), lg.argmax(-1))
    base = torch.tensor([2.0, 1.0, 0.5, 0.0])
    g = torch.Generator().manual_seed(0)
    s = sample_next(base.expand(20000, 4), temperature=0.5, generator=g)
    freq = torch.bincount(s, minlength=4).float() / 20000
    assert torch.allclose(freq, (base / 0.5).softmax(-1), atol=0.02), "la température divise les logits"


def test_kvcache_update():
    c = KVCache(2, 2, 10, 4)
    k1, v1 = c.update(torch.ones(2, 2, 3, 4), torch.ones(2, 2, 3, 4), 0)
    assert k1.shape == (2, 2, 3, 4)
    k2, _ = c.update(2 * torch.ones(1, 2, 1, 4), 2 * torch.ones(1, 2, 1, 4), 3)
    assert k2.shape == (1, 2, 4, 4) and k2[0, 0, 3, 0] == 2 and k2[0, 0, 0, 0] == 1


def test_cache_identique_sans_cache_et_un_token_par_pas():
    torch.manual_seed(0)
    m = GPT(CFG).eval()
    idx = torch.randint(0, 30, (2, 6))
    lens = []
    h = m.tok_embeddings.register_forward_hook(lambda mod, i, o: lens.append(i[0].size(1)))
    a = m.generate(idx.clone(), 20, temperature=0, use_cache=False)
    lens.clear()
    b = m.generate(idx.clone(), 20, temperature=0, use_cache=True)
    h.remove()
    assert torch.equal(a, b), "même sortie gloutonne avec et sans cache"
    assert lens[0] == 6 and all(n == 1 for n in lens[1:]), f"après le prefill, un token par appel (vu {lens[:5]})"


def test_generation_longue_au_dela_du_contexte():
    m = GPT(CFG).eval()
    out = m.generate(torch.zeros(1, 1, dtype=torch.long), 120, temperature=1.0, top_k=5)
    assert out.shape == (1, 121)
