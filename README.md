# mini-GPT

A small Mistral-style language model written from scratch in PyTorch: byte-level BPE and character tokenizers,
RMSNorm, rotary embeddings (RoPE), grouped-query attention (GQA), SwiGLU, pre-norm blocks, weight tying,
a clean training loop (AdamW, warmup + cosine, gradient clipping, bf16) and generation with a KV cache.

<!-- Séance M8 : remplace les crochets par tes propres résultats. -->

## Results

| run | params | val loss | tokens/s (train) | notes |
|---|---|---|---|---|
| tiny, GQA 4/2 | [..] M | [..] | [..] | [device] |
| small, GQA 6/2 | [..] M | [..] | [..] | |

Sample after training:

```
[colle ici un extrait généré]
```

## Architecture

`tok_embeddings → N × [x + Attention(RMSNorm(x)) ; h + SwiGLU(RMSNorm(h))] → RMSNorm → lm_head (tied)`

- Attention: causal, GQA (`n_kv_head < n_head`), RoPE on q and k (interleaved pairs, as in mistral-inference).
- KV cache: one prefill pass, then one token per step; [x]× faster generation measured.

## Run it

```bash
pip install -r requirements.txt
python scripts/prepare_shakespeare.py
python scripts/train.py --preset tiny --max_iters 2000
python scripts/sample.py --prompt "ROMEO:" --n 400 --top_k 50
pytest -q
```

## Ablations

[GQA vs MHA vs MQA, RoPE, pre-norm vs post-norm, weight tying : courbes dans `assets/`, une phrase de conclusion chacune.]

## References

- Jiang et al., *Mistral 7B* (2023) · [mistral-inference](https://github.com/mistralai/mistral-inference)
- Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT) and [minbpe](https://github.com/karpathy/minbpe)
- Su et al., *RoFormer* (RoPE, 2021) · Shazeer, *GLU Variants Improve Transformer* (2020) · Ainslie et al., *GQA* (2023)
