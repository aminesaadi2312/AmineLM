"""Configuration du modèle (fournie, rien à coder ici)."""
import math
from dataclasses import dataclass, asdict


@dataclass
class GPTConfig:
    vocab_size: int = 65         # taille du vocabulaire (65 caractères pour Tiny Shakespeare)
    block_size: int = 256        # contexte maximal, aussi la taille du KV cache
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 2           # GQA : n_head doit être un multiple de n_kv_head
    n_embd: int = 384
    d_ff: int | None = None      # None → multiple de 64 proche de 8/3 · n_embd (comme Mistral)
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_weights: bool = True
    norm_eps: float = 1e-6
    flash: bool = False          # séance M7 : utiliser F.scaled_dot_product_attention
    use_rope: bool = True        # séance M7 (ablation) : False = sans information de position explicite
    post_norm: bool = False      # séance M7 (ablation) : bloc post-norm du Transformer de 2017

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd doit être divisible par n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head doit être un multiple de n_kv_head"
        assert self.head_dim % 2 == 0, "head_dim doit être pair pour RoPE"

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def ffn_dim(self) -> int:
        return self.d_ff or 64 * math.ceil(8 * self.n_embd / 3 / 64)

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    # tourne en quelques minutes sur CPU ou sur un Mac (MPS)
    "tiny": dict(block_size=128, n_layer=4, n_head=4, n_kv_head=2, n_embd=128),
    # ~10 M de paramètres : un GPU (Colab T4, carte perso) pendant 15–30 minutes
    "small": dict(block_size=256, n_layer=6, n_head=6, n_kv_head=2, n_embd=384, dropout=0.1),
}
