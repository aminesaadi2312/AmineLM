"""Model configuration and presets."""
import math
from dataclasses import dataclass, asdict


@dataclass
class GPTConfig:
    vocab_size: int = 65         # 65 characters for Tiny Shakespeare
    block_size: int = 256        # max context length, also the size of the KV cache
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 2           # GQA: n_head must be a multiple of n_kv_head
    n_embd: int = 384
    d_ff: int | None = None      # None -> multiple of 64 close to 8/3 * n_embd (as in Mistral)
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_weights: bool = True
    norm_eps: float = 1e-6
    flash: bool = False          # use F.scaled_dot_product_attention
    use_rope: bool = True        # ablation: False = no explicit position information
    post_norm: bool = False      # ablation: post-norm blocks of the 2017 Transformer

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be a multiple of n_kv_head"
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def ffn_dim(self) -> int:
        return self.d_ff or 64 * math.ceil(8 * self.n_embd / 3 / 64)

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    # a few minutes on a laptop CPU or a Mac (MPS)
    "tiny": dict(block_size=128, n_layer=4, n_head=4, n_kv_head=2, n_embd=128),
    # ~10M parameters: one GPU (Colab T4 or similar) for 15-30 minutes
    "small": dict(block_size=256, n_layer=6, n_head=6, n_kv_head=2, n_embd=384, dropout=0.1),
}
