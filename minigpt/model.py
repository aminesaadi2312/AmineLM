"""The model: pre-norm transformer blocks, the full GPT, its optimizer and generation with a KV cache."""
import inspect
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig
from .layers import RMSNorm, Attention, FeedForward, KVCache, precompute_rope
from .sampling import sample_next


class Block(nn.Module):
    """Pre-norm block, with the same names as mistral-inference:
    h = x + attention(attention_norm(x));  out = h + feed_forward(ffn_norm(h)).
    With cfg.post_norm, the norm is applied after each residual sum instead (2017 Transformer)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.post_norm = cfg.post_norm
        self.attention_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attention = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.feed_forward = FeedForward(cfg)

    def forward(self, x, cos, sin, start_pos=0, cache=None):
        if self.post_norm:
            h = self.attention_norm(x + self.attention(x, cos, sin, start_pos, cache))
            return self.ffn_norm(h + self.feed_forward(h))
        h = x + self.attention(self.attention_norm(x), cos, sin, start_pos, cache)
        return h + self.feed_forward(self.ffn_norm(h))


class GPT(nn.Module):
    """tok_embeddings -> n_layer blocks -> final RMSNorm -> lm_head (tied to the embeddings by default)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_embeddings.weight      # one matrix for input and output
        self.apply(self._init_weights)
        # RoPE angles are not parameters: buffers move with .to(device) but are not trained or saved
        cos, sin = precompute_rope(cfg.head_dim, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_embeddings.weight.numel()
        return n

    def forward(self, idx, targets=None, start_pos=0, caches=None):
        """idx: (B, T) token ids -> (logits (B, T, V), loss or None).
        caches: one KVCache per layer during generation, else None."""
        B, T = idx.shape
        assert start_pos + T <= self.cfg.block_size, f"sequence of {start_pos + T} tokens > block_size"
        x = self.drop(self.tok_embeddings(idx))
        for i, layer in enumerate(self.layers):
            x = layer(x, self.rope_cos, self.rope_sin, start_pos, None if caches is None else caches[i])
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def configure_optimizer(self, weight_decay, lr, betas=(0.9, 0.95), device_type="cpu"):
        """AdamW with two groups: weight decay on matrices (dim >= 2), none on norm weights."""
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ]
        fused = device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters
        return torch.optim.AdamW(groups, lr=lr, betas=betas, **({"fused": True} if fused else {}))

    def new_caches(self, batch_size, device=None, dtype=None):
        """One KVCache per layer, sized for the full context (block_size)."""
        w = self.tok_embeddings.weight
        return [KVCache(batch_size, self.cfg.n_kv_head, self.cfg.block_size, self.cfg.head_dim,
                        device=device or w.device, dtype=dtype or w.dtype) for _ in self.layers]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, use_cache=True, generator=None):
        """Append max_new_tokens tokens to idx (B, T).

        With use_cache: one prefill pass on the prompt, then a single new token per forward call.
        When the context is full, keep the last half and re-prefill (simple sliding window).
        Without cache: recompute the whole (cropped) sequence at every step.
        """
        block = self.cfg.block_size
        if not use_cache:
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -block:])
                idx = torch.cat([idx, sample_next(logits[:, -1], temperature, top_k, top_p, generator)[:, None]], dim=1)
            return idx

        caches, pos, nxt = None, 0, None
        for _ in range(max_new_tokens):
            if caches is None or pos >= block:
                context = idx[:, -(block // 2):] if pos >= block else idx[:, -block:]
                caches = self.new_caches(idx.size(0))
                logits, _ = self(context, start_pos=0, caches=caches)           # prefill
                pos = context.size(1)
            else:
                logits, _ = self(nxt[:, None], start_pos=pos, caches=caches)    # one token
                pos += 1
            nxt = sample_next(logits[:, -1], temperature, top_k, top_p, generator)
            idx = torch.cat([idx, nxt[:, None]], dim=1)
        return idx
