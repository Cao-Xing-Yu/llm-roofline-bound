"""Attention components: Multi-head Latent Attention and the DSA indexer."""

from __future__ import annotations

from dataclasses import dataclass

from .tensors import Quantization, Shard, TensorSpec, linear, vector

__all__ = ["MLA", "DSAIndexer"]


@dataclass(frozen=True)
class MLA:
    """Multi-head Latent Attention (DeepSeek-V2 style) with a compressed query.

    The KV cache holds, per token and layer, the ``kv_lora_rank`` latent plus
    the ``qk_rope_head_dim`` decoupled rope key, shared by all heads. Under
    tensor parallelism every rank needs the whole latent, so the cache is
    replicated across the TP group, not split.
    """

    hidden: int
    heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def kv_elements_per_token(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    def tensors(self, prefix: str, layer: int, quant: Quantization | None,
                dense_dtype: str) -> list[TensorSpec]:
        c = "attn"
        kw = dict(component=c, layer=layer, quant=quant, dense_dtype=dense_dtype)
        return [
            # Down-projections produce the shared latents: replicated.
            *linear(prefix, "self_attn.q_a_proj", self.q_lora_rank, self.hidden,
                    shard=Shard.REPLICATED, **kw),
            vector(prefix, "self_attn.q_a_layernorm", self.q_lora_rank, component=c,
                   layer=layer, dtype=dense_dtype),
            *linear(prefix, "self_attn.q_b_proj", self.heads * self.qk_head_dim,
                    self.q_lora_rank, shard=Shard.COL, **kw),
            *linear(prefix, "self_attn.kv_a_proj_with_mqa", self.kv_elements_per_token,
                    self.hidden, shard=Shard.REPLICATED, **kw),
            vector(prefix, "self_attn.kv_a_layernorm", self.kv_lora_rank, component=c,
                   layer=layer, dtype=dense_dtype),
            *linear(prefix, "self_attn.kv_b_proj",
                    self.heads * (self.qk_nope_head_dim + self.v_head_dim),
                    self.kv_lora_rank, shard=Shard.COL, **kw),
            *linear(prefix, "self_attn.o_proj", self.hidden, self.heads * self.v_head_dim,
                    shard=Shard.ROW, **kw),
        ]


@dataclass(frozen=True)
class DSAIndexer:
    """DeepSeek Sparse Attention lightning indexer.

    Scores every earlier token with a small multi-head dot product and keeps
    the ``topk`` best for the main attention. It caches one ``head_dim`` key
    per token. Its weights are small and replicated on every TP rank.
    """

    hidden: int
    q_lora_rank: int
    heads: int
    head_dim: int
    topk: int

    @property
    def kv_elements_per_token(self) -> int:
        return self.head_dim

    def tensors(self, prefix: str, layer: int, quant: Quantization | None,
                dense_dtype: str) -> list[TensorSpec]:
        c = "indexer"
        kw = dict(component=c, layer=layer, quant=quant, dense_dtype=dense_dtype,
                  shard=Shard.REPLICATED)
        return [
            *linear(prefix, "self_attn.indexer.wq_b", self.heads * self.head_dim,
                    self.q_lora_rank, **kw),
            *linear(prefix, "self_attn.indexer.wk", self.head_dim, self.hidden, **kw),
            vector(prefix, "self_attn.indexer.k_norm", self.head_dim, component=c,
                   layer=layer, dtype=dense_dtype),
            vector(prefix, "self_attn.indexer.k_norm", self.head_dim, component=c,
                   layer=layer, dtype=dense_dtype, suffix="bias"),
            # Checkpoints list this one as "indexers_proj" in modules_to_not_convert.
            *linear(prefix, "self_attn.indexer.weights_proj", self.heads, self.hidden,
                    quant_names=("self_attn.indexers_proj",), **kw),
        ]
