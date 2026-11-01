"""Attention components: Multi-head Latent Attention and the DSA indexer."""

from __future__ import annotations

from dataclasses import dataclass

from .tensors import Quantization, Shard, TensorSpec, linear, vector

__all__ = ["MLA", "GQA", "KDA", "DSAIndexer"]


@dataclass(frozen=True)
class GQA:
    """Grouped-query attention (MHA when ``kv_heads == heads``, MQA when 1).

    The KV cache holds a key and a value of ``head_dim`` per KV head, per token
    and layer. Tensor parallelism splits KV heads across ranks; with more ranks
    than KV heads, each rank keeps one head (SGLang ``get_num_kv_heads``).
    ``qk_norm`` adds per-head RMSNorm weights on queries and keys (Qwen3).
    """

    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    qk_norm: bool = False

    @property
    def kv_elements_per_token(self) -> int:
        return 2 * self.kv_heads * self.head_dim

    def kv_heads_per_rank(self, tp: int) -> int:
        return max(1, self.kv_heads // tp)

    def tensors(self, prefix: str, layer: int, quant: Quantization | None,
                dense_dtype: str) -> list[TensorSpec]:
        c = "attn"
        kw = dict(component=c, layer=layer, quant=quant, dense_dtype=dense_dtype)
        kv = dict(shard=Shard.KV_HEAD, heads=self.kv_heads)
        out = [
            *linear(prefix, "self_attn.q_proj", self.heads * self.head_dim, self.hidden,
                    shard=Shard.COL, **kw),
            *linear(prefix, "self_attn.k_proj", self.kv_heads * self.head_dim, self.hidden,
                    **kv, **kw),
            *linear(prefix, "self_attn.v_proj", self.kv_heads * self.head_dim, self.hidden,
                    **kv, **kw),
            *linear(prefix, "self_attn.o_proj", self.hidden, self.heads * self.head_dim,
                    shard=Shard.ROW, **kw),
        ]
        if self.qk_norm:
            out += [vector(prefix, "self_attn.q_norm", self.head_dim, component=c, layer=layer,
                           dtype=dense_dtype),
                    vector(prefix, "self_attn.k_norm", self.head_dim, component=c, layer=layer,
                           dtype=dense_dtype)]
        return out


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
    output_gate: bool = False  # a sigmoid gate on the attention output (Kimi)

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
            *(linear(prefix, "self_attn.g_proj", self.heads * self.v_head_dim, self.hidden,
                     shard=Shard.COL, **kw) if self.output_gate else []),
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


@dataclass(frozen=True)
class KDA:
    """Kimi Delta Attention: gated delta-rule linear attention.

    No per-token KV cache. Each sequence carries, per layer, a recurrent state
    of ``heads x head_dim x head_dim`` and a short-convolution window of
    ``conv_kernel - 1`` past q/k/v inputs; both are split across TP by head.
    Tensor names and dtypes follow the Kimi-K3 checkpoint.
    """

    hidden: int
    heads: int
    head_dim: int
    conv_kernel: int
    full_rank_gate: bool = True

    @property
    def proj(self) -> int:
        return self.heads * self.head_dim

    def state_bytes(self, tp: int, ssm_bytes: int, conv_bytes: int) -> int:
        """Per sequence, per layer, on one TP rank (SGLang KimiLinearStateShape)."""
        temporal = (self.heads // tp) * self.head_dim * self.head_dim * ssm_bytes
        conv = (self.conv_kernel - 1) * 3 * (self.proj // tp) * conv_bytes
        return temporal + conv

    def tensors(self, prefix: str, layer: int, quant: Quantization | None,
                dense_dtype: str) -> list[TensorSpec]:
        c = "attn"
        kw = dict(component=c, layer=layer, quant=quant, dense_dtype=dense_dtype)
        h, p, d = self.hidden, self.proj, self.head_dim
        out = [
            *linear(prefix, "self_attn.q_proj", p, h, shard=Shard.COL, **kw),
            *linear(prefix, "self_attn.k_proj", p, h, shard=Shard.COL, **kw),
            *linear(prefix, "self_attn.v_proj", p, h, shard=Shard.COL, **kw),
            *[TensorSpec(f"{prefix}self_attn.{x}_conv1d.weight", (p, 1, self.conv_kernel), "F32",
                         c, Shard.COL, layer) for x in "qkv"],
            # per-channel decay base; stored at head_dim in the checkpoint
            TensorSpec(f"{prefix}self_attn.A_log", (d,), "F32", c, Shard.REPLICATED, layer),
            *linear(prefix, "self_attn.f_a_proj", d, h, shard=Shard.REPLICATED, **kw),
            *linear(prefix, "self_attn.f_b_proj", p, d, shard=Shard.COL, **kw),
            TensorSpec(f"{prefix}self_attn.dt_bias", (p,), "F32", c, Shard.COL, layer),
            *linear(prefix, "self_attn.b_proj", self.heads, h, shard=Shard.COL, **kw),
            TensorSpec(f"{prefix}self_attn.o_norm.weight", (d,), "F32", c, Shard.REPLICATED,
                       layer),
            *linear(prefix, "self_attn.o_proj", h, p, shard=Shard.ROW, **kw),
        ]
        if self.full_rank_gate:
            out += linear(prefix, "self_attn.g_proj", p, h, shard=Shard.COL, **kw)
        else:
            out += [*linear(prefix, "self_attn.g_a_proj", d, h, shard=Shard.REPLICATED, **kw),
                    *linear(prefix, "self_attn.g_b_proj", p, d, shard=Shard.COL, **kw)]
        return out
