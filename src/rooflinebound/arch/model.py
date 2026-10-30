"""A model assembled from components, independent of any one checkpoint format."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cached_property

from ..quantity import Provenance
from .attention import MLA, DSAIndexer
from .ffn import DenseFFN, MoE
from .tensors import Quantization, Shard, TensorSpec, linear, vector

__all__ = ["DecoderLayer", "ModelArch"]


@dataclass(frozen=True)
class DecoderLayer:
    index: int
    attn: MLA
    ffn: DenseFFN | MoE
    # The indexer whose weights and key cache this layer owns. A layer that reuses
    # another layer's top-k selection ("shared") has none.
    indexer: DSAIndexer | None = None
    is_mtp: bool = False

    @property
    def prefix(self) -> str:
        return f"model.layers.{self.index}."

    def tensors(self, quant: Quantization | None, dense_dtype: str) -> list[TensorSpec]:
        p, i = self.prefix, self.index
        out = [vector(p, "input_layernorm", self.attn.hidden, component="norm", layer=i,
                      dtype=dense_dtype)]
        out += self.attn.tensors(p, i, quant, dense_dtype)
        if self.indexer is not None:
            out += self.indexer.tensors(p, i, quant, dense_dtype)
        out.append(vector(p, "post_attention_layernorm", self.attn.hidden, component="norm",
                          layer=i, dtype=dense_dtype))
        out += self.ffn.tensors(p, i, quant, dense_dtype)
        if self.is_mtp:
            out += self._mtp_tensors(quant, dense_dtype)
        return out

    def _mtp_tensors(self, quant: Quantization | None, dense_dtype: str) -> list[TensorSpec]:
        """Next-token-prediction head glue (DeepSeek-V3 style).

        The MTP layer shares the main model's embedding and lm_head; it adds two
        input norms, a projection of [embedding; hidden] back to hidden, and a
        final norm.
        """
        p, i, h = self.prefix, self.index, self.attn.hidden
        return [
            vector(p, "enorm", h, component="mtp", layer=i, dtype=dense_dtype),
            vector(p, "hnorm", h, component="mtp", layer=i, dtype=dense_dtype),
            *linear(p, "eh_proj", h, 2 * h, component="mtp", shard=Shard.REPLICATED, layer=i,
                    quant=quant, dense_dtype=dense_dtype),
            vector(p, "shared_head.norm", h, component="mtp", layer=i, dtype=dense_dtype),
        ]


@dataclass(frozen=True)
class ModelArch:
    name: str
    hidden: int
    vocab: int
    tie_embeddings: bool
    layers: tuple[DecoderLayer, ...]
    mtp_layers: tuple[DecoderLayer, ...]
    quant: Quantization | None
    dense_dtype: str
    provenance: Provenance
    source_fields: dict = field(default_factory=dict, compare=False)

    @property
    def all_layers(self) -> tuple[DecoderLayer, ...]:
        return self.layers + self.mtp_layers

    def global_tensors(self) -> tuple[list[TensorSpec], list[TensorSpec]]:
        """(first-stage tensors, last-stage tensors) outside the decoder layers."""
        # Embeddings are looked up, not multiplied, so block quantization of
        # linear layers does not apply to them.
        first = [TensorSpec("model.embed_tokens.weight", (self.vocab, self.hidden),
                            self.dense_dtype, "embed", Shard.VOCAB)]
        last = [vector("model.", "norm", self.hidden, component="norm", layer=None,
                       dtype=self.dense_dtype)]
        if not self.tie_embeddings:
            last += linear("", "lm_head", self.vocab, self.hidden, component="lm_head",
                           shard=Shard.VOCAB, layer=None, quant=self.quant,
                           dense_dtype=self.dense_dtype)
        return first, last

    @cached_property
    def inventory(self) -> tuple[TensorSpec, ...]:
        """Every tensor of the checkpoint, MTP layers included."""
        first, last = self.global_tensors()
        out = list(first)
        for layer in self.all_layers:
            out += layer.tensors(self.quant, self.dense_dtype)
        out += last
        return tuple(out)

    def tensors(self, *, mtp: bool = True) -> Iterator[TensorSpec]:
        mtp_ids = {layer.index for layer in self.mtp_layers}
        for t in self.inventory:
            if mtp or t.layer not in mtp_ids:
                yield t
