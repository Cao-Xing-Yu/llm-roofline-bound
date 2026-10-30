"""Tensor inventory: every stored tensor of a checkpoint, with how it shards.

A model's weights are described tensor by tensor, the way its safetensors
checkpoint stores them, so that memory figures are sums over real tensors
rather than ``params x bytes`` and can be checked against checkpoint headers.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

__all__ = ["DTYPE_BYTES", "Shard", "TensorSpec", "Quantization", "linear"]

# safetensors dtype names -> bytes per element
DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "BF16": 2,
    "F16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I64": 8,
    "I32": 4,
    "I8": 1,
    "U8": 1,
}


class Shard(enum.Enum):
    """How a tensor is laid out across the ranks of a tensor-parallel group."""

    REPLICATED = "replicated"  # full copy on every rank
    COL = "col"  # dim 0 (output features) split
    ROW = "row"  # dim 1 (input features) split
    VOCAB = "vocab"  # dim 0 split, padded up to a multiple of the group size
    EXPERT = "expert"  # one routed expert's tensor; placed by expert parallelism


@dataclass(frozen=True)
class TensorSpec:
    name: str  # checkpoint name, e.g. "model.layers.3.mlp.experts.7.up_proj.weight"
    shape: tuple[int, ...]
    dtype: str
    component: str  # accounting bucket: "attn", "indexer", "dense_ffn", "routed_expert", ...
    shard: Shard
    layer: int | None = None  # decoder layer index, None for embed / final norm / lm_head
    is_scale: bool = False  # a quantization scale, not a model parameter
    expert_shard: Shard | None = None  # COL/ROW split of an expert tensor when EP = 1

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.numel * DTYPE_BYTES[self.dtype]


@dataclass(frozen=True)
class Quantization:
    """Block-wise weight quantization as declared in ``quantization_config``.

    ``skip`` holds module names (layer prefix stripped) that stay unquantized.
    """

    weight_dtype: str  # e.g. "F8_E4M3"
    block: tuple[int, int]  # (rows, cols) per scale
    scale_dtype: str  # e.g. "F32"
    skip: frozenset[str]

    def applies_to(self, module: str) -> bool:
        return module not in self.skip


def linear(
    prefix: str,
    module: str,
    out_features: int,
    in_features: int,
    *,
    component: str,
    shard: Shard,
    layer: int | None,
    quant: Quantization | None,
    dense_dtype: str = "BF16",
    expert_shard: Shard | None = None,
    quant_names: tuple[str, ...] = (),
) -> list[TensorSpec]:
    """A bias-free linear layer: its weight, plus a block scale if quantized.

    ``module`` is the name relative to the layer (``self_attn.q_b_proj``);
    ``quant_names`` lists other names under which the checkpoint's
    ``modules_to_not_convert`` may refer to it.
    """
    name = f"{prefix}{module}.weight"
    names = (module, *quant_names)
    if quant is None or not all(quant.applies_to(n) for n in names):
        return [TensorSpec(name, (out_features, in_features), dense_dtype, component, shard,
                           layer, expert_shard=expert_shard)]
    br, bc = quant.block
    return [
        TensorSpec(name, (out_features, in_features), quant.weight_dtype, component, shard,
                   layer, expert_shard=expert_shard),
        TensorSpec(f"{prefix}{module}.weight_scale_inv",
                   (math.ceil(out_features / br), math.ceil(in_features / bc)),
                   quant.scale_dtype, component, shard, layer, is_scale=True,
                   expert_shard=expert_shard),
    ]


def vector(prefix: str, module: str, size: int, *, component: str, layer: int | None,
           dtype: str = "BF16", suffix: str = "weight") -> TensorSpec:
    """A 1-D tensor (norm weight, bias); always replicated."""
    return TensorSpec(f"{prefix}{module}.{suffix}", (size,), dtype, component,
                      Shard.REPLICATED, layer)
