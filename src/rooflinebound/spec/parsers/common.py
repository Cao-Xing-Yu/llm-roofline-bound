"""Pieces of Hugging Face config parsing shared by every parser."""

from __future__ import annotations

import re
from collections.abc import Mapping

from ...arch.tensors import Quantization
from ..errors import SpecError

_DTYPES = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}
_FP8_FORMATS = {"e4m3": "F8_E4M3", "e5m2": "F8_E5M2"}
_LAYER_PREFIX = re.compile(r"^model\.layers\.\d+\.")


def dense_dtype(cfg: Mapping, name: str) -> str:
    """Storage dtype of unquantized tensors (``dtype``, or the older ``torch_dtype``)."""
    raw = cfg.get("dtype") or cfg.get("torch_dtype") or "bfloat16"
    if raw not in _DTYPES:
        raise SpecError(f"{name}: unsupported dtype {raw!r}")
    return _DTYPES[raw]


def quantization(cfg: Mapping) -> Quantization | None:
    """Block-wise fp8 from ``quantization_config``; None for an unquantized checkpoint."""
    q = cfg.get("quantization_config")
    if not q:
        return None
    if q.get("quant_method") != "fp8":
        raise SpecError(f"quantization method {q.get('quant_method')!r} is not supported yet")
    fmt = q.get("fmt", "e4m3")
    if fmt not in _FP8_FORMATS or "weight_block_size" not in q:
        raise SpecError("only block-wise fp8 (e4m3/e5m2, weight_block_size) is supported")
    skip = frozenset(_LAYER_PREFIX.sub("", m) for m in q.get("modules_to_not_convert") or ())
    # Block-fp8 checkpoints store the inverse scale of each block as float32.
    return Quantization(_FP8_FORMATS[fmt], tuple(q["weight_block_size"]), "F32", skip)
