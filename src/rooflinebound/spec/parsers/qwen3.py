"""Hugging Face ``qwen3`` configs: dense GQA with per-head QK RMSNorm."""

from __future__ import annotations

from collections.abc import Mapping

from ...arch.attention import GQA
from ...arch.ffn import DenseFFN
from ...arch.model import DecoderLayer, ModelArch
from ...quantity import Provenance
from ..errors import SpecError
from .common import dense_dtype, quantization

MODEL_TYPES = ("qwen3",)

_REQUIRED = ("hidden_size", "vocab_size", "num_hidden_layers", "num_attention_heads",
             "num_key_value_heads", "intermediate_size")


def parse(cfg: Mapping, name: str, provenance: Provenance) -> ModelArch:
    missing = [k for k in _REQUIRED if k not in cfg]
    if missing:
        raise SpecError(f"{name}: config is missing {missing}")
    if cfg.get("attention_bias"):
        raise SpecError(f"{name}: attention biases are not supported yet")
    if cfg.get("use_sliding_window"):
        raise SpecError(f"{name}: sliding-window attention is not modelled yet")
    h, heads = cfg["hidden_size"], cfg["num_attention_heads"]
    head_dim = cfg.get("head_dim") or h // heads
    attn = GQA(h, heads, cfg["num_key_value_heads"], head_dim, qk_norm=True)
    ffn = DenseFFN(h, cfg["intermediate_size"])
    layers = tuple(DecoderLayer(i, attn, ffn) for i in range(cfg["num_hidden_layers"]))
    return ModelArch(name, h, cfg["vocab_size"], bool(cfg.get("tie_word_embeddings", False)),
                     layers, (), quantization(cfg), dense_dtype(cfg, name), provenance,
                     source_fields={k: cfg[k] for k in _REQUIRED})
