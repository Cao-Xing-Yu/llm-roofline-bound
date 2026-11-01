"""Hugging Face ``glm_moe_dsa`` configs (GLM-5.x): MLA + DSA indexer + MoE + MTP."""

from __future__ import annotations

from collections.abc import Mapping

from ...arch.attention import MLA, DSAIndexer
from ...arch.ffn import DenseFFN, MoE
from ...arch.model import DecoderLayer, ModelArch
from ...quantity import Provenance
from ..errors import SpecError
from .common import dense_dtype, quantization

MODEL_TYPES = ("glm_moe_dsa",)

_REQUIRED = (
    "hidden_size", "vocab_size", "num_hidden_layers", "num_attention_heads",
    "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
    "intermediate_size", "moe_intermediate_size", "n_routed_experts", "num_experts_per_tok",
    "n_shared_experts", "index_n_heads", "index_head_dim", "index_topk",
    "mlp_layer_types", "indexer_types",
)

def parse(cfg: Mapping, name: str, provenance: Provenance) -> ModelArch:
    missing = [k for k in _REQUIRED if k not in cfg]
    if missing:
        raise SpecError(f"{name}: config is missing {missing}")
    n = cfg["num_hidden_layers"]
    mlp_types, idx_types = cfg["mlp_layer_types"], cfg["indexer_types"]
    if len(mlp_types) != n or len(idx_types) != n:
        raise SpecError(f"{name}: mlp_layer_types / indexer_types must have {n} entries")
    bad = {t for t in mlp_types if t not in ("dense", "sparse")} | \
          {t for t in idx_types if t not in ("full", "shared")}
    if bad:
        raise SpecError(f"{name}: unsupported layer types {sorted(bad)}")
    if cfg.get("n_group", 1) != 1 or cfg.get("topk_group", 1) != 1:
        raise SpecError(f"{name}: grouped expert routing is not supported yet")
    ddtype = dense_dtype(cfg, name)

    h = cfg["hidden_size"]
    attn = MLA(h, cfg["num_attention_heads"], cfg["q_lora_rank"], cfg["kv_lora_rank"],
               cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"])
    indexer = DSAIndexer(h, cfg["q_lora_rank"], cfg["index_n_heads"], cfg["index_head_dim"],
                         cfg["index_topk"])
    dense = DenseFFN(h, cfg["intermediate_size"])
    moe = MoE(h, cfg["n_routed_experts"], cfg["num_experts_per_tok"],
              cfg["moe_intermediate_size"], cfg["n_shared_experts"],
              score_bias=cfg.get("topk_method") == "noaux_tc")

    layers = tuple(
        DecoderLayer(i, attn, dense if mlp_types[i] == "dense" else moe,
                     indexer if idx_types[i] == "full" else None, attn_topk=indexer.topk)
        for i in range(n)
    )
    # Every layer runs top-k sparse attention: "full" layers select with their own
    # indexer, "shared" layers reuse that selection. MTP layers follow the main
    # stack, with a MoE FFN and their own indexer.
    mtp = tuple(
        DecoderLayer(n + j, attn, moe, indexer, is_mtp=True, attn_topk=indexer.topk)
        for j in range(cfg.get("num_nextn_predict_layers", 0))
    )
    return ModelArch(name, h, cfg["vocab_size"], bool(cfg.get("tie_word_embeddings", False)),
                     layers, mtp, quantization(cfg), ddtype, provenance,
                     source_fields={k: cfg[k] for k in _REQUIRED if k in cfg})
