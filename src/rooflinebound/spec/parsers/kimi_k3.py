"""Hugging Face ``kimi_k3`` (vision + ``kimi_linear`` text) and ``kimi_linear`` configs.

Hybrid attention: Kimi Delta Attention (linear) in most layers, MLA in the
``full_attn_layers``; a latent MoE; learned attention residuals; MXFP4 routed
experts. Every shape is read from the config; the inventory is checked against
the checkpoint's safetensors headers.
"""

from __future__ import annotations

from collections.abc import Mapping

from ...arch.attention import KDA, MLA
from ...arch.ffn import DenseFFN, MoE
from ...arch.model import DecoderLayer, ModelArch
from ...arch.tensors import Quantization, Shard, TensorSpec, linear, vector
from ...arch.vision import MoonViT3d
from ...quantity import Provenance
from ..errors import SpecError
from .common import dense_dtype

MODEL_TYPES = ("kimi_k3", "kimi_linear")

_REQUIRED = (
    "hidden_size", "vocab_size", "num_hidden_layers", "num_attention_heads", "q_lora_rank",
    "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "intermediate_size",
    "moe_intermediate_size", "num_experts", "num_experts_per_token", "num_shared_experts",
    "first_k_dense_replace", "linear_attn_config",
)


def _quantization(cfg: Mapping) -> Quantization | None:
    """compressed-tensors MXFP4 (``mxfp4-pack-quantized``, group 32, e8m0 scales).

    The checkpoint quantizes routed experts only. The config's ``ignore`` list
    covers attention, shared experts, dense MLPs, lm_head and vision, but not
    the latent MoE projections, which the checkpoint nonetheless stores in bf16;
    so quantization is limited to routed experts, as the checkpoint shows.
    """
    q = cfg.get("quantization_config")
    if not q:
        return None
    if q.get("quant_method") != "compressed-tensors" or q.get("format") != "mxfp4-pack-quantized":
        raise SpecError(f"quantization {q.get('quant_method')}/{q.get('format')} is not "
                        "supported yet")
    w = next(iter(q["config_groups"].values()))["weights"]
    if w.get("num_bits") != 4 or w.get("type") != "float" or w.get("strategy") != "group":
        raise SpecError("only fp4 group quantization is supported for compressed-tensors")
    return Quantization("U8", (1, w["group_size"]), "U8", frozenset(), kind="mxfp4",
                        components=frozenset({"routed_expert"}))


def _vision(v: Mapping, text_hidden: int, ddtype: str) -> MoonViT3d:
    return MoonViT3d(v["vt_hidden_size"], v["vt_intermediate_size"],
                     v.get("qkv_hidden_size") or v["vt_hidden_size"],
                     v["vt_num_hidden_layers"], v["patch_size"], v["init_pos_emb_height"],
                     v["init_pos_emb_width"], tuple(v["merge_kernel_size"]), text_hidden, ddtype)


def parse(cfg: Mapping, name: str, provenance: Provenance) -> ModelArch:
    vision_cfg = None
    root = ""
    if cfg.get("model_type") == "kimi_k3":
        vision_cfg = cfg.get("vision_config")
        cfg = cfg["text_config"]
        root = "language_model."
    missing = [k for k in _REQUIRED if k not in cfg]
    if missing:
        raise SpecError(f"{name}: config is missing {missing}")
    la = cfg["linear_attn_config"]
    n = cfg["num_hidden_layers"]
    full = {i - 1 for i in la["full_attn_layers"]}  # 1-based in the config
    kda_ids = {i - 1 for i in la.get("kda_layers", [])} or (set(range(n)) - full)
    if full & kda_ids or len(full | kda_ids) != n:
        raise SpecError(f"{name}: full_attn_layers and kda_layers must partition {n} layers")
    if cfg.get("moe_layer_freq", 1) != 1 or cfg.get("num_expert_group", 1) != 1:
        raise SpecError(f"{name}: grouped or strided MoE layouts are not supported yet")
    if cfg.get("num_nextn_predict_layers", 0):
        raise SpecError(f"{name}: MTP layers are not modelled for this family yet")
    ddtype = dense_dtype(cfg, name)

    h = cfg["hidden_size"]
    mla = MLA(h, cfg["num_attention_heads"], cfg["q_lora_rank"], cfg["kv_lora_rank"],
              cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"],
              output_gate=bool(cfg.get("mla_use_output_gate")))
    kda = KDA(h, la["num_heads"], la["head_dim"], la["short_conv_kernel_size"],
              full_rank_gate=bool(la.get("use_full_rank_gate")))
    dense = DenseFFN(h, cfg["intermediate_size"])
    moe = MoE(h, cfg["num_experts"], cfg["num_experts_per_token"],
              cfg["moe_intermediate_size"], cfg["num_shared_experts"],
              score_bias=cfg.get("topk_method") == "noaux_tc",
              latent=cfg.get("routed_expert_hidden_size"),
              latent_norm=bool(cfg.get("latent_moe_use_norm")),
              module="block_sparse_moe.", expert_names=("w1", "w3", "w2"))
    attn_res = cfg.get("attn_res_block_size") is not None
    layers = tuple(
        DecoderLayer(i, kda if i in kda_ids else mla,
                     dense if i < cfg["first_k_dense_replace"] else moe,
                     attn_res=attn_res, root=root)
        for i in range(n)
    )
    extra_first: tuple[TensorSpec, ...] = ()
    if vision_cfg is not None:
        extra_first = tuple(_vision(vision_cfg, h, ddtype).tensors())
    extra_last: tuple[TensorSpec, ...] = ()
    if attn_res:
        extra_last = (
            vector(f"{root}model.", "output_attn_res_norm", h, component="norm", layer=None,
                   dtype=ddtype),
            *linear(f"{root}model.", "output_attn_res_proj", 1, h, component="norm",
                    shard=Shard.REPLICATED, layer=None, quant=None, dense_dtype=ddtype),
        )
    return ModelArch(name, h, cfg["vocab_size"], bool(cfg.get("tie_word_embeddings", False)),
                     layers, (), _quantization(cfg), ddtype, provenance,
                     source_fields={k: cfg[k] for k in _REQUIRED if k != "linear_attn_config"},
                     root=root, extra_first=extra_first, extra_last=extra_last)
