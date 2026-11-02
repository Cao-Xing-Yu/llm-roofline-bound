"""Memory and layout on a toy glm_moe_dsa model, against hand-written arithmetic.

The expected numbers here are written out from the architecture definition,
not computed by the package, so a bug in the inventory or the sharding rules
shows up as a mismatch.
"""

import pytest

from rooflinebound.memory import (
    Deployment,
    active_params_per_token,
    memory_report,
    param_counts,
)
from rooflinebound.parallel import Parallel
from rooflinebound.quantity import Level, Provenance, Quantity, Trust
from rooflinebound.spec import SpecError, load_model
from rooflinebound.spec.hardware import parse_hardware
from rooflinebound.spec.models import parse_config

H, V = 256, 1000
HEADS, QL, KVL, NOPE, ROPE, VD = 4, 64, 32, 16, 8, 16
INTER, MOE_I, E, K, SHARED = 512, 128, 8, 2, 1
IH, ID = 2, 16

TOY = {
    "model_type": "glm_moe_dsa", "dtype": "bfloat16",
    "hidden_size": H, "vocab_size": V, "num_hidden_layers": 4,
    "num_attention_heads": HEADS, "q_lora_rank": QL, "kv_lora_rank": KVL,
    "qk_nope_head_dim": NOPE, "qk_rope_head_dim": ROPE, "v_head_dim": VD,
    "intermediate_size": INTER, "moe_intermediate_size": MOE_I, "n_routed_experts": E,
    "num_experts_per_tok": K, "n_shared_experts": SHARED, "topk_method": "noaux_tc",
    "index_n_heads": IH, "index_head_dim": ID, "index_topk": 16,
    "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
    "indexer_types": ["full", "shared", "full", "shared"],
    "num_nextn_predict_layers": 1, "tie_word_embeddings": False,
}
PROV = Provenance("model:toy", Trust.PUBLIC_CONFIG, None)

# --- the same arithmetic, by hand -------------------------------------------
ATTN = (QL * H + QL + HEADS * (NOPE + ROPE) * QL + (KVL + ROPE) * H + KVL
        + HEADS * (NOPE + VD) * KVL + H * HEADS * VD)
INDEXER = IH * ID * QL + ID * H + 2 * ID + IH * H
DENSE = 3 * H * INTER
EXPERT = 3 * H * MOE_I
MOE = E * H + E + E * EXPERT + SHARED * EXPERT  # router weight + bias, experts, shared
NORMS = 2 * H
MTP_GLUE = 2 * H + 2 * H * H + H  # enorm, hnorm, eh_proj, shared_head.norm
LAYERS = [  # (ffn, has indexer)
    (DENSE, True), (MOE, False), (MOE, True), (MOE, False),
]
MAIN = V * H * 2 + H + sum(ATTN + NORMS + f + (INDEXER if i else 0) for f, i in LAYERS)
MTP = ATTN + NORMS + MOE + INDEXER + MTP_GLUE
KV_LAYER_BF16 = (KVL + ROPE) * 2
KV_INDEX_BF16 = ID * 2


def toy(**overrides):
    return parse_config({**TOY, **overrides}, "toy", PROV)


def hw(capacity_bytes):
    src = "https://example.com/spec"
    return parse_hardware({"name": "toyhw", "fields": {
        "mem_capacity": {"value": capacity_bytes / 2**20, "unit": "MiB", "trust": "USER"},
        "mem_bw": {"value": 1000, "unit": "GB/s", "trust": "OFFICIAL", "source": src},
        "tflops_bf16": {"value": 100, "unit": "TFLOP/s", "trust": "OFFICIAL", "source": src},
        "link_bw": {"value": 100, "unit": "GB/s", "trust": "USER"},
        "devices_per_node": {"value": 8, "unit": "device", "trust": "USER"},
    }})


# --- parameters -------------------------------------------------------------


def test_param_counts_match_hand_arithmetic():
    counts = param_counts(toy())
    assert sum(v for k, v in counts.items() if k != "mtp_layers") == MAIN
    assert counts["mtp_layers"] == MTP
    assert counts["routed_expert"] == 3 * E * EXPERT


def test_active_params():
    routed_active = 3 * K * EXPERT  # three MoE layers, top-k experts each
    non_expert = MAIN - V * H - 3 * E * EXPERT  # embedding lookup excluded
    assert active_params_per_token(toy()) == non_expert + routed_active


def test_bf16_checkpoint_bytes():
    model = toy()
    # all bf16, except the router's score-correction bias, kept in fp32 (4 MoE layers)
    assert sum(t.nbytes for t in model.inventory) == 2 * (MAIN + MTP) + 2 * E * 4


def test_fp8_block_scales_exact():
    q = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [32, 32],
         "modules_to_not_convert": ["lm_head", "model.layers.0.mlp.gate"]}
    model = toy(quantization_config=q)
    qb = next(t for t in model.inventory if t.name == "model.layers.0.self_attn.q_b_proj.weight")
    sc = next(t for t in model.inventory
              if t.name == "model.layers.0.self_attn.q_b_proj.weight_scale_inv")
    assert qb.dtype == "F8_E4M3" and qb.shape == (HEADS * (NOPE + ROPE), QL)
    assert sc.dtype == "F32" and sc.shape == (3, 2)  # ceil(96/32), ceil(64/32)
    kva = next(t for t in model.inventory
               if t.name == "model.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale_inv")
    assert kva.shape == (2, 8)  # ceil(40/32) = 2: a partial block still gets a scale
    lm = next(t for t in model.inventory if t.name == "lm_head.weight")
    assert lm.dtype == "BF16"


# --- KV cache ---------------------------------------------------------------


def test_kv_per_token_bf16():
    r = memory_report(toy(), hw(10**9), Deployment())
    assert r.get("kv_per_token").value == 4 * KV_LAYER_BF16 + 2 * KV_INDEX_BF16


def test_kv_per_token_fp8_with_mtp():
    dep = Deployment(mtp=True, kv_dtype="fp8", index_kv_dtype="fp8")
    r = memory_report(toy(), hw(10**9), dep)
    assert r.get("kv_per_token").value == 5 * (KVL + ROPE) + 3 * ID


def test_kv_not_divided_by_tp():
    a = memory_report(toy(), hw(10**9), Deployment(Parallel(tp=1)))
    b = memory_report(toy(), hw(10**9), Deployment(Parallel(tp=4)))
    assert a.get("kv_per_token").value == b.get("kv_per_token").value


def test_page_rounding():
    r = memory_report(toy(), hw(10**9), Deployment(page_size=64), context=100)
    per_tok = 4 * KV_LAYER_BF16 + 2 * KV_INDEX_BF16
    assert r.get("kv_per_request").value == 128 * per_tok


# --- sharding ---------------------------------------------------------------


def _weights(model, par, mtp=False):
    return memory_report(model, hw(10**9), Deployment(par, mtp=mtp)).get(
        "weights_per_gpu").value


def test_tp_conservation():
    """per_gpu x TP = total + (TP - 1) x replicated, with no vocab padding."""
    model = toy()
    total = sum(t.nbytes for t in model.tensors(mtp=False))
    replicated = sum(t.nbytes for t in model.tensors(mtp=False) if t.shard.value == "replicated")
    for tp in (1, 2, 4):
        assert _weights(model, Parallel(tp=tp)) * tp == total + (tp - 1) * replicated


def test_vocab_padding():
    model = toy(vocab_size=1001)
    r = memory_report(model, hw(10**9), Deployment(Parallel(tp=4)))
    assert any("padded by 3 rows" in n for n in r.notes)


def test_ep_equals_tp_holds_same_expert_bytes():
    model = toy()
    assert _weights(model, Parallel(tp=4, ep=4)) == _weights(model, Parallel(tp=4, ep=1))


def test_pp_stages():
    model = toy()
    assert Parallel(pp=3).stage_layers(model) == [[0, 1], [2], [3]]
    stage_bytes = 2 * (V * H + ATTN + NORMS + DENSE + INDEXER + ATTN + NORMS + MOE) + 2 * E
    assert _weights(model, Parallel(pp=3)) == stage_bytes  # stage 0 is the largest


@pytest.mark.parametrize("par,match", [
    (Parallel(tp=3), "attention heads"),
    (Parallel(tp=4, ep=2), "EP must be"),
    (Parallel(pp=5), "exceeds"),
])
def test_invalid_layouts_rejected(par, match):
    with pytest.raises(SpecError, match=match):
        Deployment(par).validate(toy())


def test_split_off_scale_block_boundary_rejected():
    q = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [32, 32]}
    model = toy(quantization_config=q, num_attention_heads=8, moe_intermediate_size=64)
    with pytest.raises(SpecError, match="scale blocks"):
        Deployment(Parallel(tp=4)).validate(model)  # 64 / 4 = 16 rows: half a block


# --- bounds and assumptions -------------------------------------------------


def test_concurrency_bound_and_levels():
    model = toy()
    cap = 10 * 2**20
    r = memory_report(model, hw(cap), Deployment(), context=1000)
    w = 2 * MAIN + 2 * E * 3  # fp32 router bias in 3 MoE layers
    per_tok = 4 * KV_LAYER_BF16 + 2 * KV_INDEX_BF16
    tokens = (cap - w) // per_tok
    assert r.get("kv_tokens").value == tokens
    assert r.get("max_concurrency").value == tokens // 1000
    for key in ("kv_pool", "kv_tokens", "max_concurrency"):
        q = r.get(key)
        assert q.level == Level.BOUND and q.direction == "upper"
    assert r.get("weights_per_gpu").level == Level.EXACT


def test_assumed_outputs_need_reserve():
    r = memory_report(toy(), hw(10 * 2**20), Deployment(), context=1000)
    assert not r.items["assumed"]
    assert any("--reserve-gb" in m for m in r.missing)

    reserve = Quantity.assumed(2**20, "B", "user:--reserve-gb")
    r = memory_report(toy(), hw(10 * 2**20), Deployment(), context=1000, reserve=reserve)
    q = r.get("max_concurrency_assumed")
    assert q.level == Level.ASSUMED and q.direction == "upper"
    assert "user:--reserve-gb" in {p.ref for p in q.provenance}
    assert q.value <= r.get("max_concurrency").value


def test_min_gpus_lower_bound():
    model = toy()
    w = 2 * MAIN + 2 * E * 3
    r = memory_report(model, hw(w // 3), Deployment())
    q = r.get("min_gpus_weights")
    assert q.value == 4 and q.direction == "lower" and str(q.unit) == "gpu"


def test_weights_not_fitting_is_reported_not_raised():
    model = toy()
    r = memory_report(model, hw(2 * MAIN // 3), Deployment(), context=100)
    assert any("do not fit" in m for m in r.missing)
    with pytest.raises(KeyError):
        r.get("kv_pool")


def test_glm_5_3_kv_per_token():
    """GLM-5.3: 78 layers of 512 + 64 latent, 21 full indexer layers of 128."""
    model = load_model("glm-5.3")
    r = memory_report(model, hw(200 * 2**30), Deployment(Parallel(tp=8)))
    assert r.get("kv_per_token").value == 78 * 576 * 2 + 21 * 128 * 2 == 95_232
    r = memory_report(model, hw(200 * 2**30), Deployment(Parallel(tp=8), mtp=True,
                                                           kv_dtype="fp8",
                                                           index_kv_dtype="fp8"))
    assert r.get("kv_per_token").value == 79 * 576 + 22 * 128 == 48_320


def test_reserve_larger_than_free_memory_is_reported():
    big = Quantity.assumed(10**12, "B", "user:--reserve-gb")
    r = memory_report(toy(), hw(64 * 2**20), Deployment(), context=100, reserve=big)
    assert any("nothing left for KV" in m for m in r.missing)
    assert not r.items["assumed"]


def test_component_sizes_add_up_to_the_checkpoint():
    from rooflinebound.memory import component_bytes
    for m in (toy(), load_model("glm-5.3"), load_model("qwen3-32b")):
        assert sum(component_bytes(m).values()) == sum(t.nbytes for t in m.inventory)
    sizes = component_bytes(toy())
    assert sizes["attn"] == 2 * 4 * ATTN  # bf16, 4 layers
    assert sizes["router"] == 3 * (2 * E * H + 4 * E)  # bf16 weight + fp32 bias, 3 MoE layers


def test_parameter_rows_carry_their_size():
    r = memory_report(load_model("glm-5.3"), hw(200 * 2**30), Deployment(Parallel(tp=8)))
    main, mtp = (next(i for i in r.items["model"] if i.key == k) for k in
                 ("params_main", "params_mtp"))
    assert main.size.value + mtp.size.value == r.get("checkpoint_bytes").value
    experts = next(i for i in r.items["model"] if i.key == "params_routed_expert")
    assert experts.size.value / experts.q.value == pytest.approx(1 + 4 / 128**2)  # fp8 + scales
    assert main.to_dict()["size"]["unit"] == "B"


@pytest.mark.parametrize("name,par,mtp", [("glm-5.3", Parallel(tp=8), False),
                                          ("glm-5.3", Parallel(tp=8, ep=8), True),
                                          ("qwen3-32b", Parallel(tp=4), False),
                                          ("qwen3-32b", Parallel(tp=16), False)])
def test_per_gpu_components_add_up(name, par, mtp):
    r = memory_report(load_model(name), hw(200 * 2**30), Deployment(par, mtp=mtp))
    comps = [i for i in r.items["per_gpu"] if i.key.startswith("gpu_weights_")
             and i.key != "gpu_weights_replicated"]
    assert sum(i.q.value for i in comps) == r.get("weights_per_gpu").value


def test_per_gpu_replicated_share():
    from rooflinebound.memory import stage_component_weights
    m = toy()
    replicated = sum(t.nbytes for t in m.tensors(mtp=False) if t.shard.value == "replicated")
    _, rep, _ = stage_component_weights(m, Deployment(Parallel(tp=2)), 0)
    assert rep == replicated
    comps, _, _ = stage_component_weights(m, Deployment(Parallel(pp=3)), 0)
    assert "lm_head" not in comps and "embed" in comps  # head on the last stage


def test_gqa_kv_projections_count_as_replicated_beyond_kv_heads():
    from rooflinebound.memory import stage_component_weights
    q = load_model("qwen3-32b")
    _, at8, _ = stage_component_weights(q, Deployment(Parallel(tp=8)), 0)
    _, at16, _ = stage_component_weights(q, Deployment(Parallel(tp=16)), 0)
    kv_one_head = 2 * 64 * 128 * 5120 * 2  # k and v, one head, 64 layers, bf16
    assert at16 - at8 == kv_one_head


def test_total_parameters_row_matches_the_checkpoint():
    """The total row (main + MTP) has the safetensors total_size as its size."""
    r = memory_report(load_model("glm-5.3"), hw(200 * 2**30), Deployment(Parallel(tp=8)))
    rows = {i.key: i for i in r.items["model"]}
    total = rows["params_total"]
    assert total.q.value == rows["params_main"].q.value + rows["params_mtp"].q.value \
        == 753_329_940_480
    assert total.size.value == r.get("checkpoint_bytes").value == 755_617_140_416
