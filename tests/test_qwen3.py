"""Second model x hardware pair: Qwen3-32B (dense GQA) on H100 SXM.

Checked against the checkpoint headers and against hand arithmetic, like
GLM-5.3, so that extending the package is held to the same standard.
"""

import json
import shlex
from pathlib import Path

import pytest
from test_glm_inventory import fold
from test_memory import hw

from rooflinebound.bounds import bounds_report
from rooflinebound.engines import parse_sglang
from rooflinebound.kvlayout import sglang_mha
from rooflinebound.memory import Deployment, memory_report
from rooflinebound.ops import GraphBuilder, Pass
from rooflinebound.parallel import Parallel
from rooflinebound.quantity import Level, Provenance, Trust
from rooflinebound.spec import SpecError, load_hardware, load_model
from rooflinebound.spec.models import parse_config

FIXTURE = Path(__file__).parent / "fixtures" / "qwen3-32b.safetensors-summary.json"
H, HEADS, KVH, HD, INTER, V, L = 5120, 64, 8, 128, 25600, 151936, 64


@pytest.fixture(scope="module")
def qwen():
    return load_model("qwen3-32b")


def test_inventory_matches_checkpoint(qwen):
    fx = json.loads(FIXTURE.read_text("utf-8"))
    expected = {(r["pattern"], r["dtype"], tuple(r["shape"])): (r["count"], r["layers"])
                for r in fx["tensors"]}
    assert fold((t.name, t.dtype, t.shape) for t in qwen.inventory) == expected
    assert sum(t.nbytes for t in qwen.inventory) == fx["total_size"] == 65_524_246_528


@pytest.mark.parametrize("tp,heads_per_rank", [(1, 8), (2, 4), (8, 1), (16, 1)])
def test_kv_per_token_split_by_kv_head(qwen, tp, heads_per_rank):
    r = memory_report(qwen, hw(200 * 2**30), Deployment(Parallel(tp=tp)))
    # key + value, per KV head held, per layer, bf16
    assert r.get("kv_per_token").value == L * 2 * heads_per_rank * HD * 2


def test_kv_heads_replicate_beyond_their_count(qwen):
    """At TP16 each rank keeps one of the 8 KV heads: k/v weights are 1/8, not 1/16."""
    def kv_weight(tp):
        par = Parallel(tp=tp)
        return sum(par.rank_bytes(t) for t in qwen.inventory if t.name.endswith("k_proj.weight"))
    assert kv_weight(8) == kv_weight(16) == L * HD * H * 2


def test_kv_heads_that_do_not_divide_are_rejected():
    cfg = {"model_type": "qwen3", "hidden_size": 384, "vocab_size": 100,
           "num_hidden_layers": 2, "num_attention_heads": 12, "num_key_value_heads": 4,
           "head_dim": 32, "intermediate_size": 512, "torch_dtype": "bfloat16"}
    m = parse_config(cfg, "toy-gqa", Provenance("model:toy", Trust.PUBLIC_CONFIG))
    with pytest.raises(SpecError, match="KV heads"):
        Deployment(Parallel(tp=3)).validate(m)


def test_decode_flops_match_hand_count(qwen):
    c = 1000
    ops = GraphBuilder(qwen, Parallel()).build(Pass(1, 1, c, 1))
    per_layer = (2 * HEADS * HD * H + 2 * 2 * KVH * HD * H  # q, k, v projections
                 + 4 * HEADS * HD * (c + 1)  # scores and value mixing over c + 1 keys
                 + 2 * H * HEADS * HD  # o_proj
                 + 2 * 3 * INTER * H)  # gate, up, down
    assert sum(o.count * o.flops for o in ops) == pytest.approx(L * per_layer + 2 * V * H)


def test_h100_is_official():
    spec = load_hardware("h100-sxm")
    assert spec.weakest_trust == Trust.OFFICIAL
    assert spec.peak_flops("bf16").value == 989.5e12  # dense: half the "with sparsity" figure
    assert spec["mem_bw"].value == 3.35e12


def test_sglang_mha_launch(qwen):
    h100 = load_hardware("h100-sxm")
    cfg = parse_sglang(shlex.split("sglang serve --model-path /m --tp 4 --kv-cache-dtype "
                                   "fp8_e4m3 --max-running-requests 128"), qwen, h100)
    assert cfg.layout == sglang_mha("fp8_e4m3") and cfg.page_size == 1
    dep = Deployment(Parallel(tp=4), kv_layout=cfg.layout, page_size=1)
    r = memory_report(qwen, h100, dep)
    assert r.get("kv_per_token").value == L * 2 * 2 * HD * 1  # 2 KV heads per rank, fp8


def test_bounds_end_to_end_on_h100(qwen):
    h100 = load_hardware("h100-sxm")
    r = bounds_report(qwen, h100, Deployment(Parallel(tp=4)), context=8192, batch=32,
                      prompt=8192, compare=False)
    t1 = r.get("tpot_b1")
    assert t1.level == Level.BOUND and t1.direction == "lower" and t1.interval is None
    # batch 1 is HBM-bound: at least the GPU's weight share over the bandwidth
    w = memory_report(qwen, h100, Deployment(Parallel(tp=4))).get("weights_per_gpu").value
    assert t1.value / 1e3 >= (w - V * H * 2 / 4) / 3.35e12
    assert r.get("ttft").direction == "lower"
