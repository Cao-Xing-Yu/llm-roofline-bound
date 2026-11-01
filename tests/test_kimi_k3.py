"""Third pair: Kimi-K3 (hybrid KDA + MLA, latent MoE, MXFP4, vision) on B300.

Held to the same standard as the others: the inventory against the
checkpoint headers, memory and state against SGLang's layout, FLOPs against
hand counts.
"""

import json
import re
import shlex
from collections import defaultdict
from pathlib import Path

import pytest

from rooflinebound.bounds import bounds_report
from rooflinebound.engines import parse_sglang
from rooflinebound.memory import Deployment, memory_report
from rooflinebound.ops import GraphBuilder, Pass
from rooflinebound.parallel import Parallel
from rooflinebound.quantity import Trust
from rooflinebound.spec import load_hardware, load_model

FIXTURE = Path(__file__).parent / "fixtures" / "kimi-k3.safetensors-summary.json"
H, HEADS, D, PROJ, KDA_LAYERS, MLA_LAYERS = 7168, 96, 128, 12288, 69, 24
LAUNCH = ("sglang serve --model-path /models/kimi-k3 --tp 8 --kv-cache-dtype fp8_e4m3 "
          "--mem-fraction-static 0.88 --chunked-prefill-size 16384 --max-running-requests 64")


@pytest.fixture(scope="module")
def kimi():
    return load_model("kimi-k3")


@pytest.fixture(scope="module")
def b300():
    return load_hardware("b300")


def fold(names):
    rx = re.compile(r"(.*?\b(?:layers|blocks))\.(\d+)\.(.*)")
    g = defaultdict(lambda: [0, set()])
    for name, dtype, shape in names:
        m = rx.match(name)
        key = (m.group(1) + ".{L}." + re.sub(r"experts\.\d+\.", "experts.{E}.", m.group(3))
               if m else name)
        e = g[(key, dtype, tuple(shape))]
        e[0] += 1
        if m:
            e[1].add(int(m.group(2)))
    return {k: (n, sorted(layers)) for k, (n, layers) in g.items()}


def test_inventory_matches_checkpoint(kimi):
    fx = json.loads(FIXTURE.read_text("utf-8"))
    expected = {(r["pattern"], r["dtype"], tuple(r["shape"])): (r["count"], r["layers"])
                for r in fx["tensors"]}
    assert fold((t.name, t.dtype, t.shape) for t in kimi.inventory) == expected
    assert sum(t.nbytes for t in kimi.inventory) == fx["total_size"] == 1_560_860_324_864
    assert len(kimi.inventory) == fx["n_tensors"] == 497_220


def test_mxfp4_packs_two_params_per_byte(kimi):
    w = next(t for t in kimi.inventory if t.name.endswith("experts.0.w1.weight_packed"))
    s = next(t for t in kimi.inventory if t.name.endswith("experts.0.w1.weight_scale"))
    assert w.shape == (3072, 3584 // 2) and w.params == 3072 * 3584 and w.nbytes == w.params // 2
    assert s.shape == (3072, 3584 // 32) and s.params == 0


def test_kv_and_state_follow_sglang(kimi, b300):
    cfg = parse_sglang(shlex.split(LAUNCH), kimi, b300)
    dep = Deployment(Parallel(tp=8), kv_layout=cfg.layout, page_size=cfg.page_size)
    r = memory_report(kimi, b300, dep, context=32768)
    assert r.get("kv_per_token").value == MLA_LAYERS * 576 * 1  # fp8 latent + rope, MLA only
    # per rank: fp32 recurrent state (96/8 heads x 128 x 128) + bf16 conv window (3 x 3 x 12288/8)
    per_layer = (HEADS // 8) * D * D * 4 + 3 * 3 * (PROJ // 8) * 2
    assert r.get("state_per_request").value == KDA_LAYERS * per_layer
    assert r.get("kv_per_request").value == 32768 * MLA_LAYERS * 576 + KDA_LAYERS * per_layer


def test_concurrency_counts_state(kimi, b300):
    dep = Deployment(Parallel(tp=8), kv_dtype="fp8")
    r = memory_report(kimi, b300, dep, context=1)
    pool = r.get("kv_pool").value
    state = r.get("state_per_request").value
    per_req = r.get("kv_per_request").value
    assert per_req == MLA_LAYERS * 576 + state
    assert r.get("max_concurrency").value == pool // per_req


def test_language_only_drops_vision(kimi, b300):
    full = memory_report(kimi, b300, Deployment(Parallel(tp=8))).get("weights_per_gpu").value
    text = memory_report(kimi, b300, Deployment(Parallel(tp=8), language_only=True)
                         ).get("weights_per_gpu").value
    vision = sum(t.nbytes for t in kimi.inventory if t.component == "vision")
    assert full - text == vision  # replicated: every rank holds the whole tower
    cfg = parse_sglang(shlex.split(LAUNCH + " --language-only"), kimi, b300)
    assert cfg.language_only


def test_kda_decode_flops_and_state_traffic(kimi):
    b = GraphBuilder(kimi, Parallel(tp=8))
    ops = [o for o in b.build(Pass(4, 1, 1000, 4)) if ".kda." in o.name]
    rec = next(o for o in ops if o.name.endswith("kda.recurrence"))
    hl = HEADS // 8
    assert rec.flops == pytest.approx(4 * hl * D * D * 4 + 2 * 4 * 3 * (PROJ // 8) * 4)
    state = b.layout.state_bytes(kimi.layers[0].attn, 8)
    assert rec.kv_read == rec.kv_write == 4 * state  # each sequence: read once, write once
    qkv = next(o for o in ops if o.name.endswith("kda.qkv_proj"))
    assert qkv.flops == pytest.approx(2 * 4 * 3 * PROJ * H / 8)


def test_first_prefill_chunk_reads_no_state(kimi):
    rec = next(o for o in GraphBuilder(kimi, Parallel(tp=8)).build(Pass(1, 512, 0, 1))
               if o.name.endswith("kda.recurrence"))
    assert rec.kv_read == 0 and rec.kv_write > 0


def test_b300_is_official_and_has_fp4(b300):
    assert b300.weakest_trust == Trust.OFFICIAL
    assert b300.peak_flops("fp4").value == 13.5e15
    assert b300.peak_flops("bf16").value == 2.25e15
    assert b300["mem_capacity"].value == 288 * 2**30


def test_routed_experts_compute_at_fp4(kimi):
    ops = GraphBuilder(kimi, Parallel(tp=8)).build(Pass(1, 1, 100, 1))
    assert next(o for o in ops if o.name.endswith("moe.routed_experts")).dtype == "fp4"


def test_bounds_end_to_end(kimi, b300):
    r = bounds_report(kimi, b300, Deployment(Parallel(tp=8), kv_dtype="fp8"), context=8192,
                      batch=16, prompt=8192, compare=False)
    t1 = r.get("tpot_b1")
    assert t1.direction == "lower" and t1.interval is None  # one token: exactly k experts
    assert r.get("ttft").direction == "lower"


def test_capacity_and_chart_with_fp4_operators(kimi, b300, tmp_path):
    """fp4 operators reach every consumer of peak rates (capacity, roofline chart)."""
    from rooflinebound.capacity import Targets
    from rooflinebound.roofline import roofline_html
    r = bounds_report(kimi, b300, Deployment(Parallel(tp=8), kv_dtype="fp8"), context=4096,
                      prompt=4096, targets=Targets(output=512, target_qps=5), compare=False)
    assert r.get("qps").direction == "upper"
    assert "<svg" in roofline_html(kimi.name, b300, r.tables, r.notes)


def test_decode_traffic_closed_form_includes_state(kimi):
    """The per-request closed form equals building every decode step, state included."""
    from rooflinebound.capacity import decode_kv_bytes
    b = GraphBuilder(kimi, Parallel(tp=8))
    prompt, output = 100, 6
    steps = sum(o.count * (o.kv_read + o.kv_write)
                for i in range(output) for o in b.build(Pass(1, 1, prompt + i, 1)))
    assert decode_kv_bytes(b, prompt, output) == steps
