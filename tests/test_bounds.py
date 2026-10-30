"""Operator counts and bounds, checked against brute force and hand arithmetic."""

import itertools
import math

import pytest
from test_memory import (
    EXPERT,
    HEADS,
    ID,
    IH,
    INTER,
    KVL,
    MAIN,
    MOE_I,
    NOPE,
    QL,
    ROPE,
    SHARED,
    VD,
    E,
    H,
    K,
    V,
    hw,
    toy,
)

from rooflinebound.bounds import (
    _sum_ops,
    bounds_report,
    critical_batch,
    mtp_expected_tokens,
    pass_bound,
)
from rooflinebound.memory import Deployment
from rooflinebound.ops import GraphBuilder, Pass, causal_pairs, experts_touched, index_pairs
from rooflinebound.parallel import Parallel
from rooflinebound.quantity import Level
from rooflinebound.spec import SpecError

TOPK = 16  # toy index_topk


# --------------------------------------------------------------- exact sums


@pytest.mark.parametrize("start,n,cap", [
    (0, 1, None), (0, 7, None), (5, 3, None), (0, 40, 16), (10, 10, 16), (15, 1, 16),
    (16, 1, 16), (100, 5, 16), (0, 16, 16), (3, 0, 16),
])
def test_causal_pairs_brute_force(start, n, cap):
    want = sum(min(p + 1, cap) if cap else p + 1 for p in range(start, start + n))
    assert causal_pairs(start, n, cap) == want


@pytest.mark.parametrize("start,n,topk", [(0, 40, 16), (15, 1, 16), (16, 1, 16), (0, 16, 16),
                                          (100, 3, 16), (5, 20, 8)])
def test_index_pairs_brute_force(start, n, topk):
    want = sum(p + 1 for p in range(start, start + n) if p + 1 > topk)
    assert index_pairs(start, n, topk) == want


def test_experts_touched_matches_enumeration():
    """E=4, k=2, 2 tokens: average over all routings of distinct experts used."""
    n, k, tokens = 4, 2, 2
    choices = list(itertools.combinations(range(n), k))
    routings = list(itertools.product(choices, repeat=tokens))
    mean = sum(len(set().union(*r)) for r in routings) / len(routings)
    expected, fewest, most = experts_touched(n, k, tokens, held=n, ep=1)
    assert expected == pytest.approx(mean)
    assert fewest == 2 and most == 4
    assert min(len(set().union(*r)) for r in routings) == fewest


def test_experts_touched_single_token_is_exact():
    assert experts_touched(256, 8, 1, 256, 1) == (8.0, 8, 8)


def test_mtp_expected_tokens():
    q = mtp_expected_tokens([0.8, 0.5])
    assert q.value == pytest.approx(1 + 0.8 + 0.4)
    assert q.level == Level.ASSUMED and q.interval == (1.0, 3.0)
    with pytest.raises(SpecError):
        mtp_expected_tokens([1.2])


# ------------------------------------------------------- hand-counted FLOPs


def _decode_flops_by_hand(context):
    pairs = min(context + 1, TOPK)
    ipairs = context + 1 if context + 1 > TOPK else 0
    attn = (2 * QL * H + 2 * HEADS * (NOPE + ROPE) * QL + 2 * (KVL + ROPE) * H
            + 2 * HEADS * KVL * NOPE + 2 * HEADS * ((KVL + ROPE) + KVL) * pairs
            + 2 * HEADS * VD * KVL + 2 * H * HEADS * VD)
    indexer = 2 * IH * ID * QL + 2 * ID * H + 2 * IH * H + 2 * IH * (ID + 1) * ipairs
    dense = 2 * 3 * INTER * H
    moe = 2 * E * H + 2 * 3 * K * H * MOE_I + 2 * 3 * MOE_I * SHARED * H
    # layers: dense+idx, moe, moe+idx, moe
    return 4 * attn + 2 * indexer + dense + 3 * moe + 2 * V * H


@pytest.mark.parametrize("context", [0, 10, 100])
def test_decode_flops_match_hand_count(context):
    ops = GraphBuilder(toy(), Parallel()).build(Pass(1, 1, context, 1))
    assert sum(o.count * o.flops for o in ops) == pytest.approx(_decode_flops_by_hand(context))


def test_decode_hbm_bytes_match_hand_count():
    """Batch 1, bf16: every weight once (experts: k of E), minus the embedding table
    plus one embedding row, minus the indexer k_norm (not counted), plus KV."""
    c = 100
    ops = GraphBuilder(toy(), Parallel()).build(Pass(1, 1, c, 1))
    weights = 2 * MAIN + 2 * E * 3  # bf16, plus the fp32 router bias in 3 MoE layers
    weights -= 2 * V * H  # the embedding table is looked up, not read
    weights -= 3 * E * 2 * EXPERT * (1 - K / E)  # only k of E experts are read
    weights -= 2 * 2 * 2 * ID  # k_norm weight + bias in the 2 indexer layers: omitted
    embed_row = 2 * H
    kv = 4 * (min(c + 1, TOPK) - 1) * (KVL + ROPE) * 2 + 4 * (KVL + ROPE) * 2  # read + write
    kv += 2 * (c * ID * 2 + ID * 2)  # indexer keys: read all cached, write one
    got = sum(o.count * o.hbm_bytes() for o in ops)
    assert got == pytest.approx(weights + embed_row + kv)


def test_tp_splits_flops_except_replicated():
    m = toy()
    one = GraphBuilder(m, Parallel(tp=1)).build(Pass(1, 1, 100, 1))
    two = GraphBuilder(m, Parallel(tp=2)).build(Pass(1, 1, 100, 1))
    replicated = {"attn.q_a_proj", "attn.kv_a_proj", "moe.router", "indexer.wq_b",
                  "indexer.wk", "indexer.weights_proj", "indexer.score"}

    def split(ops, rep):
        f = sum(o.count * o.flops for o in ops if o.name.split(".", 1)[-1] in rep)
        return f, sum(o.count * o.flops for o in ops) - f
    r1, s1 = split(one, replicated)
    r2, s2 = split(two, replicated)
    assert r1 == pytest.approx(r2) and s2 == pytest.approx(s1 / 2)
    assert sum(o.comm_bytes * o.count for o in one) == 0
    assert sum(o.comm_bytes * o.count for o in two) > 0


def test_pp_rejected_for_bounds():
    with pytest.raises(SpecError, match="PP"):
        GraphBuilder(toy(), Parallel(pp=2))


# ------------------------------------------------------------------ bounds


def test_bound_is_max_of_resources_and_lower():
    b = pass_bound(GraphBuilder(toy(), Parallel()), hw(10**9), [(Pass(4, 1, 100, 4), "main")])
    assert b.time.direction == "lower" and b.time.level == Level.BOUND
    assert b.time.value == pytest.approx(max(b.compute.value, b.memory.value, b.comm.value))


def test_bound_takes_the_cheaper_mla_form():
    builder = GraphBuilder(toy(), Parallel())
    p = [(Pass(1, 64, 0, 1), "main")]
    b = pass_bound(builder, hw(10**9), p)
    for mode in ("absorbed", "expanded"):
        c, m, x = _sum_ops(builder.build(p[0][0], mla_mode=mode), hw(10**9))
        assert b.time.value <= max(c, m, x) + 1e-15


def test_tpot_and_throughput_monotone_in_batch():
    builder = GraphBuilder(toy(), Parallel())
    spec = hw(10**9)
    prev_t, prev_thr = 0.0, 0.0
    for b in (1, 2, 4, 8, 16, 64, 256):
        t = pass_bound(builder, spec, [(Pass(b, 1, 100, b), "main")]).time.value
        assert t >= prev_t and b / t >= prev_thr - 1e-9
        prev_t, prev_thr = t, b / t


def test_critical_batch_is_the_crossing():
    builder, spec = GraphBuilder(toy(), Parallel()), hw(10**9)
    b = critical_batch(builder, spec, 100)
    assert b is not None and b > 1

    def gap(n):
        c, m, _ = _sum_ops(builder.build(Pass(n, 1, 100, n)), spec)
        return c - m
    assert gap(b) >= 0 > gap(b - 1)


def test_report_levels_and_missing_inputs():
    m, spec = toy(), hw(64 * 2**20)
    r = bounds_report(m, spec, Deployment(mtp=True), context=100, batch=4, prompt=64,
                      draft=2)
    assert r.get("tpot_b1").level == Level.BOUND and r.get("tpot_b1").direction == "lower"
    assert r.get("tps_seq_b1").direction == "upper"
    assert r.get("ttft").direction == "lower"
    assert any("--mtp-accept" in x for x in r.missing)

    r = bounds_report(m, spec, Deployment(mtp=True), context=100, batch=4, draft=2,
                      accept=[0.9, 0.5])
    tpot = r.get("mtp_tpot")
    assert tpot.level == Level.ASSUMED and tpot.direction == "lower"
    assert "user:--mtp-accept" in {p.ref for p in tpot.provenance}
    step = r.get("mtp_step")
    assert tpot.value == pytest.approx(step.value / 2.35)


def test_max_throughput_uses_kv_capacity():
    m = toy()
    r = bounds_report(m, hw(64 * 2**20), Deployment(), context=100)
    thr = r.get("tps_max")
    assert thr.direction == "upper" and thr.level == Level.BOUND
    assert "batch" in next(i.note for i in r.items["decode"] if i.key == "tps_max")


def test_glm_5_3_decode_batch_1_is_hbm_bound():
    from rooflinebound.spec import load_hardware, load_model
    r = bounds_report(load_model("glm-5.3"), load_hardware("h20-3e"),
                      Deployment(Parallel(tp=8)), context=32768)
    b = r.tables["decode@1"]
    assert b.limiter == "memory"
    # At batch 1 exactly k experts are read: the routing range collapses.
    assert b.time.interval is None
    assert math.isclose(b.time.value, b.memory.value)


def test_strict_uses_fastest_rate_and_never_raises_a_bound():
    m, spec = toy(), hw(64 * 2**20)
    fast = spec.fastest_compute()
    assert fast.peak_flops("bf16").value == max(
        f.quantity.value for k, f in spec.fields.items() if k.startswith("tflops_"))
    kw = dict(context=100, batch=4, prompt=512)
    base = bounds_report(m, spec, Deployment(), **kw)
    strict = bounds_report(m, spec, Deployment(), strict=True, **kw)
    assert strict.inputs["compute"] == "strict"
    for key in ("tpot_b1", "tpot_batch", "ttft"):
        assert strict.get(key).value <= base.get(key).value
        assert strict.get(key).direction == "lower"


def test_default_report_notes_strict_alternative_when_it_matters():
    from rooflinebound.spec import load_hardware, load_model
    m, spec = load_model("glm-5.3"), load_hardware("h20-3e")
    dep = Deployment(Parallel(tp=8))  # bf16 KV: attention and indexer run at the bf16 rate
    r = bounds_report(m, spec, dep, prompt=8192)
    assert any("--strict" in n and "TTFT" in n for n in r.notes)
    dep8 = Deployment(Parallel(tp=8), kv_dtype="fp8", index_kv_dtype="fp8")
    r8 = bounds_report(m, spec, dep8, context=8192)  # decode is HBM-bound: no difference
    assert not any("fastest tensor rate (--strict):" in n for n in r8.notes)
