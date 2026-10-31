"""Sustained capacity, SLA, fleet size and cost."""

import pytest
from test_memory import hw, toy

from rooflinebound.bounds import _sum_ops, bounds_report, pass_bound
from rooflinebound.capacity import (
    Targets,
    decode_kv_bytes,
    request_work,
    sla_max_batch,
    sum_from,
    sum_min,
)
from rooflinebound.memory import Deployment
from rooflinebound.ops import GraphBuilder, Pass
from rooflinebound.parallel import Parallel
from rooflinebound.quantity import Level, Quantity


@pytest.mark.parametrize("lo,hi,cap", [(0, 10, None), (5, 30, 16), (20, 25, 16), (0, 16, 16),
                                       (3, 3, 4), (10, 40, 15)])
def test_sum_min_brute_force(lo, hi, cap):
    assert sum_min(lo, hi, cap) == sum(min(c, cap) if cap is not None else c
                                       for c in range(lo, hi))


@pytest.mark.parametrize("lo,hi,t", [(0, 40, 16), (16, 20, 16), (0, 10, 16), (30, 35, 16)])
def test_sum_from_brute_force(lo, hi, t):
    assert sum_from(lo, hi, t) == sum(c for c in range(lo, hi) if c >= t)


def test_decode_kv_equals_sum_of_steps():
    """The closed form matches building every decode step one by one."""
    b = GraphBuilder(toy(), Parallel())
    prompt, output = 10, 25  # crosses the toy top-k of 16
    steps = sum(o.count * (o.kv_read + o.kv_write)
                for i in range(output) for o in b.build(Pass(1, 1, prompt + i, 1)))
    assert decode_kv_bytes(b, prompt, output) == steps


def test_request_compute_equals_steps():
    b, spec = GraphBuilder(toy(), Parallel()), hw(10**9)
    prompt, output = 10, 25
    work = request_work(b, spec, prompt, output, None, Quantity.exact(4, "req"))
    peak = spec.peak_flops("bf16").value
    dec = sum(o.count * o.flops for i in range(output)
              for o in b.build(Pass(1, 1, prompt + i, 1)))
    pre = min(sum(o.count * o.flops for o in b.build(Pass(1, prompt, 0, 1), mla_mode=m))
              for m in ("absorbed", "expanded"))
    assert work.compute.value == pytest.approx((pre + dec) / peak)


def test_sustained_never_beats_decode_alone():
    """Output tokens/s with prefill included cannot exceed decode alone at the same batch."""
    m, spec = toy(), hw(64 * 2**20)
    dep = Deployment(max_running=8)
    r = bounds_report(m, spec, dep, prompt=64, targets=Targets(output=256), compare=False)
    sustained = r.get("tps_sustained")
    assert sustained.direction == "upper"
    b = pass_bound(GraphBuilder(m, Parallel()), spec, [(Pass(8, 1, 64, 8), "main")])
    assert sustained.value <= 8 / b.time.value * (1 + 1e-9)


def test_sla_batch_is_the_crossing():
    b, spec = GraphBuilder(toy(), Parallel()), hw(10**9)
    t1 = pass_bound(b, spec, [(Pass(1, 1, 100, 1), "main")]).time.value
    target = t1 * 3
    q = sla_max_batch(b, spec, 100, target, 1 << 12)
    n = int(q.value)

    def tpot(k):
        return max(_sum_ops(b.build(Pass(k, 1, 100, k)), spec))
    assert tpot(n) <= target < tpot(n + 1)
    assert q.direction == "upper"
    assert sla_max_batch(b, spec, 100, t1 / 2, 1 << 12) is None


def test_fleet_and_cost():
    m, spec = toy(), hw(64 * 2**20)
    t = Targets(output=128, target_qps=50.0, price_gpu_hour=3.0)
    r = bounds_report(m, spec, Deployment(Parallel(tp=2)), prompt=64, targets=t,
                      compare=False)
    qps = r.get("qps").value
    assert r.get("replicas").value == -(-50 // qps) if qps else False
    assert r.get("gpus_layout").value == 2 * r.get("replicas").value
    assert r.get("gpus_any").value <= r.get("gpus_layout").value
    cost = r.get("cost")
    assert cost.level == Level.ASSUMED and cost.direction == "lower"
    out = r.get("tps_sustained").value
    assert cost.value == pytest.approx(3.0 * 2 / 3600 / out * 1e6)
    assert "user:--price-gpu-hour" in {p.ref for p in cost.provenance}


def test_capacity_needs_inputs_and_refuses_mtp():
    m, spec = toy(), hw(64 * 2**20)
    r = bounds_report(m, spec, Deployment(), context=100, targets=Targets(output=10),
                      compare=False)
    assert any("--prompt and --output" in x for x in r.missing)
    r = bounds_report(m, spec, Deployment(mtp=True), prompt=64, draft=1,
                      targets=Targets(output=10), compare=False)
    assert any("MTP are not modelled" in x for x in r.missing)
    assert not r.items["capacity"]


def test_ttft_sla_verdict():
    m, spec = toy(), hw(64 * 2**20)
    r = bounds_report(m, spec, Deployment(), prompt=64,
                      targets=Targets(output=8, sla_ttft_ms=1e-9), compare=False)
    assert any("cannot be met" in n for n in r.notes)
