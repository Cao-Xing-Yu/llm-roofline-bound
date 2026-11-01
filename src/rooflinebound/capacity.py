"""Sustained capacity: requests per second a replica can serve, and what follows.

Every request needs a minimum amount of each resource, whatever the schedule:

* **compute**: the exact FLOPs of its prefill and of its decode tokens;
* **HBM**: the KV it must read and write, plus its share of weight reads. A
  decode step reads the weights once for every sequence in the batch, so a
  request's share is at least ``output x weights(B) / B`` at the largest batch
  ``B`` the replica can hold. Prefill weight reads can ride along with decode
  steps in a mixed batch, so none are charged;
* **interconnect**: all-reduce bytes, proportional to its tokens.

A replica's GPUs run in parallel, so a request occupies each resource for at
least its work divided by the per-GPU rate, and

    QPS <= min over resources of rate / work-per-request.

That bound holds for any batching, chunking or scheduling policy.
"""

from __future__ import annotations

from dataclasses import dataclass

from .arch.model import ModelArch
from .bounds import _sum_ops
from .ops.graph import GraphBuilder, Pass
from .quantity import Level, Quantity, qmax
from .spec.hardware import HardwareSpec

__all__ = ["RequestWork", "request_work", "sla_max_batch", "sum_min", "sum_from"]

_SLA_SEARCH = 1 << 20  # largest batch the SLA search considers


def sum_min(lo: int, hi: int, cap: int | None) -> int:
    """Σ_{c=lo}^{hi-1} min(c, cap) (or Σ c when cap is None), exactly."""
    if hi <= lo:
        return 0
    if cap is None or hi - 1 <= cap:
        return (lo + hi - 1) * (hi - lo) // 2
    if lo >= cap:
        return cap * (hi - lo)
    return (lo + cap) * (cap - lo + 1) // 2 + cap * (hi - 1 - cap)


def sum_from(lo: int, hi: int, threshold: int) -> int:
    """Σ_{c=lo}^{hi-1} c over the c >= threshold, exactly."""
    a = max(lo, threshold)
    return 0 if a > hi - 1 else (a + hi - 1) * (hi - a) // 2


def decode_kv_bytes(builder: GraphBuilder, prompt: int, output: int) -> int:
    """HBM bytes one request's decode moves through the KV cache, per GPU.

    The step that produces token ``i`` runs at context ``c = prompt + i``: each
    layer reads the cached keys its query attends (all ``c``, or ``topk - 1``
    besides the new one under sparse attention), the indexer reads all ``c``
    cached index keys once it has more than ``topk`` to choose from, and each
    layer writes the new token.
    """
    lay = builder.layout
    total = 0
    lo, hi = prompt, prompt + output
    for layer in builder.model.layers:
        main = lay.main_bytes(layer.attn, builder.par.tp)
        cap = layer.attn_topk
        total += main * (sum_min(lo, hi, None if cap is None else cap - 1) + output)
        if layer.indexer is not None:
            ix = lay.index_key_bytes(layer.indexer)
            total += ix * (sum_from(lo, hi, layer.indexer.topk) + output)
    return total


@dataclass
class RequestWork:
    compute: Quantity  # s per request at peak, per GPU
    hbm: Quantity
    link: Quantity

    @property
    def time(self) -> Quantity:
        """Lower bound on the replica time one request occupies."""
        return qmax(self.compute, self.hbm, self.link).as_bound(
            "lower", derivation="max over resources of work per request / rate")

    @property
    def limiter(self) -> str:
        parts = {"compute": self.compute.value, "HBM": self.hbm.value,
                 "interconnect": self.link.value}
        return max(parts, key=parts.get)


def _prefill_passes(prompt: int, chunk: int | None) -> list[Pass]:
    c = chunk or prompt
    return [Pass(1, min(c, prompt - s), s, 1 if s + c >= prompt else 0)
            for s in range(0, prompt, c)]


def request_work(builder: GraphBuilder, hw: HardwareSpec, prompt: int, output: int,
                 chunk: int | None, batch: Quantity) -> RequestWork:
    """Minimum work of one request (``prompt`` in, ``output`` out) per GPU.

    ``batch`` is an upper bound on the sequences decoding together; the
    request's share of decode weight reads is evaluated there.
    """
    model: ModelArch = builder.model
    prov = model.provenance
    b = max(int(batch.value), 1)

    # Prefill: for each resource the cheaper MLA form (each form is a valid way).
    pre = {m: [o for p in _prefill_passes(prompt, chunk)
               for o in builder.build(p, mla_mode=m)] for m in ("absorbed", "expanded")}
    peaks = {d: hw.peak_flops(d).value for d in ("bf16", "fp8")}
    pre_flops_s = min(sum(o.count * o.flops / peaks[o.dtype] for o in ops)
                      for ops in pre.values())
    pre_kv = min(sum(o.count * (o.kv_read + o.kv_write) for o in ops) for ops in pre.values())
    pre_comm = min(sum(o.count * o.comm_bytes for o in ops) for ops in pre.values())

    # Decode: FLOPs of all its tokens (attention over exactly the attended pairs).
    dec = builder.build(Pass(1, output, prompt, output), mla_mode="absorbed")
    dec_flops_s = sum(o.count * o.flops / peaks[o.dtype] for o in dec)
    dec_comm = sum(o.count * o.comm_bytes for o in dec)
    kv = pre_kv + decode_kv_bytes(builder, prompt, output)

    # Weight reads of one decode step at batch b, shared by its b sequences.
    step = builder.build(Pass(b, 1, prompt, b))
    w = [sum(o.count * o.weight_bytes * (o.read_fraction[i] if o.read_fraction else 1.0)
             for o in step) for i in range(3)]
    share = [x * output / b for x in w]

    compute_s = Quantity.exact(pre_flops_s + dec_flops_s, "s", prov, *_rate_prov(hw))
    hbm_bytes = Quantity.exact(kv, "B", prov) + Quantity.expected(
        share[0], "B", min(share), max(share), prov,
        derivation=f"decode weight reads shared by {b} sequences")
    # Minimum work: a lower bound, which then also rests on the batch bound.
    hbm_s = (hbm_bytes / hw["mem_bw"]).as_bound("lower").relying_on(batch)
    link_s = Quantity.exact(pre_comm + dec_comm, "B", prov) / (hw["link_bw"] * 0.5)
    return RequestWork(compute_s, hbm_s, link_s)


def _rate_prov(hw: HardwareSpec):
    return tuple(p for d in ("bf16", "fp8") for p in hw.peak_flops(d).provenance)


def sla_max_batch(builder: GraphBuilder, hw: HardwareSpec, context: int, tpot_s: float,
                  limit: int) -> Quantity | None:
    """Largest decode batch whose TPOT bound is within ``tpot_s``, up to ``limit``.

    The TPOT bound rises with batch, so every larger batch misses the target:
    an upper BOUND on the batch an SLA allows. Under uniform routing; the range
    covers the most and least concentrated routings. None if batch 1 misses.
    """
    def largest(which: int) -> int:
        def ok(n: int) -> bool:
            c, m, x = _sum_ops(builder.build(Pass(n, 1, context, n)), hw, which)
            return max(c, m, x) <= tpot_s
        if not ok(1):
            return 0
        lo, hi = 1, limit
        if ok(hi):
            return hi
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        return lo

    exp_, fewest, most = largest(0), largest(1), largest(2)
    if fewest == 0:
        return None
    lo, hi = min(most, exp_), max(fewest, exp_)
    return Quantity(float(exp_), "seq", Level.BOUND, "upper",
                    interval=(float(lo), float(hi)) if lo != hi else None,
                    provenance=(builder.model.provenance,))


def replicas_needed(target_qps: Quantity, qps: Quantity) -> Quantity:
    return (target_qps / qps).ceil()


def gpus_any_layout(model: ModelArch, hw: HardwareSpec, prompt: int, output: int,
                    target_qps: Quantity) -> Quantity:
    """GPUs needed for the target by compute alone, whatever the layout or precision.

    Total FLOPs of a request (no TP replication) at the device's fastest
    tensor rate, times the target rate.
    """
    from .parallel import Parallel
    one = GraphBuilder(model, Parallel())
    flops = sum(o.count * o.flops for p in _prefill_passes(prompt, None)
                for o in one.build(p, mla_mode="absorbed"))
    flops_exp = sum(o.count * o.flops for p in _prefill_passes(prompt, None)
                    for o in one.build(p, mla_mode="expanded"))
    flops = min(flops, flops_exp) + sum(
        o.count * o.flops for o in one.build(Pass(1, output, prompt, output)))
    fast = hw.fastest_compute().peak_flops("bf16")
    per_gpu = (fast / Quantity.exact(flops, "FLOP/req", model.provenance)).as_bound("upper")
    return (target_qps / per_gpu).ceil() * Quantity.exact(1, "gpu")


@dataclass(frozen=True)
class Targets:
    """What the user asks of the deployment; all optional."""

    output: int | None = None  # output tokens per request
    sla_tpot_ms: float | None = None
    sla_ttft_ms: float | None = None
    target_qps: float | None = None
    price_gpu_hour: float | None = None  # USD per GPU-hour


def capacity_section(rep, builder: GraphBuilder, hw: HardwareSpec, dep, *, prompt: int | None,
                     context: int | None, chunk: int | None, draft: int, reserve,
                     t: Targets) -> None:
    """Fill ``rep.items["capacity"]`` and its missing-input notes."""
    from .memory import Item, memory_report

    items = rep.items.setdefault("capacity", [])
    model = builder.model
    asked = any(v is not None for v in (t.output, t.sla_tpot_ms, t.target_qps,
                                        t.price_gpu_hour))
    if not asked:
        return
    if draft:
        rep.missing.append("sustained capacity, SLA and fleet figures with MTP are not "
                           "modelled yet; run without --mtp-draft")
        return
    if prompt is None or t.output is None:
        rep.missing.append("--prompt and --output (tokens in and out per request): needed for "
                           "sustained capacity, SLA batch, fleet size and cost")
        return

    # Largest batch the replica can hold while these requests decode: KV memory at
    # the shortest decode context (the prompt), the engine's cap, and the SLA.
    mem = memory_report(model, hw, dep, context=prompt + 1, reserve=reserve)
    key = "max_concurrency_assumed" if reserve is not None else "max_concurrency"
    try:
        batch = mem.get(key)
    except KeyError:
        rep.missing.append("sustained capacity: the weights leave no KV room")
        return
    if batch.value < 1:
        rep.missing.append("sustained capacity: one request does not fit in KV memory")
        return
    why = "KV memory at the prompt length"
    if dep.max_running is not None and dep.max_running < batch.value:
        batch, why = Quantity.exact(dep.max_running, "req"), "the engine's max running requests"

    if t.sla_tpot_ms is not None:
        ctx = context if context is not None else prompt
        # Searched without the memory or engine caps: this is what the TPOT target
        # alone allows. At the shortest decode context, the most permissive one.
        sla_b = sla_max_batch(builder, hw, ctx, t.sla_tpot_ms / 1e3, _SLA_SEARCH)
        label = f"batch meeting TPOT ≤ {t.sla_tpot_ms:g} ms"
        if sla_b is None:
            items.append(Item("sla_batch", label, Quantity.exact(0, "seq").as_bound("upper"),
                              f"even batch 1 misses it at context {ctx:,}: infeasible"))
            return
        if sla_b.value >= _SLA_SEARCH:
            rep.notes.append(f"SLA: TPOT ≤ {t.sla_tpot_ms:g} ms does not limit the batch "
                             f"(met up to {_SLA_SEARCH:,} sequences at context {ctx:,})")
        else:
            items.append(Item("sla_batch", label, sla_b,
                              f"at context {ctx:,}; every larger batch misses it"))
        if sla_b.value < batch.value:
            batch, why = sla_b, f"the TPOT target of {t.sla_tpot_ms:g} ms"

    if t.sla_ttft_ms is not None:
        try:
            ttft = rep.get("ttft")
        except KeyError:
            ttft = None
        if ttft is not None and ttft.value > t.sla_ttft_ms:
            rep.notes.append(f"SLA: TTFT ≤ {t.sla_ttft_ms:g} ms cannot be met for a "
                             f"{prompt:,}-token prompt: TTFT {ttft.format()}")
        elif ttft is not None:
            rep.notes.append(f"SLA: TTFT ≤ {t.sla_ttft_ms:g} ms is not ruled out by the bound "
                             f"({ttft.format()}); queueing can still break it")

    work = request_work(builder, hw, prompt, t.output, chunk, batch)
    qps = Quantity.exact(1, "req") / work.time
    out = qps * Quantity.exact(t.output, "token/req")
    items += [
        Item("qps", f"requests/s per replica ({prompt:,} in, {t.output:,} out)", qps,
             f"{work.limiter}-bound; decode weight reads shared by ≤ {int(batch.value)} "
             f"sequences ({why})"),
        Item("tps_sustained", "output tokens/s per replica, sustained", out,
             "prefill and decode together, any schedule"),
    ]

    gpus_per_replica = dep.parallel.gpus
    if t.target_qps is not None:
        target = Quantity.exact(t.target_qps, "req/s", _user("--target-qps"))
        reps = (target / qps).ceil()
        items += [
            Item("replicas", f"replicas for {t.target_qps:g} requests/s", reps * Quantity.exact(
                1, "replica"), f"{dep.parallel} each"),
            Item("gpus_layout", f"GPUs for {t.target_qps:g} requests/s, this layout",
                 reps * Quantity.exact(gpus_per_replica, "gpu"), None),
            Item("gpus_any", f"GPUs for {t.target_qps:g} requests/s, any layout",
                 gpus_any_layout(model, hw, prompt, t.output, target),
                 "compute alone at the fastest tensor rate"),
        ]
    if t.price_gpu_hour is not None:
        price = Quantity.assumed(t.price_gpu_hour, "USD/gpu*h", "user:--price-gpu-hour")
        per_h = price * Quantity.exact(gpus_per_replica, "gpu") / Quantity.exact(3600, "s/h")
        cost = (per_h / out).to("USD/Mtoken", 1e6)
        items.append(Item("cost", "cost per million output tokens", cost,
                          f"{gpus_per_replica} GPUs per replica at the sustained bound"))


def _user(ref: str):
    from .quantity import Provenance, Trust
    return Provenance(f"user:{ref}", Trust.USER)
