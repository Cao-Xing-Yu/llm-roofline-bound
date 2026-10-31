"""Latency and throughput lower bounds from the operator graph.

A pass cannot finish faster than any one resource allows, even if every
operator overlapped perfectly with every other:

    t >= max( Σ FLOPs / peak,  Σ compulsory HBM bytes / bandwidth,
              Σ interconnect bytes / per-direction link bandwidth )

That is the bound reported, at datasheet peaks. It assumes nothing about
scheduling. The per-operator sum ``Σ max(t_compute, t_memory)`` is tighter but
holds only if operators never overlap, so it is shown for reference only.

Throughput upper bounds follow: every FLOP count scales with the number of
tokens in the pass while weight reads do not, so ``batch / t(batch)`` never
decreases with batch, and the largest batch the KV memory admits gives the
largest throughput.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .arch.model import ModelArch
from .memory import Deployment, Item, memory_report
from .ops.graph import GraphBuilder, MlaMode, Op, Pass
from .quantity import Level, Provenance, Quantity, Trust, qmax
from .spec.errors import SpecError
from .spec.hardware import HardwareSpec

__all__ = ["PassBound", "BoundsReport", "bounds_report", "pass_bound", "mtp_expected_tokens"]

_MODES: tuple[MlaMode, ...] = ("absorbed", "expanded")
_MAX_BATCH_SEARCH = 1 << 20


@dataclass
class PassBound:
    ops: list[Op]
    compute: Quantity  # s, Σ FLOPs / peak
    memory: Quantity  # s, Σ bytes / bandwidth (EXPECTED when experts are involved)
    comm: Quantity  # s
    time: Quantity  # s, lower BOUND: the max of the three
    mla_mode: str

    @property
    def limiter(self) -> str:
        parts = {"compute": self.compute.value, "memory": self.memory.value,
                 "interconnect": self.comm.value}
        return max(parts, key=parts.get)


def _sum_ops(ops: list[Op], hw: HardwareSpec, which: int = 0) -> tuple[float, float, float]:
    """(compute s, memory s, comm s) as floats for one routing case."""
    peaks = {d: hw.peak_flops(d).value for d in {o.dtype for o in ops}}
    c = sum(o.count * o.flops / peaks[o.dtype] for o in ops)
    m = sum(o.count * o.hbm_bytes(which) for o in ops) / hw["mem_bw"].value
    x = sum(o.count * o.comm_bytes for o in ops) / (hw["link_bw"].value / 2)
    return c, m, x


def _quantities(ops: list[Op], hw: HardwareSpec, model: ModelArch
                ) -> tuple[Quantity, Quantity, Quantity]:
    prov = model.provenance
    compute = Quantity.exact(0, "s")
    for d in sorted({o.dtype for o in ops}):
        flops = sum(o.count * o.flops for o in ops if o.dtype == d)
        compute = compute + Quantity.exact(flops, "FLOP", prov) / hw.peak_flops(d)
    fixed = sum(o.count * o.hbm_bytes() for o in ops if o.read_fraction is None)
    exp_, lo, hi = (sum(o.count * o.hbm_bytes(i) for o in ops if o.read_fraction is not None)
                    for i in (0, 1, 2))
    mem_bytes = Quantity.exact(fixed, "B", prov)
    if hi > 0:
        mem_bytes = mem_bytes + Quantity.expected(exp_, "B", lo, hi, prov,
                                                  derivation="experts read, uniform routing")
    memory = mem_bytes / hw["mem_bw"]
    comm_b = sum(o.count * o.comm_bytes for o in ops)
    comm = Quantity.exact(comm_b, "B", prov) / (hw["link_bw"] * 0.5)
    return compute, memory, comm


def pass_bound(builder: GraphBuilder, hw: HardwareSpec,
               passes: list[tuple[Pass, str]]) -> PassBound:
    """Lower bound on a sequence of passes ((pass, "main"|"mtp"), ...).

    The MLA formulation is a choice the implementation makes; the bound takes
    whichever of the absorbed and expanded forms gives the smaller time.
    """
    best: PassBound | None = None
    for mode in _MODES:
        ops = [o for p, layers in passes for o in builder.build(p, mla_mode=mode, layers=layers)]
        c, m, x = _quantities(ops, hw, builder.model)
        t = qmax(c, m, x).as_bound("lower", derivation="max(compute, HBM, interconnect) at "
                                   "peak rates")
        if best is None or t.value < best.time.value:
            best = PassBound(ops, c, m, x, t, mode)
    assert best is not None
    return best


def mtp_expected_tokens(accept: list[float]) -> Quantity:
    """Tokens emitted per verify step: 1 + Σ_i Π_{j<=i} p_j.

    ``accept[i]`` is the probability draft token i+1 is accepted given the
    ones before it were. User-supplied, so the result is ASSUMED, with the
    range [1, 1 + k] it can take whatever the rates are.
    """
    if not accept or any(not 0 <= p <= 1 for p in accept):
        raise SpecError("--mtp-accept needs one rate in [0, 1] per draft token")
    e, run = 1.0, 1.0
    for p in accept:
        run *= p
        e += run
    return Quantity(e, "token", Level.ASSUMED, interval=(1.0, 1.0 + len(accept)),
                    provenance=(Provenance("user:--mtp-accept", Trust.USER),))


# ------------------------------------------------------------------- report


@dataclass
class BoundsReport:
    model: str
    hardware: str
    inputs: dict
    items: dict[str, list[Item]]
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    tables: dict[str, PassBound] = field(default_factory=dict)

    def get(self, key: str) -> Quantity:
        for items in self.items.values():
            for it in items:
                if it.key == key:
                    return it.q
        raise KeyError(key)

    def to_dict(self) -> dict:
        return {
            "model": self.model, "hardware": self.hardware, "inputs": self.inputs,
            "sections": {s: {it.key: {**it.q.to_dict(), "label": it.label, "note": it.note}
                             for it in items} for s, items in self.items.items()},
            "missing": self.missing, "notes": self.notes,
        }


def _decode_passes(batch: int, context: int, draft: int) -> list[tuple[Pass, str]]:
    """Verify pass with 1 + draft tokens per sequence, then ``draft`` MTP steps."""
    out = [(Pass(batch, 1 + draft, context, batch * (1 + draft)), "main")]
    out += [(Pass(batch, 1, context + i, batch), "mtp") for i in range(draft)]
    return out


def _carry(q: Quantity, dep: Quantity) -> Quantity:
    """``q`` taking on ``dep``'s level, direction and provenance (value unchanged)."""
    return q * (dep / Quantity.exact(dep.value, dep.unit))


def _ms(q: Quantity) -> str:
    v = q.value * 1e3
    return f"{v:,.0f}" if v >= 100 else f"{v:.3g}"


def _breakdown(b: PassBound) -> str:
    ms = _ms
    return (f"compute {ms(b.compute)} / HBM {ms(b.memory)} / interconnect {ms(b.comm)} ms; "
            f"{b.limiter}-bound; MLA {b.mla_mode}")


def critical_batch(builder: GraphBuilder, hw: HardwareSpec, context: int,
                   which: int = 0) -> int | None:
    """Smallest decode batch whose compute time reaches its HBM time.

    Compute is proportional to batch; HBM time per sequence falls as weights are
    shared, so once compute catches up it stays ahead: binary search is exact.
    """
    def ahead(b: int) -> bool:
        ops = builder.build(Pass(b, 1, context, b))
        c, m, _ = _sum_ops(ops, hw, which)
        return c >= m

    if not ahead(_MAX_BATCH_SEARCH):
        return None
    lo, hi = 1, _MAX_BATCH_SEARCH
    while lo < hi:
        mid = (lo + hi) // 2
        if ahead(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def bounds_report(model: ModelArch, hw: HardwareSpec, dep: Deployment, *,
                  context: int | None = None, batch: int | None = None,
                  prompt: int | None = None, chunk: int | None = None, draft: int = 0,
                  accept: list[float] | None = None, reserve: Quantity | None = None,
                  strict: bool = False, compare: bool = True) -> BoundsReport:
    """TTFT, TPOT and throughput bounds for one replica.

    ``context``: tokens each decoding sequence already holds in KV.
    ``prompt`` / ``chunk``: one request's prefill, in chunks of ``chunk`` tokens.
    ``draft``: MTP draft tokens per step (0 = no MTP); ``accept`` their rates.
    ``strict``: every operator at the device's fastest tensor rate, so the bounds
    hold whatever precision an implementation computes in. Otherwise each
    operator computes in its storage dtype, and (with ``compare``) the report
    notes any figure the strict rule would lower by more than 0.5%.
    """
    kwargs = dict(context=context, batch=batch, prompt=prompt, chunk=chunk, draft=draft,
                  accept=accept, reserve=reserve)
    if strict:
        hw = hw.fastest_compute()
    if draft and not model.mtp_layers:
        raise SpecError(f"{model.name} has no MTP layers")
    if draft < 0:
        raise SpecError("draft must be >= 0")
    if dep.mtp != (draft > 0):
        raise SpecError("Deployment.mtp must be set exactly when draft tokens are used")
    if context is None and prompt is None:
        raise SpecError("give a decode context, a prompt length, or both")
    dep.validate(model)
    builder = GraphBuilder(model, dep.parallel, layout=dep.layout)
    sec: dict[str, list[Item]] = {"decode": [], "mtp": [], "prefill": []}
    rep = BoundsReport(model.name, hw.name, {}, sec)
    ms = ("ms", 1e3)

    if context is not None:
        if context < 0:
            raise SpecError("context must be >= 0")
        b1 = pass_bound(builder, hw, _decode_passes(1, context, 0))
        rep.tables["decode@1"] = b1
        sec["decode"] += [
            Item("tpot_b1", "TPOT, batch 1", b1.time.to(*ms), _breakdown(b1)),
            Item("tps_seq_b1", "tokens/s per sequence, batch 1",
                 (Quantity.exact(1, "token") / b1.time), None),
        ]
        if batch:
            bb = pass_bound(builder, hw, _decode_passes(batch, context, 0))
            rep.tables[f"decode@{batch}"] = bb
            sec["decode"] += [
                Item("tpot_batch", f"TPOT, batch {batch}", bb.time.to(*ms), _breakdown(bb)),
                Item("tps_batch", f"decode tokens/s, batch {batch}",
                     Quantity.exact(batch, "token") / bb.time, None),
            ]
        # Largest batch the KV memory admits at this context.
        mem = memory_report(model, hw, replace(dep, mtp=False),
                            context=max(context + 1, 1), reserve=reserve)
        key = "max_concurrency_assumed" if reserve is not None else "max_concurrency"
        try:
            bmax_q = mem.get(key)
        except KeyError:
            bmax_q = None
            rep.missing.append("maximum decode throughput: the weights leave no KV room "
                               "under this layout")
        if bmax_q is not None and batch and batch > bmax_q.value:
            rep.notes.append(f"batch {batch} exceeds the {int(bmax_q.value)} sequences KV "
                             f"memory admits at context {context:,}")
        if batch and dep.max_running and batch > dep.max_running:
            rep.notes.append(f"batch {batch} exceeds the engine's max running requests "
                             f"({dep.max_running})")
        if bmax_q is not None and bmax_q.value >= 1:
            bmax = int(bmax_q.value)
            capped = dep.max_running is not None and dep.max_running < bmax
            if capped:
                bmax = dep.max_running
            bm = pass_bound(builder, hw, _decode_passes(bmax, context, 0))
            rep.tables[f"decode@{bmax}"] = bm
            thr = Quantity.exact(bmax, "token") / bm.time
            if not capped:  # the batch itself is a KV-memory bound: carry its level
                thr = _carry(thr, bmax_q)
            why = (f"the engine's max running requests (KV memory admits {int(bmax_q.value)})"
                   if capped else "the most KV memory admits at this context")
            sec["decode"].append(Item(
                "tps_max", "decode tokens/s, max over batches", thr,
                f"at batch {bmax}, {why}; " + _breakdown(bm)))
        elif bmax_q is not None:
            rep.missing.append("maximum decode throughput: not one sequence fits in KV memory")

        bstar = [critical_batch(builder, hw, context, w) for w in (0, 1, 2)]
        if bstar[0] is None:
            rep.notes.append(f"decode stays HBM-bound at every batch at context {context:,}: "
                             "KV reads per sequence outpace its compute")
        else:
            lo, hi = (bstar[1] or bstar[0]), (bstar[2] or _MAX_BATCH_SEARCH)
            sec["decode"].append(Item(
                "critical_batch", "batch where decode turns compute-bound",
                Quantity.expected(bstar[0], "seq", min(lo, bstar[0]), max(hi, bstar[0]),
                                  model.provenance),
                "uniform routing; range spans the most and least concentrated routing"))

    if context is not None and draft:
        mb = batch or 1
        bd = pass_bound(builder, hw, _decode_passes(mb, context, draft))
        rep.tables[f"mtp@{mb}"] = bd
        sec["mtp"].append(Item(
            "mtp_step", f"verify + {draft} draft steps, batch {mb}", bd.time.to(*ms),
            _breakdown(bd)))
        if accept is None:
            rep.missing.append("--mtp-accept (acceptance rate per draft token): needed for "
                               "TPOT with MTP")
        else:
            if len(accept) != draft:
                raise SpecError(f"--mtp-accept needs {draft} rates, got {len(accept)}")
            e = mtp_expected_tokens(accept)
            sec["mtp"] += [
                Item("mtp_tokens", "tokens per step", e, "1 + Σ Π acceptance rates"),
                Item("mtp_tpot", f"TPOT with MTP, batch {mb}", (bd.time / e).to(*ms),
                     "step bound / tokens per step"),
            ]

    if prompt is not None:
        if prompt < 1:
            raise SpecError("prompt must be >= 1 token")
        c = chunk or prompt
        if c < 1:
            raise SpecError("chunk must be >= 1 token")
        passes: list[tuple[Pass, str]] = []
        for start in range(0, prompt, c):
            n = min(c, prompt - start)
            last = start + n == prompt
            passes.append((Pass(1, n, start, 1 if last else 0), "main"))
            if draft:  # the MTP layer must also see the prompt to fill its KV
                passes.append((Pass(1, n, start, 0), "mtp"))
        pb = pass_bound(builder, hw, passes)
        rep.tables["prefill"] = pb
        sec["prefill"] += [
            Item("ttft", f"TTFT, {prompt:,}-token prompt", pb.time.to(*ms),
                 _breakdown(pb) + (f"; {len(passes)} passes" if len(passes) > 1 else "")),
            Item("prefill_tps", "prefill tokens/s, this request",
                 Quantity.exact(prompt, "token") / pb.time, None),
        ]

    if hw.weakest_trust.name in ("THIRD_PARTY", "USER"):
        rep.notes.append(f"{hw.display}: some spec values are not vendor-published; bounds are "
                         f"conditional on them (rooflinebound hw {hw.name})")
    rule = ("every operator at the fastest tensor rate (--strict)" if strict else
            "each operator computes in its storage dtype")
    rep.notes.append("bounds assume datasheet peak rates, perfect overlap of compute, HBM and "
                     f"interconnect, and count only compulsory HBM traffic (weights, KV); {rule} "
                     "(docs/assumptions.md)")
    if not strict and compare:
        alt = bounds_report(model, hw, dep, strict=True, compare=False, **kwargs)
        lower = []
        for items in rep.items.values():
            for it in items:
                try:
                    q = alt.get(it.key)
                except KeyError:
                    continue
                if q.direction and abs(q.value - it.q.value) > 0.005 * it.q.value:
                    lower.append(f"{it.label.strip()} {q.format()}")
        if lower:
            rep.notes.append("if an implementation computed every operator at the fastest "
                             "tensor rate (--strict): " + "; ".join(lower))
    par = dep.parallel
    rep.inputs = {"tp": par.tp, "ep": par.ep, "compute": "strict" if strict else "storage dtype",
                  "kv_layout": dep.layout.name, "max_running": dep.max_running,
                  "context": context, "batch": batch,
                  "prompt": prompt, "chunk": chunk, "mtp_draft": draft or None,
                  "mtp_accept": accept, "reserve_gb": None if reserve is None
                  else round(reserve.value / 1e9, 3)}
    return rep
