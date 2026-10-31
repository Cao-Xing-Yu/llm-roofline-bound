"""Memory: parameters, weights per GPU, KV cache, and what fits.

Everything here is EXACT arithmetic on the tensor inventory and the hardware
spec, except where noted:

* The KV pool per GPU is at most ``capacity - weights``: an upper BOUND, since
  the runtime also needs activations, workspaces and graphs, which no config
  determines. With ``reserve`` (ASSUMED, from the user) it becomes
  ``capacity - weights - reserve``.
* Counts derived from the pool (tokens, concurrent requests) inherit its level.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from .arch.ffn import MoE
from .arch.model import DecoderLayer, ModelArch
from .kvlayout import KV_DTYPE_BYTES, KVLayout, uniform
from .parallel import Parallel
from .quantity import Quantity
from .spec.errors import SpecError
from .spec.hardware import HardwareSpec

__all__ = ["KV_DTYPE_BYTES", "Deployment", "Item", "MemoryReport", "memory_report"]


@dataclass(frozen=True)
class Deployment:
    parallel: Parallel = field(default_factory=Parallel)
    mtp: bool = False  # load and cache the MTP layers
    kv_dtype: str = "bf16"
    index_kv_dtype: str = "bf16"
    page_size: int = 1  # KV allocation granularity, tokens
    kv_layout: KVLayout | None = None  # an engine's exact layout; overrides the dtypes
    max_running: int | None = None  # engine cap on sequences decoding at once

    @property
    def layout(self) -> KVLayout:
        return self.kv_layout or uniform(self.kv_dtype, self.index_kv_dtype)

    def with_parallel(self, par: Parallel) -> Deployment:
        return replace(self, parallel=par)

    def validate(self, model: ModelArch) -> None:
        self.layout  # noqa: B018 -- raises on an unknown dtype
        if self.max_running is not None and self.max_running < 1:
            raise SpecError("max_running must be >= 1")
        if self.page_size < 1:
            raise SpecError("page_size must be >= 1")
        if self.mtp and not model.mtp_layers:
            raise SpecError(f"{model.name} has no MTP layers")
        self.parallel.validate(model, mtp=self.mtp)


@dataclass(frozen=True)
class Item:
    key: str
    label: str
    q: Quantity
    note: str | None = None


@dataclass
class MemoryReport:
    model: str
    hardware: str
    inputs: dict
    items: dict[str, list[Item]]  # section -> items
    missing: list[str]  # outputs not computed, and the input each needs
    notes: list[str]

    def get(self, key: str) -> Quantity:
        for items in self.items.values():
            for it in items:
                if it.key == key:
                    return it.q
        raise KeyError(key)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "hardware": self.hardware,
            "inputs": self.inputs,
            "sections": {s: {it.key: {**it.q.to_dict(), "label": it.label, "note": it.note}
                             for it in items} for s, items in self.items.items()},
            "missing": self.missing,
            "notes": self.notes,
        }


# ------------------------------------------------------------------ counting


def _exact(model: ModelArch, value: float, unit: str, how: str) -> Quantity:
    return Quantity.exact(value, unit, model.provenance, derivation=how)


def param_counts(model: ModelArch) -> dict[str, int]:
    """Parameters per component (scales excluded); MTP layers as one bucket."""
    mtp_ids = {layer.index for layer in model.mtp_layers}
    out: dict[str, int] = {}
    for t in model.inventory:
        if t.is_scale:
            continue
        key = "mtp_layers" if t.layer in mtp_ids else t.component
        out[key] = out.get(key, 0) + t.numel
    return out


def active_params_per_token(model: ModelArch) -> int:
    """Parameters one token multiplies through in the main model.

    All non-expert weights, ``top_k`` routed experts per MoE layer, and the
    lm_head. The embedding is a row lookup and is excluded; MTP is excluded.
    """
    total = 0
    by_layer: dict[int, DecoderLayer] = {layer.index: layer for layer in model.layers}
    for t in model.tensors(mtp=False):
        if t.is_scale or t.component == "embed":
            continue
        if t.component == "routed_expert":
            ffn = by_layer[t.layer].ffn
            assert isinstance(ffn, MoE)
            total += t.numel * ffn.top_k / ffn.n_experts  # exact: every expert is the same shape
        else:
            total += t.numel
    return round(total)


def _stage_weights(model: ModelArch, dep: Deployment) -> list[int]:
    par = dep.parallel
    stages = par.stage_layers(model)
    out = [0] * par.pp
    for t, n in model.tensor_groups(mtp=dep.mtp):
        out[par.stage_of(model, t, stages)] += par.rank_bytes(t) * n
    return [int(b) for b in out]  # each stage's sum is a whole number of bytes


def _stage_kv_per_token(model: ModelArch, dep: Deployment) -> list[int]:
    """KV bytes per token held by one rank of each stage.

    MLA latents and indexer keys are needed whole by every TP rank, so they are
    replicated across TP, not divided.
    """
    par = dep.parallel
    lay = dep.layout
    layers = {layer.index: layer for layer in model.all_layers}
    stages = par.stage_layers(model)
    if dep.mtp:
        stages[-1] = stages[-1] + [layer.index for layer in model.mtp_layers]
    out = []
    for ids in stages:
        b = 0
        for i in ids:
            layer = layers[i]
            b += lay.main_bytes(layer.attn)
            if layer.indexer is not None:
                b += lay.index_key_bytes(layer.indexer)
        out.append(b)
    return out


def _deployed_bytes(model: ModelArch, dep: Deployment) -> int:
    return sum(t.nbytes * n for t, n in model.tensor_groups(mtp=dep.mtp))


# ------------------------------------------------------------- layout search


def smallest_layout(model: ModelArch, hw: HardwareSpec, dep: Deployment,
                    need_per_rank: Callable[[Deployment], list[int]] | None = None,
                    max_gpus: int = 512) -> Parallel | None:
    """Fewest GPUs over TP x PP (EP in {1, TP}) whose weights leave room on every rank.

    ``need_per_rank(deployment)`` returns extra bytes each stage's ranks must also
    hold, e.g. one request's KV, which depends on the layout.
    """
    capacity = hw["mem_capacity"].value
    candidates = []
    tp = 1
    while tp <= max_gpus:
        for pp in range(1, len(model.layers) + 1):
            if tp * pp > max_gpus:
                break
            for ep in sorted({1, tp}):
                candidates.append(Parallel(tp, pp, ep))
        tp *= 2
    candidates.sort(key=lambda p: (p.gpus, p.pp, -p.ep))
    for par in candidates:
        d = dep.with_parallel(par)
        try:
            d.validate(model)
        except SpecError:
            continue
        w = _stage_weights(model, d)
        extra = need_per_rank(d) if need_per_rank else [0] * par.pp
        if all(ws + e < capacity for ws, e in zip(w, extra, strict=True)):
            return par
    return None


def _pool_items(model, cap, stage_w, stage_kv, par, req_tokens, context, reserve,
                sec, missing) -> None:
    """KV pool, tokens and concurrency: as BOUNDs, and under the user's reserve."""
    pools = []
    for s, (w, k) in enumerate(zip(stage_w, stage_kv, strict=True)):
        pool = (cap - _exact(model, w, "B", "weights on stage")).as_bound("upper")
        tokens = (pool / _exact(model, k, "B/token", "")).floor()
        pools.append((s, pool, tokens))
    binding, pool, tokens = min(pools, key=lambda x: x[2].value)
    sec["bounds"] += [
        Item("kv_pool", "KV pool per GPU", pool,
             "capacity - weights; runtime memory not subtracted"),
        Item("kv_tokens", "tokens of KV per replica", tokens,
             f"pool / KV per token (stage {binding} binds)" if par.pp > 1 else
             "pool / KV per token"),
    ]
    if req_tokens is not None:
        sec["bounds"].append(Item(
            "max_concurrency", f"concurrent requests at {context:,} tokens",
            (tokens / Quantity.exact(req_tokens, "token/req")).floor(),
            "KV memory only; says nothing about latency"))

    if reserve is None:
        missing.append("--reserve-gb (runtime memory per GPU besides weights and KV): needed "
                       "for the KV pool, token and concurrency figures under that assumption")
        return
    w = _exact(model, stage_w[binding], "B", "weights on stage")
    if cap.value - w.value - reserve.value <= 0:  # checked first: quantities are non-negative
        missing.append(f"weights plus reserve exceed GPU memory on stage {binding}: "
                       "nothing left for KV")
        return
    apool = (cap - w - reserve).as_bound("upper")
    atok = (apool / _exact(model, stage_kv[binding], "B/token", "")).floor()
    sec["assumed"] += [
        Item("kv_pool_assumed", "KV pool per GPU", apool, "capacity - weights - reserve"),
        Item("kv_tokens_assumed", "tokens of KV per replica", atok, None),
    ]
    if req_tokens is not None:
        sec["assumed"].append(Item(
            "max_concurrency_assumed", f"concurrent requests at {context:,} tokens",
            (atok / Quantity.exact(req_tokens, "token/req")).floor(),
            "KV memory only; says nothing about latency"))


# -------------------------------------------------------------------- report


def memory_report(model: ModelArch, hw: HardwareSpec, dep: Deployment, *,
                  context: int | None = None, reserve: Quantity | None = None
                  ) -> MemoryReport:
    """Every memory figure for one replica, labelled by level.

    ``context`` is the tokens one request holds in KV (prompt plus output).
    ``reserve`` is per-GPU memory for everything except weights and KV
    (activations, CUDA graphs, workspaces); ASSUMED, supplied by the user.
    """
    dep.validate(model)
    if context is not None and context < 1:
        raise SpecError("context must be >= 1 token")
    par = dep.parallel
    sec: dict[str, list[Item]] = {"model": [], "per_gpu": [], "bounds": [], "assumed": []}
    missing: list[str] = []
    notes: list[str] = []

    # --- model (EXACT, independent of deployment)
    counts = param_counts(model)
    main = sum(v for k, v in counts.items() if k != "mtp_layers")
    sec["model"] += [
        Item("params_main", "parameters (main model)",
             _exact(model, main, "param", "sum of tensor sizes, scales excluded")),
        *[Item(f"params_{k}", f"  {k}", _exact(model, v, "param", "tensor sizes"))
          for k, v in sorted(counts.items(), key=lambda kv: -kv[1]) if k != "mtp_layers"],
        Item("params_mtp", "parameters (MTP layers)",
             _exact(model, counts.get("mtp_layers", 0), "param", "tensor sizes")),
        Item("checkpoint_bytes", "checkpoint size",
             _exact(model, sum(t.nbytes for t in model.inventory), "B",
                    "sum of numel x dtype bytes, scales included")),
        Item("active_params", "active parameters per token",
             _exact(model, active_params_per_token(model), "param",
                    "non-expert + top_k/E of routed experts + lm_head; embedding and MTP "
                    "excluded")),
    ]

    # --- per GPU (EXACT for the stated layout)
    stage_w = _stage_weights(model, dep)
    stage_kv = _stage_kv_per_token(model, dep)
    page = dep.page_size
    sec["per_gpu"] += [
        Item("weights_per_gpu", "weights per GPU (largest stage)",
             _exact(model, max(stage_w), "B", f"tensor bytes per rank under {par}")),
        Item("kv_per_token", "KV cache per token per GPU (largest stage)",
             _exact(model, max(stage_kv), "B/token",
                    f"layout {dep.layout.name}; "
                    "replicated across TP")),
    ]
    vocab_pad = par.tp * math.ceil(model.vocab / par.tp) - model.vocab
    if vocab_pad:
        notes.append(f"vocab padded by {vocab_pad} rows to split over TP={par.tp}")
    req_tokens = None
    if context is not None:
        req_tokens = math.ceil(context / page) * page
        sec["per_gpu"].append(Item(
            "kv_per_request", f"KV cache per request at {context:,} tokens",
            _exact(model, req_tokens * max(stage_kv), "B/req",
                   f"ceil({context}/{page}) x {page} tokens x KV per token")))

    # --- bounds (no assumptions): the pool cannot exceed capacity minus weights
    cap = hw["mem_capacity"]
    deployed = _exact(model, _deployed_bytes(model, dep), "B", "weights loaded for serving")
    sec["bounds"].append(Item(
        "min_gpus_weights", "GPUs to hold the weights (any layout)",
        (deployed / cap).ceil().as_bound("lower") * Quantity.exact(1, "gpu"),
        "ceil(weights / capacity per GPU)"))

    def kv_need(d: Deployment) -> list[int]:
        if req_tokens is None:
            return [0] * d.parallel.pp
        return [req_tokens * k for k in _stage_kv_per_token(model, d)]

    best = smallest_layout(model, hw, dep, kv_need)
    if best is None:
        missing.append("no TP x PP layout up to 512 GPUs fits the weights")
    else:
        what = "weights + 1 request" if req_tokens else "weights"
        sec["bounds"].append(Item(
            "min_gpus_layout", f"GPUs for {what} (TP x PP layouts)",
            Quantity.exact(best.gpus, "gpu", cap.provenance[0], model.provenance)
            .as_bound("lower"),
            f"smallest feasible: {best}"))

    cap_b = cap.value
    if max(stage_w) >= cap_b:
        missing.append(f"weights do not fit under {par}: {max(stage_w) / 1e9:.3f} GB per GPU "
                       f"vs {cap_b / 1e9:.3f} GB capacity; KV pool and concurrency not computed")
    else:
        _pool_items(model, cap, stage_w, stage_kv, par, req_tokens, context, reserve,
                    sec, missing)

    if hw.weakest_trust.name in ("THIRD_PARTY", "USER"):
        notes.append(f"{hw.display}: some spec values are not vendor-published; bounds are "
                     "conditional on them (rooflinebound hw " + hw.name + ")")

    inputs = {"tp": par.tp, "pp": par.pp, "ep": par.ep, "mtp": dep.mtp,
              "kv_layout": dep.layout.name,
              "page_size": dep.page_size, "context": context,
              "reserve_gb": None if reserve is None else round(reserve.value / 1e9, 3)}
    return MemoryReport(model.name, hw.name, inputs, sec, missing, notes)
