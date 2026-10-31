"""Read an SGLang launch command into a deployment.

Every flag is put in one of four groups:

* **modelled**: turned into the equivalent setting here (TP, KV layout,
  chunk size, page size, running-request cap, memory fraction, MTP);
* **not modelled**: it can change memory or speed, but this package does not
  model it; the report says so;
* **no effect**: it does not change memory or speed (host, port, parsers ...);
* **unrecognised**: listed, so the user can check it.

Flags whose effect would make the figures wrong if ignored (data-parallel
attention, tree speculation, other quantization) are rejected.
Semantics follow SGLang @ e3718813ec5f.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..arch.model import ModelArch
from ..kvlayout import KVLayout, sglang_dsa
from ..quantity import Quantity
from ..spec.errors import SpecError
from ..spec.hardware import HardwareSpec

__all__ = ["LaunchConfig", "parse_sglang"]

_ALIASES = {
    "--tp-size": "--tp", "--tensor-parallel-size": "--tp",
    "--ep-size": "--ep", "--expert-parallel-size": "--ep",
    "--pp-size": "--pp", "--pipeline-parallel-size": "--pp",
    "--dp-size": "--dp", "--data-parallel-size": "--dp",
    "--model": "--model-path",
}

_MODELLED = {
    "--tp", "--ep", "--pp", "--dp", "--kv-cache-dtype", "--dsa-prefill-backend",
    "--dsa-decode-backend", "--chunked-prefill-size", "--page-size", "--mem-fraction-static",
    "--max-running-requests", "--context-length", "--speculative-algorithm",
    "--speculative-num-steps", "--speculative-eagle-topk", "--speculative-num-draft-tokens",
    "--quantization", "--model-path",
}

_HICACHE = "hierarchical (host-memory) prefix cache: prefix hits shorten TTFT; bounds assume no hit"
_GRAPHS = "CUDA graphs: launch overhead is not modelled"
_SCHED = "scheduling is not modelled"
_KERNELS = "kernel choices are not modelled; bounds hold for any kernel"
_NOT_MODELLED = {
    "--enable-hierarchical-cache": _HICACHE, "--hicache-size": _HICACHE,
    "--hicache-ratio": _HICACHE, "--hicache-write-policy": _HICACHE,
    "--hicache-io-backend": _HICACHE, "--hicache-storage-backend": _HICACHE,
    "--cuda-graph-max-bs": _GRAPHS, "--cuda-graph-max-bs-decode": _GRAPHS,
    "--cuda-graph-bs": _GRAPHS, "--disable-cuda-graph": _GRAPHS,
    "--disable-radix-cache": "prefix (radix) caching is not modelled either way",
    "--schedule-policy": _SCHED, "--max-prefill-tokens": _SCHED,
    "--schedule-conservativeness": _SCHED,
    "--enable-torch-compile": _KERNELS, "--attention-backend": _KERNELS,
    "--moe-runner-backend": _KERNELS,
}

_NO_EFFECT = {
    "--host", "--port", "--served-model-name", "--reasoning-parser", "--tool-call-parser",
    "--enable-metrics", "--enable-cache-report", "--preferred-sampling-params",
    "--enable-strict-thinking", "--max-queued-requests", "--api-key", "--log-level",
    "--log-requests", "--trust-remote-code", "--chat-template", "--tokenizer-path",
    "--watchdog-timeout", "--enable-request-time-stats-logging", "--decode-log-interval",
    "--show-time-cost", "--file-storage-path", "--random-seed", "--tokenizer-mode",
    "--skip-server-warmup", "--allow-auto-truncate",
}

_REJECT = {
    "--enable-dp-attention": "data-parallel attention changes how the MLA cache is "
                             "replicated, which is not modelled",
    "--enable-two-batch-overlap": "two-batch overlap is not modelled",
    "--disaggregation-mode": "prefill/decode disaggregation is not modelled",
}

_BOOLEAN_PREFIXES = ("--enable-", "--disable-")


@dataclass
class LaunchConfig:
    engine: str
    tp: int = 1
    ep: int = 1
    pp: int = 1
    dp: int = 1
    layout: KVLayout | None = None
    chunk: int | None = None
    page_size: int | None = None
    max_running: int | None = None
    reserve: Quantity | None = None
    context_length: int | None = None
    draft: int = 0
    notes: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    unrecognised: list[str] = field(default_factory=list)


def _split(tokens: Sequence[str]) -> tuple[str, dict[str, str | None]]:
    """(program, {flag: value or None}); accepts ``--flag value`` and ``--flag=value``."""
    i = 0
    while i < len(tokens) and not tokens[i].startswith("--"):
        i += 1
    program = " ".join(tokens[:i])
    if "sglang" not in program:
        raise SpecError(f"not an SGLang launch command: {program or tokens[:1]!r}")
    flags: dict[str, str | None] = {}
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            raise SpecError(f"unexpected argument {tok!r} in the launch command")
        name, eq, value = tok.partition("=")
        name = _ALIASES.get(name, name)
        if not eq:
            value = None
            boolean = name.startswith(_BOOLEAN_PREFIXES) or name in _REJECT
            if not boolean and i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                value = tokens[i + 1]
                i += 1
        flags[name] = value
        i += 1
    return program, flags


def _int(flags: dict, name: str) -> int | None:
    v = flags.get(name)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        raise SpecError(f"{name} expects an integer, got {v!r}") from None


def parse_sglang(tokens: Sequence[str], model: ModelArch, hw: HardwareSpec) -> LaunchConfig:
    _, flags = _split(tokens)
    for name, why in _REJECT.items():
        if name in flags:
            raise SpecError(f"{name}: {why}")
    cfg = LaunchConfig("sglang")
    cfg.tp = _int(flags, "--tp") or 1
    cfg.ep = _int(flags, "--ep") or 1
    cfg.pp = _int(flags, "--pp") or 1
    cfg.dp = _int(flags, "--dp") or 1
    if cfg.dp > 1:
        cfg.notes.append(f"--dp {cfg.dp}: {cfg.dp} independent replicas; figures are per "
                         "replica")

    has_dsa = any(layer.indexer is not None for layer in model.layers)
    if not has_dsa:
        raise SpecError("SGLang launches are modelled for DSA models (GLM-5.x) so far")
    kv = flags.get("--kv-cache-dtype") or "auto"
    pre, dec = flags.get("--dsa-prefill-backend"), flags.get("--dsa-decode-backend")
    raw = "trtllm" in (pre, dec) or (pre == "tilelang" and dec == "tilelang")
    cfg.layout = sglang_dsa(kv, scaled=not raw)

    page = _int(flags, "--page-size")
    if page not in (None, 64):
        raise SpecError(f"--page-size {page}: SGLang's DSA pool on CUDA requires 64-token pages")
    cfg.page_size = 64
    cfg.chunk = _int(flags, "--chunked-prefill-size")
    if cfg.chunk is not None and cfg.chunk <= 0:
        cfg.chunk = None  # SGLang: -1 disables chunking
    cfg.max_running = _int(flags, "--max-running-requests")
    cfg.context_length = _int(flags, "--context-length")

    if "--mem-fraction-static" in flags:
        f = float(flags["--mem-fraction-static"])
        if not 0 < f < 1:
            raise SpecError(f"--mem-fraction-static must be in (0, 1), got {f}")
        # SGLang keeps (1 - f) x (free memory before loading the model) out of the KV
        # pool. Free memory before loading is close to, but below, device capacity.
        cap = hw["mem_capacity"]
        cfg.reserve = Quantity.assumed((1 - f) * cap.value, "B",
                                       "sglang:--mem-fraction-static")
        cfg.notes.append(f"--mem-fraction-static {f}: runtime reserve taken as (1 - {f}) x "
                         f"device memory = {(1 - f) * cap.value / 2**30:.2f} GiB (ASSUMED: "
                         "SGLang uses free memory before model load, slightly less)")
    else:
        cfg.notes.append("no --mem-fraction-static: SGLang picks one at startup, so the "
                         "runtime reserve is unknown")

    q = flags.get("--quantization")
    if q is not None and not (q == "fp8" and model.quant is not None):
        raise SpecError(f"--quantization {q}: only the checkpoint's own fp8 is modelled")

    algo = flags.get("--speculative-algorithm")
    if algo is not None:
        if algo.upper() not in ("NEXTN", "EAGLE"):
            raise SpecError(f"--speculative-algorithm {algo} is not modelled")
        steps = _int(flags, "--speculative-num-steps") or 0
        topk = _int(flags, "--speculative-eagle-topk") or 1
        if topk != 1:
            raise SpecError("tree speculation (--speculative-eagle-topk > 1) is not modelled")
        if steps < 1:
            raise SpecError("--speculative-num-steps must be >= 1")
        cfg.draft = steps

    seen_not_modelled: set[str] = set()
    for name in flags:
        if name in _MODELLED:
            continue
        if name in _NOT_MODELLED:
            why = _NOT_MODELLED[name]
            if why not in seen_not_modelled:  # one note per family of flags
                cfg.notes.append(f"{name}: {why}")
                seen_not_modelled.add(why)
        elif name in _NO_EFFECT:
            cfg.ignored.append(name)
        else:
            cfg.unrecognised.append(name)
    return cfg
