"""Read an SGLang launch command into a deployment.

Every flag is put in one of four groups:

* **modelled**: turned into the equivalent setting here (TP, KV layout,
  chunk size, page size, running-request cap, memory fraction, MTP);
* **not modelled**: it can change memory or speed, but this package does not
  model it; the report says so;
* **no effect**: it does not change memory or speed (host, port, parsers ...);
* **unrecognised**: listed, so the user can check it.

Flags whose effect would make the figures wrong if ignored (attention context
parallelism, tree speculation, other quantization) are rejected.
Semantics follow SGLang @ e3718813ec5f.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..arch.attention import GQA, KDA, MLA
from ..arch.model import ModelArch
from ..kvlayout import KVLayout, sglang_dsa, sglang_mha, sglang_mla
from ..quantity import Quantity
from ..spec.errors import SpecError
from ..spec.hardware import HardwareSpec

__all__ = ["LaunchConfig", "parse_sglang"]

_ALIASES = {
    "--tp-size": "--tp", "--tensor-parallel-size": "--tp",
    "--ep-size": "--ep", "--expert-parallel-size": "--ep",
    "--pp-size": "--pp", "--pipeline-parallel-size": "--pp",
    "--dp-size": "--dp", "--data-parallel-size": "--dp",
    "--attention-data-parallel-size": "--attn-dp-size",
    "--attention-context-parallel-size": "--attn-cp-size",
    "--model": "--model-path",
}

_MODELLED = {
    "--tp", "--ep", "--pp", "--dp", "--kv-cache-dtype", "--dsa-prefill-backend",
    "--dsa-decode-backend", "--chunked-prefill-size", "--page-size", "--mem-fraction-static",
    "--max-running-requests", "--context-length", "--speculative-algorithm",
    "--speculative-num-steps", "--speculative-eagle-topk", "--speculative-num-draft-tokens",
    "--quantization", "--model-path", "--mamba-ssm-dtype", "--language-only",
    "--enable-dp-attention", "--attn-dp-size", "--enable-dp-lm-head", "--attn-cp-size",
    "--moe-dp-size", "--moe-dense-tp-size", "--moe-a2a-backend",
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
    dp: int = 1  # independent replicas
    attn_dp: int = 1  # data-parallel attention groups inside one replica's TP group
    dp_lm_head: bool = False
    layout: KVLayout | None = None
    chunk: int | None = None
    page_size: int | None = None
    max_running: int | None = None
    reserve: Quantity | None = None
    context_length: int | None = None
    draft: int = 0
    language_only: bool = False
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
    # Attention DP. The deprecated `--dp-size N --enable-dp-attention` means
    # `--attn-dp-size N` with one replica (arg_groups/parallel_hook.py).
    cfg.attn_dp = _int(flags, "--attn-dp-size") or 1
    if "--enable-dp-attention" in flags and cfg.attn_dp == 1:
        cfg.attn_dp, cfg.dp = cfg.dp, 1
    cfg.dp_lm_head = "--enable-dp-lm-head" in flags
    if cfg.dp_lm_head and cfg.attn_dp == 1:
        cfg.notes.append("--enable-dp-lm-head: no effect without attention DP")
        cfg.dp_lm_head = False
    if cfg.attn_dp > 1:
        cfg.notes.append(
            f"attention DP {cfg.attn_dp}: {cfg.attn_dp} groups of TP {cfg.tp // cfg.attn_dp} "
            "ranks, each caching only its own requests; "
            + ("attention, embedding and lm_head split over a group, FFN and experts over "
               "the full TP" if cfg.dp_lm_head else
               "attention and embedding split over a group, FFN, experts and lm_head over "
               "the full TP")
            + "; assumes SGLANG_ENABLE_EMBED_REPLICATION and SGLANG_SHARED_EXPERT_TP1 unset")
    for name in ("--attn-cp-size", "--moe-dp-size"):
        if (_int(flags, name) or 1) > 1:
            raise SpecError(f"{name} > 1 is not modelled")
    dense_tp = _int(flags, "--moe-dense-tp-size")
    if dense_tp is not None and dense_tp != cfg.tp:
        raise SpecError(f"--moe-dense-tp-size {dense_tp}: a dense-FFN TP other than --tp is "
                        "not modelled")
    if (flags.get("--moe-a2a-backend") or "none") != "none":
        raise SpecError("--moe-a2a-backend: an all-to-all MoE backend replicates the shared "
                        "experts, which is not modelled")
    if cfg.dp > 1:
        cfg.notes.append(f"--dp {cfg.dp}: {cfg.dp} independent replicas; figures are per "
                         "replica")

    kv = flags.get("--kv-cache-dtype") or "auto"
    page = _int(flags, "--page-size")
    if any(layer.indexer is not None for layer in model.layers):
        pre, dec = flags.get("--dsa-prefill-backend"), flags.get("--dsa-decode-backend")
        raw = "trtllm" in (pre, dec) or (pre == "tilelang" and dec == "tilelang")
        cfg.layout = sglang_dsa(kv, scaled=not raw)
        if page not in (None, 64):
            raise SpecError(f"--page-size {page}: SGLang's DSA pool on CUDA requires "
                            "64-token pages")
        cfg.page_size = 64
    elif all(isinstance(layer.attn, GQA) for layer in model.layers):
        cfg.layout = sglang_mha(kv)
        cfg.page_size = page or 1
    elif all(isinstance(layer.attn, MLA | KDA) for layer in model.layers):
        # MLA pool (no DSA) plus a linear-attention state pool for hybrid models.
        ssm = {"float32": 4, "bfloat16": 2, "float16": 2}
        dt = flags.get("--mamba-ssm-dtype") or "float32"
        if dt not in ssm:
            raise SpecError(f"--mamba-ssm-dtype {dt!r} is not modelled")
        cfg.layout = sglang_mla(kv, ssm_bytes=ssm[dt])
        cfg.page_size = page or 1
        if any(isinstance(layer.attn, KDA) for layer in model.layers):
            cfg.notes.append(f"linear-attention state: recurrent state at {dt}, conv window at "
                             "bfloat16 (SGLang defaults unless --mamba-ssm-dtype is given)")
    else:
        raise SpecError("SGLang launches are modelled for DSA (GLM-5.x), GQA/MHA and "
                        "MLA/KDA hybrid models so far")
    cfg.chunk = _int(flags, "--chunked-prefill-size")
    if cfg.chunk is not None and cfg.chunk <= 0:
        cfg.chunk = None  # SGLang: -1 disables chunking
    cfg.max_running = _int(flags, "--max-running-requests")
    cfg.language_only = "--language-only" in flags
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
