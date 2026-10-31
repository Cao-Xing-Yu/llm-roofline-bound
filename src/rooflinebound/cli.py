"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata

from . import __version__
from .bounds import BoundsReport, PassBound, bounds_report
from .capacity import Targets
from .engines import LaunchConfig, parse_sglang
from .i18n import LANGS, t
from .memory import Deployment, MemoryReport, memory_report
from .parallel import Parallel
from .quantity import Quantity, Trust
from .spec import SpecError, list_hardware, list_models, load_hardware, load_model

# Display units for hardware fields: (unit label, scale from base units).
_HW_DISPLAY = {
    "mem_capacity": ("GiB", 2**-30),
    "mem_bw": ("GB/s", 1e-9),
    "tflops_bf16": ("TFLOP/s", 1e-12),
    "tflops_fp8": ("TFLOP/s", 1e-12),
    "tflops_fp4": ("TFLOP/s", 1e-12),
    "link_bw": ("GB/s", 1e-9),
    "devices_per_node": ("device", 1),
}


def _cmd_hw(args: argparse.Namespace) -> int:
    if args.list:
        for name in list_hardware():
            print(name)
        return 0
    spec = load_hardware(args.name)
    if args.json:
        out = {
            "name": spec.name,
            "display": spec.display,
            "fields": {
                k: {**f.quantity.to_dict(), "note": f.note, "disputed": f.disputed}
                for k, f in spec.fields.items()
            },
        }
        json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0

    print(f"{spec.display}  ({spec.path})")
    for key, f in spec.fields.items():
        unit, scale = _HW_DISPLAY[key]
        shown = f.quantity.to(unit, scale)
        flag = "  DISPUTED" if f.disputed else ""
        print(f"  {key:<17} {shown.format():<32} {f.quantity.trust.name}{flag}")
    if spec.weakest_trust >= Trust.THIRD_PARTY:
        print("\nNote: some values are not vendor-published. Every bound computed on this "
              "spec is conditional on them.")
    for key in spec.disputed_fields:
        print(f"Disputed {key}: {spec.fields[key].note}")
    return 0


# Display units for report quantities, by base unit.
_DISPLAY = {"B": ("GB", 1e-9), "B/req": ("GB/req", 1e-9), "req/s": ("req/s", 1)}

_SECTIONS = {
    "model": "Model  (EXACT: arithmetic on the published config)",
    "per_gpu": "Per GPU  ({layout})",
    "bounds": "Bounds  (no assumptions)",
    "assumed": "Under your assumptions",
    "decode": "Decode at context {context}  (bounds; their assumptions are under Notes)",
    "mtp": "MTP decode  (tokens per step and TPOT rest on your acceptance rates)",
    "prefill": "Prefill  (bounds; their assumptions are under Notes)",
    "capacity": "Sustained capacity, SLA and fleet  (any schedule; assumptions under Notes)",
}


def _show(q: Quantity) -> str:
    unit, scale = _DISPLAY.get(str(q.unit), (None, 1))
    return (q.to(unit, scale) if unit else q).format()


def _width(s: str) -> int:
    """Terminal columns: CJK characters take two."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def _pad(s: str, width: int) -> str:
    return s + " " * max(width - _width(s), 0)


def _render(r: MemoryReport | BoundsReport, lang: str = "en") -> None:
    i = r.inputs
    fmt = {"layout": f"TP{i['tp']} PP{i.get('pp', 1)} EP{i['ep']}",
           "context": f"{i['context']:,}" if i.get("context") is not None else "-"}
    print(f"{r.model} {t('on', lang)} {r.hardware}")
    sep = "：" if lang == "zh" else ": "
    print(t("Inputs", lang) + sep + ", ".join(f"{k}={v}" for k, v in i.items() if v is not None))
    for sec, items in r.items.items():
        if not items:
            continue
        print(f"\n{t(_SECTIONS[sec], lang).format(**fmt)}")
        for it in items:
            line = f"  {_pad(t(it.label, lang), 48)} {_show(it.q)}"
            print(line + (f"   -- {t(it.note, lang)}" if it.note and sec != "model" else ""))
    if r.missing:
        print(f"\n{t('Not computed', lang)}")
        for m in r.missing:
            print(f"  - {t(m, lang)}")
    if r.notes:
        print(f"\n{t('Notes', lang)}")
        for n in r.notes:
            print(f"  - {t(n, lang)}")


_LAUNCH_FIELDS = (("tp", "tp"), ("ep", "ep"), ("pp", "pp"), ("page_size", "page_size"),
                  ("chunk", "chunk"), ("mtp_draft", "draft"))


def _resolve(args: argparse.Namespace, model, hw) -> tuple[Deployment, dict, LaunchConfig | None]:
    """Deployment from CLI flags, or from a launch command given after ``--``.

    A flag given on both sides must agree: nothing is silently overridden.
    """
    launch = parse_sglang(args.launch, model, hw) if args.launch else None
    v = {"tp": args.tp, "ep": args.ep, "pp": getattr(args, "pp", None),
         "page_size": args.page_size, "chunk": getattr(args, "chunk", None),
         "mtp_draft": getattr(args, "mtp_draft", None)}
    layout = max_running = reserve = None
    if launch is not None:
        for arg, attr in _LAUNCH_FIELDS:
            mine, theirs = v[arg], getattr(launch, attr)
            if mine is not None and theirs is not None and mine != theirs:
                flag = "--" + arg.replace("_", "-")
                raise SpecError(f"{flag} {mine} conflicts with the launch command ({theirs})")
            if mine is None:
                v[arg] = theirs
        if args.kv_dtype or args.index_kv_dtype:
            raise SpecError("--kv-dtype / --index-kv-dtype come from the launch command; "
                            "drop them")
        layout, max_running, reserve = launch.layout, launch.max_running, launch.reserve
    if args.reserve_gb is not None:
        if reserve is not None:
            raise SpecError("--reserve-gb conflicts with --mem-fraction-static in the launch "
                            "command")
        reserve = Quantity.assumed(args.reserve_gb * 1e9, "B", "user:--reserve-gb")
    draft = v["mtp_draft"] or 0
    mtp = draft > 0 or bool(getattr(args, "mtp", False))
    dep = Deployment(Parallel(v["tp"] or 1, v["pp"] or 1, v["ep"] or 1), mtp,
                     args.kv_dtype or "bf16", args.index_kv_dtype or "bf16",
                     v["page_size"] or 1, kv_layout=layout, max_running=max_running)
    if launch is not None and launch.context_length:
        for name in ("context", "prompt"):
            n = getattr(args, name, None)
            if n is not None and n > launch.context_length:
                raise SpecError(f"--{name} {n:,} exceeds the server's --context-length "
                                f"{launch.context_length:,}")
    return dep, {"reserve": reserve, "chunk": v["chunk"], "draft": draft}, launch


def _launch_notes(r: MemoryReport | BoundsReport, launch: LaunchConfig | None) -> None:
    if launch is None:
        return
    r.inputs = {"engine": launch.engine, **r.inputs}
    r.notes += [f"launch: {n}" for n in launch.notes]
    if launch.ignored:
        r.notes.append("launch flags with no effect on these figures: "
                       + " ".join(launch.ignored))
    if launch.unrecognised:
        r.notes.append("launch flags not recognised (check they do not change memory or "
                       "speed): " + " ".join(launch.unrecognised))


def _emit(r, args) -> None:
    if args.json:
        json.dump(r.to_dict(), sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        _render(r, args.lang)


def _cmd_memory(args: argparse.Namespace) -> int:
    if args.list_models:
        for name in list_models():
            print(name)
        return 0
    model = load_model(args.model)
    hw = load_hardware(args.hw)
    dep, extra, launch = _resolve(args, model, hw)
    r = memory_report(model, hw, dep, context=args.context, reserve=extra["reserve"])
    _launch_notes(r, launch)
    _emit(r, args)
    return 0


def _render_ops(title: str, b: PassBound, hw) -> None:
    """Per-operator table: where the time goes, at peak rates."""
    peaks = {d: hw.peak_flops(d).value for d in {o.dtype for o in b.ops}}
    bw = hw["mem_bw"].value
    print(f"\nOperators: {title}  (MLA {b.mla_mode}; times per pass at peak, expected routing)")
    print(f"  {'operator':<34} {'n':>3} {'GFLOP':>10} {'MB':>10} {'FLOP/B':>8} "
          f"{'t_comp us':>10} {'t_hbm us':>10}  bound")
    serial = 0.0
    for o in b.ops:
        f, m = o.count * o.flops, o.count * o.hbm_bytes()
        tc, tm = f / peaks[o.dtype], m / bw
        serial += max(tc, tm)
        ai = f"{f / m:8.1f}" if m else f"{'-':>8}"
        print(f"  {o.name:<34} {o.count:>3} {f / 1e9:>10.3f} {m / 1e6:>10.2f} {ai} "
              f"{tc * 1e6:>10.1f} {tm * 1e6:>10.1f}  {'compute' if tc > tm else 'HBM'}")
    print(f"  Σ per-operator max: {serial * 1e3:.3f} ms -- tighter, but only if operators "
          "never overlap; not a bound")


def _cmd_bounds(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    hw = load_hardware(args.hw)
    dep, extra, launch = _resolve(args, model, hw)
    accept = None
    if args.mtp_accept:
        try:
            accept = [float(x) for x in args.mtp_accept.split(",")]
        except ValueError:
            raise SpecError("--mtp-accept takes comma-separated rates, e.g. 0.8,0.6") from None
    targets = Targets(args.output, args.sla_tpot_ms, args.sla_ttft_ms, args.target_qps,
                      args.price_gpu_hour)
    r = bounds_report(model, hw, dep, context=args.context, batch=args.batch,
                      prompt=args.prompt, chunk=extra["chunk"], draft=extra["draft"],
                      accept=accept, reserve=extra["reserve"], strict=args.strict,
                      targets=targets)
    _launch_notes(r, launch)
    _emit(r, args)
    if args.ops and not args.json:
        for title, b in r.tables.items():
            _render_ops(title, b, hw)
    return 0


def _deployment_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=None, help="bundled name or path to a config.json")
    p.add_argument("--hw", default=None, help="bundled name or path to a .toml spec")
    p.add_argument("--tp", type=int, help="tensor parallel size (default 1)")
    p.add_argument("--ep", type=int, help="1, or equal to --tp (default 1)")
    p.add_argument("--kv-dtype", choices=["bf16", "fp16", "fp8"],
                   help="uniform KV layout: every cached element at this width (default bf16)")
    p.add_argument("--index-kv-dtype", choices=["bf16", "fp16", "fp8"],
                   help="width of cached DSA indexer keys (default bf16)")
    p.add_argument("--page-size", type=int, help="KV allocation unit, tokens (default 1)")
    p.add_argument("--reserve-gb", type=float,
                   help="ASSUMED runtime memory per GPU besides weights and KV")
    p.add_argument("--json", action="store_true", help="machine-readable output (English)")
    p.add_argument("--lang", choices=LANGS, default="en", help="report language")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rooflinebound",
        description="Provable lower bounds and clearly-labelled estimates for LLM inference.",
        epilog="memory and bounds accept an SGLang launch command after a bare --, e.g. "
               "rooflinebound bounds --model glm-5.3 --hw h20-3e --context 32768 -- "
               "sglang serve --tp 8 --kv-cache-dtype fp8_e4m3 ...",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    hw = sub.add_parser("hw", help="show an accelerator spec and where each value comes from")
    hw.add_argument("name", nargs="?", help="bundled name or path to a .toml spec")
    hw.add_argument("--list", action="store_true", help="list bundled specs")
    hw.add_argument("--json", action="store_true")
    hw.set_defaults(func=_cmd_hw)

    mem = sub.add_parser("memory", help="weights, KV cache and what fits, per GPU")
    _deployment_args(mem)
    mem.add_argument("--list-models", action="store_true")
    mem.add_argument("--pp", type=int, help="pipeline parallel size (default 1)")
    mem.add_argument("--mtp", action="store_true", help="load and cache the MTP layers")
    mem.add_argument("--context", type=int, help="tokens one request keeps in KV")
    mem.set_defaults(func=_cmd_memory)

    bd = sub.add_parser("bounds", help="TTFT, TPOT and throughput bounds for one replica")
    _deployment_args(bd)
    bd.add_argument("--context", type=int, help="decode: tokens each sequence holds in KV")
    bd.add_argument("--batch", type=int, help="decode: sequences decoding together")
    bd.add_argument("--prompt", type=int, help="prefill: prompt tokens of one request")
    bd.add_argument("--chunk", type=int, help="prefill: chunk size (default: whole prompt)")
    bd.add_argument("--mtp-draft", type=int, help="MTP draft tokens per step")
    bd.add_argument("--mtp-accept", help="ASSUMED acceptance rate per draft token, e.g. 0.8,0.6")
    bd.add_argument("--output", type=int, help="output tokens per request (with --prompt): "
                    "sustained requests/s, fleet size, cost")
    bd.add_argument("--sla-tpot-ms", type=float, help="target time per output token")
    bd.add_argument("--sla-ttft-ms", type=float, help="target time to first token")
    bd.add_argument("--target-qps", type=float, help="requests/s the fleet must serve")
    bd.add_argument("--price-gpu-hour", type=float, help="USD per GPU-hour, for cost")
    bd.add_argument("--ops", action="store_true", help="print the per-operator tables")
    bd.add_argument("--strict", action="store_true",
                    help="every operator at the fastest tensor rate: bounds that hold whatever "
                         "precision an implementation computes in")
    bd.set_defaults(func=_cmd_bounds)
    return p


def main(argv: list[str] | None = None) -> int:
    # Reports use ≤ ≥ ≈. A console renders them natively; piped output is UTF-8
    # rather than the platform's legacy code page.
    if not sys.stdout.isatty() and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    launch: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, launch = argv[:i], argv[i + 1:]
    args = parser.parse_args(argv)
    args.launch = launch
    if launch and args.cmd not in ("memory", "bounds"):
        parser.error(f"{args.cmd} does not take a launch command")
    if args.cmd == "hw" and not args.list and not args.name:
        parser.error("hw: give a spec name or --list")
    if args.cmd == "memory" and not args.list_models and not (args.model and args.hw):
        parser.error("memory: give --model and --hw")
    if args.cmd == "bounds" and not (args.model and args.hw):
        parser.error("bounds: give --model and --hw")
    try:
        return args.func(args)
    except SpecError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
