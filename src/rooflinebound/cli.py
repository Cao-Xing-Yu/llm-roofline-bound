"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .bounds import BoundsReport, PassBound, bounds_report
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
_DISPLAY = {"B": ("GB", 1e-9), "B/req": ("GB/req", 1e-9)}

_SECTIONS = {
    "model": "Model  (EXACT: arithmetic on the published config)",
    "per_gpu": "Per GPU  ({layout})",
    "bounds": "Bounds  (no assumptions)",
    "assumed": "Under your assumptions",
    "decode": "Decode at context {context}  (bounds; their assumptions are under Notes)",
    "mtp": "MTP decode  (tokens per step and TPOT rest on your acceptance rates)",
    "prefill": "Prefill  (bounds; their assumptions are under Notes)",
}


def _show(q: Quantity) -> str:
    unit, scale = _DISPLAY.get(str(q.unit), (None, 1))
    return (q.to(unit, scale) if unit else q).format()


def _render(r: MemoryReport | BoundsReport) -> None:
    i = r.inputs
    fmt = {"layout": f"TP{i['tp']} PP{i.get('pp', 1)} EP{i['ep']}",
           "context": f"{i['context']:,}" if i.get("context") is not None else "-"}
    print(f"{r.model} on {r.hardware}")
    print("Inputs: " + ", ".join(f"{k}={v}" for k, v in i.items() if v is not None))
    for sec, items in r.items.items():
        if not items:
            continue
        print(f"\n{_SECTIONS[sec].format(**fmt)}")
        for it in items:
            line = f"  {it.label:<48} {_show(it.q)}"
            print(line + (f"   -- {it.note}" if it.note and sec != "model" else ""))
    if r.missing:
        print("\nNot computed")
        for m in r.missing:
            print(f"  - {m}")
    if r.notes:
        print("\nNotes")
        for n in r.notes:
            print(f"  - {n}")


def _cmd_memory(args: argparse.Namespace) -> int:
    if args.list_models:
        for name in list_models():
            print(name)
        return 0
    model = load_model(args.model)
    hw = load_hardware(args.hw)
    dep = Deployment(Parallel(args.tp, args.pp, args.ep), args.mtp, args.kv_dtype,
                     args.index_kv_dtype, args.page_size)
    reserve = None
    if args.reserve_gb is not None:
        reserve = Quantity.assumed(args.reserve_gb * 1e9, "B", "user:--reserve-gb")
    r = memory_report(model, hw, dep, context=args.context, reserve=reserve)
    if args.json:
        json.dump(r.to_dict(), sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        _render(r)
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
    draft = args.mtp_draft or 0
    dep = Deployment(Parallel(args.tp, 1, args.ep), draft > 0, args.kv_dtype,
                     args.index_kv_dtype, args.page_size)
    reserve = None
    if args.reserve_gb is not None:
        reserve = Quantity.assumed(args.reserve_gb * 1e9, "B", "user:--reserve-gb")
    accept = None
    if args.mtp_accept:
        try:
            accept = [float(x) for x in args.mtp_accept.split(",")]
        except ValueError:
            raise SpecError("--mtp-accept takes comma-separated rates, e.g. 0.8,0.6") from None
    r = bounds_report(model, hw, dep, context=args.context, batch=args.batch,
                      prompt=args.prompt, chunk=args.chunk, draft=draft, accept=accept,
                      reserve=reserve, strict=args.strict)
    if args.json:
        json.dump(r.to_dict(), sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0
    _render(r)
    if args.ops:
        for title, b in r.tables.items():
            _render_ops(title, b, hw)
    return 0


def _deployment_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=None, help="bundled name or path to a config.json")
    p.add_argument("--hw", default=None, help="bundled name or path to a .toml spec")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--ep", type=int, default=1, help="1, or equal to --tp")
    p.add_argument("--kv-dtype", default="bf16", choices=["bf16", "fp16", "fp8"])
    p.add_argument("--index-kv-dtype", default="bf16", choices=["bf16", "fp16", "fp8"])
    p.add_argument("--page-size", type=int, default=1, help="KV allocation unit, tokens")
    p.add_argument("--reserve-gb", type=float,
                   help="ASSUMED runtime memory per GPU besides weights and KV")
    p.add_argument("--json", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rooflinebound",
        description="Provable lower bounds and clearly-labelled estimates for LLM inference.",
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
    mem.add_argument("--pp", type=int, default=1)
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
    args = parser.parse_args(argv)
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
