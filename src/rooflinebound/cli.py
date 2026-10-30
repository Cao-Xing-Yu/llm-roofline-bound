"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .quantity import Trust
from .spec import SpecError, list_hardware, load_hardware

# Display units for hardware fields: (unit label, scale from base units).
_HW_DISPLAY = {
    "mem_capacity": ("GB", 1e-9),
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
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "hw" and not args.list and not args.name:
        parser.error("hw: give a spec name or --list")
    try:
        return args.func(args)
    except SpecError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
