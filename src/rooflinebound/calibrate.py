"""Calibration: efficiency = bound / measurement, kept on the user's machine.

A measured TPOT or TTFT divided into its lower bound gives the fraction of the
roofline the deployment reaches. Estimates elsewhere are the bound divided by
that fraction; they are labelled ESTIMATE and name the conditions the fraction
was measured under. A measurement below its bound means the spec, layout or
inputs do not describe the deployment (or the measurement is off); it is
rejected, never saved.

Calibrations live in ``$ROOFLINEBOUND_CALIBRATIONS`` or the user config
directory, never in the repository.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .quantity import Quantity
from .spec.errors import SpecError

__all__ = ["Point", "calibration_path", "deployment_key", "efficiency", "find", "save"]


@dataclass(frozen=True)
class Point:
    kind: str  # "decode" or "prefill"
    efficiency: float
    measured_ms: float
    bound_ms: float
    batch: int | None = None  # decode
    context: int | None = None  # decode
    prompt: int | None = None  # prefill
    chunk: int | None = None  # prefill
    date: str = ""

    @property
    def conditions(self) -> str:
        if self.kind == "decode":
            return f"batch {self.batch}, context {self.context:,}"
        return f"{self.prompt:,}-token prompt" + (f", chunk {self.chunk:,}" if self.chunk else "")


def calibration_path() -> Path:
    env = os.environ.get("ROOFLINEBOUND_CALIBRATIONS")
    if env:
        return Path(env)
    base = os.environ.get("APPDATA") or os.environ.get("XDG_CONFIG_HOME") \
        or str(Path.home() / ".config")
    return Path(base) / "rooflinebound" / "calibrations.json"


def deployment_key(model: str, hw: str, layout: str, tp: int, ep: int) -> str:
    return f"{model} | {hw} | {layout} | TP{tp} EP{ep}"


def efficiency(bound_s: float, measured_s: float) -> float:
    if measured_s <= 0:
        raise SpecError("a measurement must be positive")
    if measured_s < bound_s:
        raise SpecError(
            f"measured {measured_s * 1e3:.4g} ms is below the lower bound {bound_s * 1e3:.4g} ms: "
            "the hardware spec, layout or inputs do not describe this deployment, or the "
            "measurement is wrong. Nothing was saved.")
    return bound_s / measured_s


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError as e:
        raise SpecError(f"{path}: not valid JSON ({e})") from None


def save(key: str, point: Point, path: Path | None = None) -> Path:
    """Add a point, replacing one taken under the same conditions."""
    path = path or calibration_path()
    data = _load(path)
    points = [p for p in data.get(key, [])
              if not (p["kind"] == point.kind and p.get("batch") == point.batch
                      and p.get("context") == point.context and p.get("prompt") == point.prompt
                      and p.get("chunk") == point.chunk)]
    stamped = Point(**{**asdict(point), "date": point.date or _dt.date.today().isoformat()})
    points.append(asdict(stamped))
    data[key] = points
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def find(key: str, kind: str, *, batch: int | None = None, context: int | None = None,
         prompt: int | None = None, path: Path | None = None) -> Point | None:
    """The stored point closest to the asked conditions (log distance), if any."""
    pts = [Point(**p) for p in _load(path or calibration_path()).get(key, [])
           if p["kind"] == kind]
    if not pts:
        return None

    def dist(p: Point) -> float:
        def ln(a, b):
            return abs(math.log(max(a, 1) / max(b, 1))) if a and b else 0.0
        if kind == "decode":
            return ln(p.batch, batch) + ln(p.context, context)
        return ln(p.prompt, prompt)
    return min(pts, key=dist)


def as_quantity(p: Point, key: str) -> Quantity:
    return Quantity.efficiency(p.efficiency, f"calib:{p.kind} ({p.conditions}, {p.date})",
                               source=key)


# item key -> (calibration kind, what the bound measures)
_ESTIMATED = {
    "tpot_b1": ("decode", "time"), "tps_seq_b1": ("decode", "rate"),
    "tpot_batch": ("decode", "time"), "tps_batch": ("decode", "rate"),
    "tps_max": ("decode", "rate"),
    "ttft": ("prefill", "time"), "prefill_tps": ("prefill", "rate"),
}


def estimates(rep, key: str, path: Path | None = None) -> None:
    """Add ESTIMATE items (bound scaled by a stored efficiency) to ``rep``."""
    from .memory import Item

    items = rep.items.setdefault("estimate", [])
    inputs = rep.inputs
    big = max((int(k.split("@")[1]) for k in rep.tables if k.startswith("decode@")), default=1)
    batch_of = {"tpot_b1": 1, "tps_seq_b1": 1, "tpot_batch": inputs.get("batch"),
                "tps_batch": inputs.get("batch"), "tps_max": big}
    found_any = {"decode": False, "prefill": False}
    for sec in ("decode", "prefill"):
        for it in rep.items.get(sec, []):
            if it.key not in _ESTIMATED:
                continue
            kind, what = _ESTIMATED[it.key]
            b, c, pr = batch_of.get(it.key), inputs.get("context"), inputs.get("prompt")
            p = find(key, kind, batch=b, context=c, prompt=pr, path=path)
            if p is None:
                continue
            found_any[kind] = True
            eff = as_quantity(p, key)
            q = it.q / eff if what == "time" else it.q * eff
            same = (p.batch == b and p.context == c) if kind == "decode" else p.prompt == pr
            note = f"bound ÷ efficiency {p.efficiency:.3f}, measured at {p.conditions}"
            if not same:
                here = (f"batch {b}, context {c:,}" if kind == "decode"
                        else f"{pr:,}-token prompt")
                note += f"; applied at {here}, where efficiency differs"
            items.append(Item(f"est_{it.key}", it.label, q, note))
    for kind, ok in found_any.items():
        wanted = (kind == "decode" and inputs.get("context") is not None) or \
                 (kind == "prefill" and inputs.get("prompt") is not None)
        if wanted and not ok:
            flag = "--measured-tpot-ms" if kind == "decode" else "--measured-ttft-ms"
            rep.missing.append(f"{kind} estimates: no calibration for this deployment; run "
                               f"rooflinebound calibrate ... {flag} ... --save")
