"""Accelerator specifications.

A spec file is TOML. Every field carries its unit, how far its source can be
trusted, and (unless the user typed it in) where it was read from. A field
without a source is rejected rather than silently accepted.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

from ..quantity import Provenance, Quantity, Trust

__all__ = ["HardwareSpec", "SpecError", "load_hardware", "list_hardware", "REQUIRED_FIELDS"]


class SpecError(ValueError):
    """A spec file is incomplete, mis-typed or unsourced."""


# field -> (unit written in the file, scale to base unit, base unit)
REQUIRED_FIELDS: dict[str, tuple] = {
    "mem_capacity": ("GB", 1e9, "B"),
    "mem_bw": ("GB/s", 1e9, "B/s"),
    "tflops_bf16": ("TFLOP/s", 1e12, "FLOP/s"),
    "link_bw": ("GB/s", 1e9, "B/s"),
    "devices_per_node": ("device", 1, "device"),
}
OPTIONAL_FIELDS: dict[str, tuple] = {
    "tflops_fp8": ("TFLOP/s", 1e12, "FLOP/s"),
    "tflops_fp4": ("TFLOP/s", 1e12, "FLOP/s"),
}


@dataclass(frozen=True)
class HardwareField:
    quantity: Quantity  # in base units
    note: str | None
    disputed: bool


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    display: str
    fields: Mapping[str, HardwareField]
    path: str

    def __getitem__(self, key: str) -> Quantity:
        try:
            return self.fields[key].quantity
        except KeyError:
            raise KeyError(f"{self.name} has no '{key}' in its spec") from None

    def has(self, key: str) -> bool:
        return key in self.fields

    def peak_flops(self, dtype: str) -> Quantity:
        """Dense peak for a compute dtype, falling back fp4 -> fp8 -> bf16.

        Falling back is exact, not an estimate: a GEMM that cannot use a unit
        runs on the next wider one. Callers report which rate was used.
        """
        chain = {"fp4": ["fp4", "fp8", "bf16"], "fp8": ["fp8", "bf16"], "bf16": ["bf16"]}
        if dtype not in chain:
            raise SpecError(f"unknown compute dtype '{dtype}'")
        for d in chain[dtype]:
            if self.has(f"tflops_{d}"):
                return self[f"tflops_{d}"]
        raise SpecError(f"{self.name} has no tensor rate for {dtype} or wider")

    @property
    def disputed_fields(self) -> list[str]:
        return [k for k, f in self.fields.items() if f.disputed]

    @property
    def weakest_trust(self) -> Trust:
        return max(f.quantity.trust or Trust.USER for f in self.fields.values())


def _parse_field(spec_name: str, key: str, raw: Mapping, expected: tuple) -> HardwareField:
    unit, scale, base = expected
    where = f"{spec_name}.fields.{key}"
    if raw.get("unit") != unit:
        raise SpecError(f"{where}: unit must be '{unit}', got {raw.get('unit')!r}")
    value = raw.get("value")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise SpecError(f"{where}: value must be a positive number, got {value!r}")
    try:
        trust = Trust[raw.get("trust", "")]
    except KeyError:
        names = ", ".join(t.name for t in (Trust.OFFICIAL, Trust.THIRD_PARTY, Trust.USER))
        raise SpecError(f"{where}: trust must be one of {names}") from None
    if trust in (Trust.DEFINITION, Trust.PUBLIC_CONFIG):
        raise SpecError(f"{where}: trust {trust.name} does not apply to hardware")
    source = raw.get("source")
    if trust != Trust.USER and not (isinstance(source, str) and source.startswith("https://")):
        raise SpecError(f"{where}: a {trust.name} value needs an https:// source")
    prov = Provenance(f"hw:{spec_name}#{key}", trust, source)
    q = Quantity.exact(value * scale, base, prov)
    return HardwareField(q, raw.get("note"), bool(raw.get("disputed", False)))


def parse_hardware(data: Mapping, path: str = "<memory>") -> HardwareSpec:
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise SpecError(f"{path}: missing 'name'")
    raw_fields = data.get("fields", {})
    unknown = set(raw_fields) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS)
    if unknown:
        raise SpecError(f"{name}: unknown fields {sorted(unknown)}")
    missing = set(REQUIRED_FIELDS) - set(raw_fields)
    if missing:
        raise SpecError(f"{name}: missing required fields {sorted(missing)}")
    fields = {
        k: _parse_field(name, k, raw_fields[k], {**REQUIRED_FIELDS, **OPTIONAL_FIELDS}[k])
        for k in raw_fields
    }
    return HardwareSpec(name, data.get("display", name), fields, path)


def _bundled() -> dict[str, Traversable]:
    root = resources.files(__package__) / "data"
    out = {}
    for f in root.iterdir():
        if f.name.endswith(".toml"):
            data = tomllib.loads(f.read_text("utf-8"))
            if "fields" in data:  # hardware files have fields; model files do not
                out[data["name"]] = f
    return out


def list_hardware() -> list[str]:
    return sorted(_bundled())


def load_hardware(name_or_path: str | Path) -> HardwareSpec:
    """Load a bundled spec by name, or a user spec from a .toml path."""
    p = Path(name_or_path)
    if p.suffix == ".toml" and p.exists():
        return parse_hardware(tomllib.loads(p.read_text("utf-8")), str(p))
    bundled = _bundled()
    if str(name_or_path) not in bundled:
        known = ", ".join(sorted(bundled)) or "none"
        raise SpecError(f"unknown hardware '{name_or_path}' (bundled: {known})")
    f = bundled[str(name_or_path)]
    return parse_hardware(tomllib.loads(f.read_text("utf-8")), f"bundled:{f.name}")
