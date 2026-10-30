"""Model specs: a bundled name, or any local Hugging Face ``config.json``."""

from __future__ import annotations

import json
import tomllib
from importlib import resources
from pathlib import Path

from ..arch.model import ModelArch
from ..quantity import Provenance, Trust
from .errors import SpecError
from .parsers import PARSERS

__all__ = ["list_models", "load_model"]


def _models_dir():
    return resources.files(__package__) / "data" / "models"


def _bundled() -> dict[str, dict]:
    out = {}
    for f in _models_dir().iterdir():
        if f.name.endswith(".toml"):
            meta = tomllib.loads(f.read_text("utf-8"))
            out[meta["name"]] = meta
    return out


def list_models() -> list[str]:
    return sorted(_bundled())


def parse_config(cfg: dict, name: str, provenance: Provenance) -> ModelArch:
    model_type = cfg.get("model_type")
    if model_type not in PARSERS:
        known = ", ".join(sorted(PARSERS))
        raise SpecError(f"{name}: model_type {model_type!r} is not supported (supported: {known})")
    return PARSERS[model_type](cfg, name, provenance)


def load_model(name_or_path: str | Path) -> ModelArch:
    """Load a bundled model by name, or parse a local config.json.

    A local config is attributed to the user: this package cannot tell where
    the file came from.
    """
    p = Path(name_or_path)
    if p.suffix == ".json" and p.exists():
        cfg = json.loads(p.read_text("utf-8"))
        prov = Provenance(f"config:{p.name}", Trust.USER, None)
        return parse_config(cfg, p.stem, prov)
    bundled = _bundled()
    if str(name_or_path) not in bundled:
        known = ", ".join(sorted(bundled)) or "none"
        raise SpecError(f"unknown model '{name_or_path}' (bundled: {known}); "
                        "or pass a path to a config.json")
    meta = bundled[str(name_or_path)]
    cfg = json.loads((_models_dir() / meta["config"]).read_text("utf-8"))
    rev = meta["revision"]
    prov = Provenance(f"model:{meta['name']}@{rev[:7]}", Trust.PUBLIC_CONFIG,
                      f"https://huggingface.co/{meta['hf_repo']}/blob/{rev}/config.json")
    return parse_config(cfg, meta["name"], prov)
