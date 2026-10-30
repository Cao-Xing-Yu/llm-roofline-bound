"""Hugging Face config parsers, dispatched on ``model_type``."""

from . import glm_moe_dsa

PARSERS = {t: m.parse for m in (glm_moe_dsa,) for t in m.MODEL_TYPES}

__all__ = ["PARSERS"]
