"""How a KV cache stores one token: bytes per layer, and the dtype attention reads.

A layout is exact for the engine and kernel it names. The uniform layouts store
every cached element at one width with no scale metadata. They are the
smallest any real format can be, so figures computed with them are valid
bounds, but looser than the engine's real layout.
"""

from __future__ import annotations

from dataclasses import dataclass

from .arch.attention import MLA, DSAIndexer
from .spec.errors import SpecError

__all__ = ["KVLayout", "uniform", "sglang_dsa", "KV_DTYPE_BYTES"]

KV_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp8": 1}


@dataclass(frozen=True)
class KVLayout:
    name: str
    # MLA latent (the kv_lora_rank part): element width and optional block scales
    latent_bytes: int
    latent_scale_block: int | None  # elements per scale; None = no scales
    latent_scale_bytes: int
    rope_bytes: int  # the decoupled rope key, often kept wider than the latent
    # DSA indexer key
    index_bytes: int
    index_scale_block: int | None
    index_scale_bytes: int
    # dtype attention and indexer scoring read, which sets their peak rate
    compute_dtype: str
    index_compute_dtype: str
    source: str | None = None  # where the layout is defined, for engine layouts

    def main_bytes(self, attn: MLA) -> int:
        """Bytes per token per layer for the MLA cache."""
        b = attn.kv_lora_rank * self.latent_bytes + attn.qk_rope_head_dim * self.rope_bytes
        if self.latent_scale_block:
            b += _blocks(attn.kv_lora_rank, self.latent_scale_block, "kv_lora_rank") \
                * self.latent_scale_bytes
        return b

    def index_key_bytes(self, ix: DSAIndexer) -> int:
        """Bytes per token per indexer layer for the DSA indexer key cache."""
        b = ix.head_dim * self.index_bytes
        if self.index_scale_block:
            b += _blocks(ix.head_dim, self.index_scale_block, "index_head_dim") \
                * self.index_scale_bytes
        return b


def _blocks(n: int, block: int, what: str) -> int:
    if n % block:
        raise SpecError(f"{what} {n} is not a multiple of the {block}-element scale block")
    return n // block


def uniform(kv_dtype: str = "bf16", index_kv_dtype: str = "bf16") -> KVLayout:
    for name, d in (("kv_dtype", kv_dtype), ("index_kv_dtype", index_kv_dtype)):
        if d not in KV_DTYPE_BYTES:
            raise SpecError(f"{name} must be one of {sorted(KV_DTYPE_BYTES)}, got {d!r}")
    kb, ib = KV_DTYPE_BYTES[kv_dtype], KV_DTYPE_BYTES[index_kv_dtype]
    return KVLayout(
        f"uniform {kv_dtype} / index {index_kv_dtype}", kb, None, 0, kb, ib, None, 0,
        "fp8" if kv_dtype == "fp8" else "bf16", "fp8" if index_kv_dtype == "fp8" else "bf16")


_SGLANG_SRC = ("SGLang @ e3718813ec5f: mem_cache/kv_cache_configurator.py "
               "calculate_mla_kv_cache_dim, mem_cache/index_key_cache.py")


def sglang_dsa(kv_cache_dtype: str, scaled: bool = True) -> KVLayout:
    """SGLang's DSA (DeepSeek sparse attention) pool on CUDA.

    The indexer key is always fp8 with one fp32 scale per 128 elements. With an
    fp8 KV cache and a scaled-layout backend (FlashMLA, Triton sparse MLA), the
    latent is fp8 with one fp32 scale per 128 elements and the rope key stays
    bf16. TRT-LLM, and TileLang on both sides, store raw fp8 rows instead.
    """
    index = dict(index_bytes=1, index_scale_block=128, index_scale_bytes=4,
                 index_compute_dtype="fp8")
    if kv_cache_dtype in ("bf16", "auto"):
        return KVLayout("sglang-dsa bf16", 2, None, 0, 2, compute_dtype="bf16",
                        source=_SGLANG_SRC, **index)
    if kv_cache_dtype != "fp8_e4m3":
        raise SpecError(f"SGLang --kv-cache-dtype {kv_cache_dtype!r} is not modelled")
    if scaled:
        return KVLayout("sglang-dsa fp8 (scaled)", 1, 128, 4, 2, compute_dtype="fp8",
                        source=_SGLANG_SRC, **index)
    return KVLayout("sglang-dsa fp8 (raw)", 1, None, 0, 1, compute_dtype="fp8",
                    source=_SGLANG_SRC, **index)
