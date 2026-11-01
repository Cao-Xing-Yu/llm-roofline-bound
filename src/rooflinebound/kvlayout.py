"""How a KV cache stores one token: bytes per layer, and the dtype attention reads.

A layout is exact for the engine and kernel it names. The uniform layouts store
every cached element at one width with no scale metadata. They are the
smallest any real format can be, so figures computed with them are valid
bounds, but looser than the engine's real layout.
"""

from __future__ import annotations

from dataclasses import dataclass

from .arch.attention import GQA, KDA, MLA, DSAIndexer
from .spec.errors import SpecError

__all__ = ["KVLayout", "uniform", "sglang_dsa", "sglang_mha", "sglang_mla", "KV_DTYPE_BYTES"]

KV_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp8": 1}


@dataclass(frozen=True)
class KVLayout:
    name: str
    # cached elements: width, and optional block scales (MLA latent only)
    elem_bytes: int
    latent_scale_block: int | None  # elements per scale; None = no scales
    latent_scale_bytes: int
    rope_bytes: int  # MLA's decoupled rope key, often kept wider than the latent
    # DSA indexer key
    index_bytes: int
    index_scale_block: int | None
    index_scale_bytes: int
    # dtype attention and indexer scoring read, which sets their peak rate
    compute_dtype: str
    index_compute_dtype: str
    source: str | None = None  # where the layout is defined, for engine layouts
    # Linear-attention state per sequence: recurrent state and conv window widths
    # (SGLang defaults: fp32 state, bf16 conv; the HF reference also keeps fp32).
    ssm_bytes: int = 4
    conv_bytes: int = 2

    def main_bytes(self, attn: MLA | GQA, tp: int = 1) -> int:
        """Bytes per token per layer that one TP rank caches for attention.

        MLA: every rank keeps the whole latent and rope key (replicated).
        GQA: a key and a value per KV head the rank holds (split by head).
        """
        if isinstance(attn, GQA):
            return 2 * attn.kv_heads_per_rank(tp) * attn.head_dim * self.elem_bytes
        if isinstance(attn, KDA):
            return 0  # no per-token cache: a fixed state per sequence instead
        b = attn.kv_lora_rank * self.elem_bytes + attn.qk_rope_head_dim * self.rope_bytes
        if self.latent_scale_block:
            b += _blocks(attn.kv_lora_rank, self.latent_scale_block, "kv_lora_rank") \
                * self.latent_scale_bytes
        return b

    def state_bytes(self, attn: MLA | GQA | KDA, tp: int = 1) -> int:
        """Bytes per sequence per layer that one TP rank keeps whatever the length."""
        if isinstance(attn, KDA):
            return attn.state_bytes(tp, self.ssm_bytes, self.conv_bytes)
        return 0

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
_SGLANG_MHA_SRC = ("SGLang @ e3718813ec5f: mem_cache/memory_pool.py MHATokenToKVPool, "
                   "configs/model_config.py get_num_kv_heads")


def sglang_mla(kv_cache_dtype: str, ssm_bytes: int = 4, conv_bytes: int = 2) -> KVLayout:
    """SGLang's MLA pool without DSA: ``kv_lora_rank + rope`` elements per token and
    layer at the KV dtype, rope included (``calculate_mla_kv_cache_dim``). For
    hybrid models, the linear-attention state widths ride along."""
    widths = {"auto": ("bf16", 2), "bf16": ("bf16", 2), "fp8_e4m3": ("fp8", 1)}
    if kv_cache_dtype not in widths:
        raise SpecError(f"SGLang --kv-cache-dtype {kv_cache_dtype!r} is not modelled")
    d, w = widths[kv_cache_dtype]
    return KVLayout(f"sglang-mla {d}", w, None, 0, w, w, None, 0, d, d, source=_SGLANG_SRC,
                    ssm_bytes=ssm_bytes, conv_bytes=conv_bytes)


def sglang_mha(kv_cache_dtype: str) -> KVLayout:
    """SGLang's MHA/GQA pool: K and V buffers of ``kv heads per rank x head_dim``
    per token and layer, at the KV dtype (fp8 scales are per tensor, not per token)."""
    widths = {"auto": ("bf16", 2), "bf16": ("bf16", 2), "fp8_e4m3": ("fp8", 1)}
    if kv_cache_dtype not in widths:
        raise SpecError(f"SGLang --kv-cache-dtype {kv_cache_dtype!r} is not modelled")
    d, w = widths[kv_cache_dtype]
    return KVLayout(f"sglang-mha {d}", w, None, 0, w, w, None, 0, d, d, source=_SGLANG_MHA_SRC)


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
