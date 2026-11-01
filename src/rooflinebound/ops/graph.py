"""Operator graph of one forward pass, per GPU.

Each architecture component emits the operators one forward pass runs on one
GPU: FLOPs, the HBM bytes it must move, and the bytes it must receive over the
interconnect. Identical layers are grouped with a ``count``.

What is counted, and why it keeps bounds valid:

* **FLOPs**: the standard count, ``2 x M x N x K`` per matmul, plus attention
  scores and value mixing over exactly the (query, key) pairs attended.
  Small elementwise work (norms, activations, softmax, top-k) is omitted;
  leaving work out can only loosen a lower bound, never break it.
* **HBM bytes**: only compulsory traffic. Weights are read once per pass (a
  GPU's shard is far larger than on-chip cache); cached KV that attention or
  the indexer must look at is read once per sequence; new KV is written once.
  Activations are not counted: a perfectly fused kernel could keep them on chip.
* **Interconnect**: each TP all-reduce of ``N`` bytes requires every rank to
  receive at least ``(p - 1) / p x N``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

from ..arch.attention import GQA, KDA, MLA, DSAIndexer
from ..arch.ffn import DenseFFN, MoE
from ..arch.model import DecoderLayer, ModelArch
from ..arch.tensors import TensorSpec
from ..kvlayout import KVLayout, uniform
from ..parallel import Parallel
from ..spec.errors import SpecError

__all__ = ["Op", "Pass", "GraphBuilder", "build_pass", "causal_pairs", "index_pairs",
           "experts_touched"]

ACT_BYTES = 2  # activations exchanged between ranks are bf16
_COMPUTE_DTYPE = {"F8_E4M3": "fp8", "F8_E5M2": "fp8", "BF16": "bf16", "F16": "bf16",
                  "F32": "bf16"}

MlaMode = Literal["absorbed", "expanded"]


# ------------------------------------------------------------------ exact sums


def causal_pairs(start: int, n: int, cap: int | None = None) -> int:
    """Σ over queries at positions start..start+n-1 of the keys each attends.

    A query at position p attends p + 1 keys (itself included), or ``cap`` of
    them under top-k sparse attention.
    """
    a, b = start + 1, start + n  # attended counts run a..b
    if n <= 0:
        return 0
    if cap is None or b <= cap:
        return (a + b) * n // 2
    if a > cap:
        return cap * n
    return (a + cap) * (cap - a + 1) // 2 + cap * (b - cap)


def index_pairs(start: int, n: int, topk: int) -> int:
    """Σ of keys the DSA indexer scores: a query scores every earlier key, but
    only when there are more than ``topk`` of them (otherwise all are kept)."""
    lo, b = max(start + 1, topk + 1), start + n
    return 0 if lo > b else (lo + b) * (b - lo + 1) // 2


def experts_touched(n_experts: int, top_k: int, tokens: int, held: int,
                    ep: int) -> tuple[float, float, float]:
    """Experts one rank must read, as (expected, fewest, most).

    Each token picks ``top_k`` distinct experts. Under uniform independent
    routing an expert is missed by all ``tokens`` with probability
    ``(1 - k/E)^tokens``, so a rank holding ``held`` experts expects to touch
    ``held x (1 - (1 - k/E)^tokens)``. Whatever the routing, the busiest rank
    touches at least ``ceil(k / ep)`` (every token's k experts land somewhere)
    and at most ``min(held, k x tokens)``.
    """
    expected = held * (1 - (1 - top_k / n_experts) ** tokens)
    fewest = min(held, math.ceil(top_k / ep))
    most = min(held, top_k * tokens)
    if math.isclose(expected, round(expected), rel_tol=1e-12):
        expected = float(round(expected))  # e.g. exactly k for one token, not k + 1e-12
    return expected, fewest, max(most, expected)


# ---------------------------------------------------------------------- types


@dataclass(frozen=True)
class Op:
    """One operator family on one GPU; totals are ``count`` x the per-instance figures."""

    name: str
    category: str
    count: int
    flops: float
    dtype: str  # compute dtype: which peak rate applies
    weight_bytes: float = 0.0
    kv_read: float = 0.0
    kv_write: float = 0.0
    comm_bytes: float = 0.0  # bytes this GPU must receive
    # Routed experts: weight_bytes is every local expert; the fraction actually
    # read is (expected, fewest, most) over routings.
    read_fraction: tuple[float, float, float] | None = None

    def hbm_bytes(self, which: int = 0) -> float:
        """Per-instance compulsory HBM bytes; ``which`` picks expected/fewest/most."""
        w = self.weight_bytes * (self.read_fraction[which] if self.read_fraction else 1.0)
        return w + self.kv_read + self.kv_write


@dataclass(frozen=True)
class Pass:
    """One forward pass: ``seqs`` sequences, ``tokens`` new tokens each, starting
    at KV position ``start`` (the tokens already cached)."""

    seqs: int
    tokens: int
    start: int
    logits_tokens: int  # tokens that need logits (decode: all; prefill: the last)

    @property
    def total_tokens(self) -> int:
        return self.seqs * self.tokens


# ------------------------------------------------------------------- emitters


class GraphBuilder:
    """Builds operator lists for one model, layout and KV format.

    Reuse one builder across passes: weight bytes per module are cached.
    """

    def __init__(self, model: ModelArch, par: Parallel, kv_dtype: str = "bf16",
                 index_kv_dtype: str = "bf16", layout: KVLayout | None = None):
        if par.pp != 1:
            raise SpecError("latency and throughput bounds with PP > 1 are not modelled yet")
        self.model, self.par = model, par
        self.layout = layout or uniform(kv_dtype, index_kv_dtype)
        self.kv_dtype = self.layout.compute_dtype
        self.ikv_dtype = self.layout.index_compute_dtype
        self.mla_mode: MlaMode = "absorbed"
        self.by_name: dict[str, TensorSpec] = {t.name: t for t in model.inventory}
        # module stem ("model.layers.3.mlp.gate") -> its tensors (weight, scale, bias)
        self.by_module: dict[str, list[TensorSpec]] = {}
        for t in model.inventory:
            self.by_module.setdefault(t.name.rsplit(".", 1)[0], []).append(t)
        self._wcache: dict[str, float] = {}

    def build(self, p: Pass, *, mla_mode: MlaMode = "absorbed",
              layers: Literal["main", "mtp"] = "main") -> list[Op]:
        """Operators of one forward pass through the main model or one MTP draft step."""
        self.mla_mode = mla_mode
        ops = [self.embed(p)]
        if layers == "main":
            for layer, count in _group(self.model.layers):
                ops += self.layer(layer, p, count)
        else:
            for layer in self.model.mtp_layers:
                ops.append(self.mtp_glue(layer, p))
                ops += self.layer(layer, p, 1)
        if p.logits_tokens:
            ops.append(self.lm_head(p))
        return ops

    # --- helpers

    def _wbytes(self, prefix: str, module: str) -> float:
        key = prefix + module
        if key not in self._wcache:
            ts = self.by_module.get(key)
            if not ts:
                raise SpecError(f"no tensors for {key}")
            self._wcache[key] = float(sum(self.par.rank_bytes(t) for t in ts))
        return self._wcache[key]

    def _dtype(self, prefix: str, module: str) -> str:
        packed = self.by_name.get(f"{prefix}{module}.weight_packed")
        if packed is not None and packed.packed_fp4:
            return "fp4"  # storage dtype; weight-only fp4 is treated as fp4 compute
        w = self.by_name[f"{prefix}{module}.weight"]
        return _COMPUTE_DTYPE[w.dtype]

    def _allreduce(self, tokens: int) -> float:
        p = self.par.tp
        return 0.0 if p == 1 else (p - 1) / p * tokens * self.model.hidden * ACT_BYTES

    def _gemm(self, name, cat, prefix, module, tokens, out_f, in_f, count, *, split=1,
              fraction=1.0, comm=0.0) -> Op:
        """``split`` divides FLOPs (a TP-split matmul); ``fraction`` takes part of
        a stored tensor (e.g. one half of kv_b_proj)."""
        return Op(name, cat, count, 2.0 * tokens * out_f * in_f / split,
                  self._dtype(prefix, module),
                  weight_bytes=self._wbytes(prefix, module) * fraction, comm_bytes=comm)

    # --- components

    def mla(self, a: MLA, layer: DecoderLayer, p: Pass, count: int, cap: int | None) -> list[Op]:
        tp, T, pre = self.par.tp, p.total_tokens, layer.prefix
        hl = a.heads // tp
        kv_elems = a.kv_elements_per_token
        kv_tok = self.layout.main_bytes(a, tp)
        # Cached keys that must come from HBM for one sequence: every one of them for
        # dense attention; for top-k attention, at least what the last query attends
        # beyond the new tokens.
        last = p.start + p.tokens
        cached = p.start if cap is None else max(min(last, cap) - p.tokens, 0)
        pairs = p.seqs * causal_pairs(p.start, p.tokens, cap)
        kv_read = p.seqs * cached * kv_tok
        kv_write = T * kv_tok
        nope_share = a.qk_nope_head_dim / (a.qk_nope_head_dim + a.v_head_dim)
        ops = [
            self._gemm("attn.q_a_proj", "attn_proj", pre, "self_attn.q_a_proj", T,
                       a.q_lora_rank, a.hidden, count),
            self._gemm("attn.q_b_proj", "attn_proj", pre, "self_attn.q_b_proj", T,
                       a.heads * a.qk_head_dim, a.q_lora_rank, count, split=tp),
            self._gemm("attn.kv_a_proj", "attn_proj", pre, "self_attn.kv_a_proj_with_mqa", T,
                       kv_elems, a.hidden, count),
        ]
        if self.mla_mode == "absorbed":
            # q_nope folded into latent space by W_UK; attention over the latent; the
            # latent output lifted to v by W_UV. kv_b_proj holds both W_UK and W_UV.
            ops += [
                self._gemm("attn.absorb_q (W_UK)", "attn_proj", pre, "self_attn.kv_b_proj", T,
                           a.heads * a.kv_lora_rank, a.qk_nope_head_dim, count, split=tp,
                           fraction=nope_share),
                Op("attn.core (absorbed)", "attn_core", count,
                   2.0 * hl * (kv_elems + a.kv_lora_rank) * pairs, self.kv_dtype,
                   kv_read=kv_read, kv_write=kv_write),
                self._gemm("attn.absorb_o (W_UV)", "attn_proj", pre, "self_attn.kv_b_proj", T,
                           a.heads * a.v_head_dim, a.kv_lora_rank, count, split=tp,
                           fraction=1 - nope_share),
            ]
        else:
            # Expand cached latents to per-head keys and values, then plain attention.
            # Keys expanded: at least every key some query attends.
            keys = p.seqs * (last if cap is None else min(last, cap))
            ops += [
                self._gemm("attn.kv_b_proj (expand)", "attn_proj", pre, "self_attn.kv_b_proj",
                           keys, a.heads * (a.qk_nope_head_dim + a.v_head_dim),
                           a.kv_lora_rank, count, split=tp),
                Op("attn.core (expanded)", "attn_core", count,
                   2.0 * hl * (a.qk_head_dim + a.v_head_dim) * pairs, self.kv_dtype,
                   kv_read=kv_read, kv_write=kv_write),
            ]
        if a.output_gate:
            ops.append(self._gemm("attn.g_proj (output gate)", "attn_proj", pre,
                                  "self_attn.g_proj", T, a.heads * a.v_head_dim, a.hidden,
                                  count, split=tp))
        ops.append(self._gemm("attn.o_proj", "attn_proj", pre, "self_attn.o_proj", T,
                              a.hidden, a.heads * a.v_head_dim, count, split=tp,
                              comm=self._allreduce(T)))
        return ops

    def kda(self, a: KDA, layer: DecoderLayer, p: Pass, count: int) -> list[Op]:
        """Kimi Delta Attention: projections, short convolutions, the delta-rule
        recurrence, gated output norm, o_proj.

        The recurrence is counted at 4 x head_dim^2 FLOPs per token and head: the
        state read-out (q against the state) and the rank-one state update, which
        both the recurrent and the chunked algorithms perform. Each sequence's
        state is read and written once per pass (read only if it already exists).
        """
        tp, T, pre = self.par.tp, p.total_tokens, layer.prefix
        hl, d, proj = a.heads // tp, a.head_dim, a.proj
        state = self.layout.state_bytes(a, tp)
        small = sum(self._wbytes(pre, m) for m in ("self_attn", "self_attn.o_norm",
                                                   "self_attn.q_conv1d", "self_attn.k_conv1d",
                                                   "self_attn.v_conv1d")
                    if pre + m in self.by_module)
        ops = [
            Op("kda.qkv_proj", "attn_proj", count, 2.0 * T * 3 * proj * a.hidden / tp,
               self._dtype(pre, "self_attn.q_proj"),
               weight_bytes=sum(self._wbytes(pre, f"self_attn.{x}_proj") for x in "qkv")),
            self._gemm("kda.f_a_proj", "attn_proj", pre, "self_attn.f_a_proj", T, d, a.hidden,
                       count),
            self._gemm("kda.f_b_proj", "attn_proj", pre, "self_attn.f_b_proj", T, proj, d,
                       count, split=tp),
            self._gemm("kda.b_proj", "attn_proj", pre, "self_attn.b_proj", T, a.heads,
                       a.hidden, count, split=tp),
        ]
        if a.full_rank_gate:
            ops.append(self._gemm("kda.g_proj", "attn_proj", pre, "self_attn.g_proj", T, proj,
                                  a.hidden, count, split=tp))
        else:
            ops += [self._gemm("kda.g_a_proj", "attn_proj", pre, "self_attn.g_a_proj", T, d,
                               a.hidden, count),
                    self._gemm("kda.g_b_proj", "attn_proj", pre, "self_attn.g_b_proj", T, proj,
                               d, count, split=tp)]
        ops += [
            Op("kda.recurrence", "attn_core", count,
               4.0 * hl * d * d * T + 2.0 * a.conv_kernel * 3 * (proj // tp) * T, "bf16",
               weight_bytes=small,
               kv_read=p.seqs * state if p.start > 0 else 0.0, kv_write=p.seqs * state),
            self._gemm("kda.o_proj", "attn_proj", pre, "self_attn.o_proj", T, a.hidden, proj,
                       count, split=tp, comm=self._allreduce(T)),
        ]
        return ops

    def gqa(self, a: GQA, layer: DecoderLayer, p: Pass, count: int,
            cap: int | None) -> list[Op]:
        """Grouped-query attention: q/k/v projections, attention, o_proj.

        Each rank computes its query heads and the KV heads it holds (one, when
        TP exceeds the KV head count), reading those heads' cached keys and values.
        """
        tp, T, pre = self.par.tp, p.total_tokens, layer.prefix
        hl, kvl = a.heads // tp, a.kv_heads_per_rank(tp)
        kv_tok = self.layout.main_bytes(a, tp)
        last = p.start + p.tokens
        cached = p.start if cap is None else max(min(last, cap) - p.tokens, 0)
        pairs = p.seqs * causal_pairs(p.start, p.tokens, cap)
        return [
            self._gemm("attn.q_proj", "attn_proj", pre, "self_attn.q_proj", T,
                       a.heads * a.head_dim, a.hidden, count, split=tp),
            self._gemm("attn.k_proj", "attn_proj", pre, "self_attn.k_proj", T,
                       kvl * a.head_dim, a.hidden, count),
            self._gemm("attn.v_proj", "attn_proj", pre, "self_attn.v_proj", T,
                       kvl * a.head_dim, a.hidden, count),
            # scores and value mixing, head_dim each, per attended pair and query head
            Op("attn.core", "attn_core", count, 4.0 * hl * a.head_dim * pairs, self.kv_dtype,
               kv_read=p.seqs * cached * kv_tok, kv_write=T * kv_tok),
            self._gemm("attn.o_proj", "attn_proj", pre, "self_attn.o_proj", T,
                       a.hidden, a.heads * a.head_dim, count, split=tp,
                       comm=self._allreduce(T)),
        ]

    def indexer(self, ix: DSAIndexer, layer: DecoderLayer, p: Pass, count: int) -> list[Op]:
        T, pre = p.total_tokens, layer.prefix
        pairs = p.seqs * index_pairs(p.start, p.tokens, ix.topk)
        scores = pairs > 0
        return [
            self._gemm("indexer.wq_b", "indexer_proj", pre, "self_attn.indexer.wq_b", T,
                       ix.heads * ix.head_dim, ix.q_lora_rank, count),
            self._gemm("indexer.wk", "indexer_proj", pre, "self_attn.indexer.wk", T,
                       ix.head_dim, ix.hidden, count),
            self._gemm("indexer.weights_proj", "indexer_proj", pre,
                       "self_attn.indexer.weights_proj", T, ix.heads, ix.hidden, count),
            # Per scored pair: one dot product per head, then a weighted sum over heads.
            Op("indexer.score", "index_score", count, 2.0 * ix.heads * (ix.head_dim + 1) * pairs,
               self.ikv_dtype,
               kv_read=p.seqs * p.start * self.layout.index_key_bytes(ix) if scores else 0.0,
               kv_write=T * self.layout.index_key_bytes(ix)),
        ]

    def dense(self, f: DenseFFN, layer: DecoderLayer, p: Pass, count: int) -> list[Op]:
        T, tp, pre = p.total_tokens, self.par.tp, layer.prefix
        return [
            Op("ffn.gate_up", "dense_ffn", count, 2.0 * T * 2 * f.inter * f.hidden / tp,
               self._dtype(pre, "mlp.up_proj"),
               weight_bytes=self._wbytes(pre, "mlp.gate_proj") + self._wbytes(pre, "mlp.up_proj")),
            self._gemm("ffn.down", "dense_ffn", pre, "mlp.down_proj", T, f.hidden, f.inter,
                       count, split=tp, comm=self._allreduce(T)),
        ]

    def moe(self, m: MoE, layer: DecoderLayer, p: Pass, count: int) -> list[Op]:
        T, tp, ep, pre = p.total_tokens, self.par.tp, self.par.ep, layer.prefix
        held = m.n_experts // ep
        expected, fewest, most = experts_touched(m.n_experts, m.top_k, T, held, ep)
        mod, names = m.module, m.expert_names
        expert_bytes = sum(self._wbytes(pre, f"{mod}experts.{e}.{w}")
                           for e in range(m.n_experts) for w in names)
        expert_dtype = self._dtype(pre, f"{mod}experts.0.{names[1]}")
        ops = [self._gemm("moe.router", "router", pre, f"{mod}gate", T, m.n_experts, m.hidden,
                          count)]
        if m.latent:  # replicated projection into the experts' latent space and back
            ops.append(self._gemm("moe.latent_down", "moe_latent", pre,
                                  f"{mod}routed_expert_down_proj", T, m.latent, m.hidden, count))
        ops.append(
            # FLOPs: T x k assignments spread over ``ep`` ranks (or each split ``tp``
            # ways); the busiest rank does at least the average.
            Op("moe.routed_experts", "routed_expert", count,
               2.0 * 3 * T * m.top_k * m.expert_hidden * m.expert_inter / tp,
               expert_dtype, weight_bytes=expert_bytes,
               read_fraction=(expected / held, fewest / held, most / held)))
        if m.latent:
            up = self._gemm("moe.latent_up", "moe_latent", pre, f"{mod}routed_expert_up_proj",
                            T, m.hidden, m.latent, count)
            if pre + f"{mod}routed_expert_norm" in self.by_module:
                up = replace(up, weight_bytes=up.weight_bytes
                             + self._wbytes(pre, f"{mod}routed_expert_norm"))
            ops.append(up)
        if m.n_shared:
            si = m.expert_inter * m.n_shared
            shared_bytes = sum(self._wbytes(pre, f"{mod}shared_experts.{w}")
                               for w in ("gate_proj", "up_proj", "down_proj"))
            ops.append(Op("moe.shared_experts", "shared_expert", count,
                          2.0 * 3 * T * si * m.hidden / tp,
                          self._dtype(pre, f"{mod}shared_experts.up_proj"),
                          weight_bytes=shared_bytes))
        ops[-1] = replace(ops[-1], comm_bytes=self._allreduce(T))  # one reduce after the FFN
        return ops

    def norms(self, layer: DecoderLayer, count: int) -> Op:
        mods = ("input_layernorm", "post_attention_layernorm", "self_attn.q_a_layernorm",
                "self_attn.kv_a_layernorm", "self_attn.q_norm", "self_attn.k_norm",
                "self_attention_res_norm", "self_attention_res_proj", "mlp_res_norm",
                "mlp_res_proj")
        b = sum(self._wbytes(layer.prefix, m) for m in mods
                if layer.prefix + m in self.by_module)
        return Op("norms", "norm", count, 0.0, "bf16", weight_bytes=b)

    def layer(self, layer: DecoderLayer, p: Pass, count: int) -> list[Op]:
        cap = layer.attn_topk
        if isinstance(layer.attn, GQA):
            ops = self.gqa(layer.attn, layer, p, count, cap)
        elif isinstance(layer.attn, KDA):
            ops = self.kda(layer.attn, layer, p, count)
        else:
            ops = self.mla(layer.attn, layer, p, count, cap)
        if layer.indexer is not None:
            ops += self.indexer(layer.indexer, layer, p, count)
        if isinstance(layer.ffn, MoE):
            ops += self.moe(layer.ffn, layer, p, count)
        else:
            ops += self.dense(layer.ffn, layer, p, count)
        ops.append(self.norms(layer, count))
        return [replace(o, name=f"{_layer_kind(layer)}.{o.name}") for o in ops]

    def embed(self, p: Pass) -> Op:
        h, tp = self.model.hidden, self.par.tp
        return Op("embed", "embed", 1, 0.0, "bf16",
                  weight_bytes=p.total_tokens * h * ACT_BYTES / tp,
                  comm_bytes=self._allreduce(p.total_tokens))

    def lm_head(self, p: Pass) -> Op:
        tok = p.logits_tokens
        vloc = math.ceil(self.model.vocab / self.par.tp)
        r = self.model.root
        prefix, module = ((r, "lm_head") if not self.model.tie_embeddings
                          else (f"{r}model.", "embed_tokens"))
        # The final norm (and any output-side extras) run just before the head.
        extras = sum(self._wbytes(f"{r}model.", m) for m in
                     ("norm", "output_attn_res_norm", "output_attn_res_proj")
                     if f"{r}model.{m}" in self.by_module)
        return Op("lm_head", "lm_head", 1, 2.0 * tok * vloc * self.model.hidden,
                  self._dtype(prefix, module),
                  weight_bytes=self._wbytes(prefix, module) + extras)

    def mtp_glue(self, layer: DecoderLayer, p: Pass) -> Op:
        h, T = self.model.hidden, p.total_tokens
        pre = layer.prefix
        b = self._wbytes(pre, "eh_proj") + sum(self._wbytes(pre, m) for m in
                                                ("enorm", "hnorm", "shared_head.norm"))
        return Op("mtp.eh_proj", "mtp", 1, 2.0 * T * h * 2 * h, self._dtype(pre, "eh_proj"),
                  weight_bytes=b)


def _layer_kind(layer: DecoderLayer) -> str:
    if layer.is_mtp:
        return "mtp"
    ffn = "moe" if isinstance(layer.ffn, MoE) else "dense"
    return f"{ffn}+idx" if layer.indexer is not None else ffn


def _group(layers: tuple[DecoderLayer, ...]) -> list[tuple[DecoderLayer, int]]:
    """Group layers whose operators are identical; keep the first as representative."""
    groups: dict[tuple, list[DecoderLayer]] = {}
    for layer in layers:
        key = (type(layer.ffn), layer.ffn, layer.indexer is not None, layer.attn, layer.is_mtp,
               layer.attn_topk)
        groups.setdefault(key, []).append(layer)
    return [(ls[0], len(ls)) for ls in groups.values()]


def build_pass(model: ModelArch, par: Parallel, p: Pass, *, kv_dtype: str = "bf16",
               index_kv_dtype: str = "bf16", mla_mode: MlaMode = "absorbed",
               layers: Literal["main", "mtp"] = "main") -> list[Op]:
    """One-off convenience wrapper around :class:`GraphBuilder`."""
    return GraphBuilder(model, par, kv_dtype, index_kv_dtype).build(
        p, mla_mode=mla_mode, layers=layers)
