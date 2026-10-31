"""Tensor, pipeline and expert parallel layout of one model replica.

The layout rules are stated here once and applied exactly:

* **TP** splits attention heads (``q_b_proj``, ``kv_b_proj`` by output,
  ``o_proj`` by input) and FFN intermediates (gate/up by output, down by
  input). The MLA and indexer down-projections, the router, norms and the MTP
  ``eh_proj`` are replicated. Embedding and ``lm_head`` are split by vocab row,
  padded up to a multiple of TP.
* **EP** is either 1 (routed experts split by TP like any FFN) or equal to TP
  (each rank holds ``n_experts / EP`` whole experts). Other EP sizes are
  rejected rather than approximated.
* **PP** gives each stage a contiguous run of decoder layers, as even as
  possible with earlier stages taking the remainder. The embedding sits on the
  first stage; final norm, ``lm_head`` and MTP layers on the last.

A split dimension must divide evenly by TP, and a quantized tensor's split
must fall on scale-block boundaries; otherwise the layout is rejected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from .arch.ffn import MoE
from .arch.model import ModelArch
from .arch.tensors import DTYPE_BYTES, Shard, TensorSpec
from .spec.errors import SpecError

__all__ = ["Parallel"]


@dataclass(frozen=True)
class Parallel:
    tp: int = 1
    pp: int = 1
    ep: int = 1

    @property
    def gpus(self) -> int:
        return self.tp * self.pp

    def __str__(self) -> str:
        return f"TP{self.tp} PP{self.pp} EP{self.ep}"

    # ---------------------------------------------------------------- checks

    def validate(self, model: ModelArch, *, mtp: bool) -> None:
        for name, v in (("tp", self.tp), ("pp", self.pp), ("ep", self.ep)):
            if not isinstance(v, int) or v < 1:
                raise SpecError(f"{name} must be a positive integer, got {v!r}")
        if self.ep not in (1, self.tp):
            raise SpecError(f"EP must be 1 or equal to TP ({self.tp}); EP={self.ep} is not "
                            "modelled")
        if self.pp > len(model.layers):
            raise SpecError(f"PP={self.pp} exceeds the {len(model.layers)} decoder layers")
        layers = model.all_layers if mtp else model.layers
        for layer in layers:
            if layer.attn.heads % self.tp:
                raise SpecError(f"{layer.attn.heads} attention heads do not split over "
                                f"TP={self.tp}")
            if isinstance(layer.ffn, MoE) and self.ep > 1 and layer.ffn.n_experts % self.ep:
                raise SpecError(f"{layer.ffn.n_experts} experts do not split over EP={self.ep}")
        for t, _ in model.tensor_groups(mtp=mtp):
            dim = self._split_dim(t)
            if dim is not None and t.shape[dim] % self.tp:
                kind = "scale blocks" if t.is_scale else "rows" if dim == 0 else "columns"
                raise SpecError(f"{t.name}: {t.shape[dim]} {kind} do not split over "
                                f"TP={self.tp}")

    def _split_dim(self, t: TensorSpec) -> int | None:
        shard = t.shard
        if shard is Shard.EXPERT:
            if self.ep > 1:
                return None
            shard = t.expert_shard
        return {Shard.COL: 0, Shard.ROW: 1}.get(shard)

    # ---------------------------------------------------------------- layout

    def stage_layers(self, model: ModelArch) -> list[list[int]]:
        """Main decoder layer indices on each pipeline stage."""
        n = len(model.layers)
        base, rem = divmod(n, self.pp)
        out, start = [], 0
        for s in range(self.pp):
            size = base + (1 if s < rem else 0)
            out.append([model.layers[i].index for i in range(start, start + size)])
            start += size
        return out

    def stage_of(self, model: ModelArch, t: TensorSpec, stages: list[list[int]]) -> int:
        if t.layer is None:
            return 0 if t.component == "embed" else self.pp - 1
        for s, ids in enumerate(stages):
            if t.layer in ids:
                return s
        return self.pp - 1  # MTP layers

    def rank_bytes(self, t: TensorSpec) -> Fraction:
        """Bytes of ``t`` held by one rank of its stage.

        A Fraction because under EP one rank holds whole experts: each expert
        tensor contributes 1/EP of itself on average, exactly.
        """
        if t.shard is Shard.REPLICATED:
            return Fraction(t.nbytes)
        if t.shard is Shard.VOCAB:
            rows = math.ceil(t.shape[0] / self.tp)
            return Fraction(rows * math.prod(t.shape[1:]) * DTYPE_BYTES[t.dtype])
        if t.shard is Shard.EXPERT and self.ep > 1:
            return Fraction(t.nbytes, self.ep)
        return Fraction(t.nbytes, self.tp)
