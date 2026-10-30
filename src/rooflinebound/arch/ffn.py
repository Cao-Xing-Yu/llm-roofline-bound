"""Feed-forward components: dense gated MLP and mixture of experts."""

from __future__ import annotations

from dataclasses import dataclass

from .tensors import Quantization, Shard, TensorSpec, linear, vector

__all__ = ["DenseFFN", "MoE"]


def _gated(prefix: str, base: str, hidden: int, inter: int, *, component: str, layer: int,
           quant: Quantization | None, dense_dtype: str, expert: bool = False
           ) -> list[TensorSpec]:
    """gate_proj / up_proj (column-parallel) and down_proj (row-parallel)."""
    kw = dict(component=component, layer=layer, quant=quant, dense_dtype=dense_dtype)

    def shard(s: Shard) -> dict:
        return {"shard": Shard.EXPERT, "expert_shard": s} if expert else {"shard": s}

    return [
        *linear(prefix, f"{base}gate_proj", inter, hidden, **shard(Shard.COL), **kw),
        *linear(prefix, f"{base}up_proj", inter, hidden, **shard(Shard.COL), **kw),
        *linear(prefix, f"{base}down_proj", hidden, inter, **shard(Shard.ROW), **kw),
    ]


@dataclass(frozen=True)
class DenseFFN:
    hidden: int
    inter: int

    def tensors(self, prefix: str, layer: int, quant: Quantization | None,
                dense_dtype: str) -> list[TensorSpec]:
        return _gated(prefix, "mlp.", self.hidden, self.inter, component="dense_ffn",
                      layer=layer, quant=quant, dense_dtype=dense_dtype)


@dataclass(frozen=True)
class MoE:
    """Routed experts plus always-on shared experts and a linear router.

    ``score_bias`` adds the fp32 ``e_score_correction_bias`` used by
    auxiliary-loss-free load balancing.
    """

    hidden: int
    n_experts: int
    top_k: int
    expert_inter: int
    n_shared: int
    score_bias: bool

    def tensors(self, prefix: str, layer: int, quant: Quantization | None,
                dense_dtype: str) -> list[TensorSpec]:
        kw = dict(layer=layer, quant=quant, dense_dtype=dense_dtype)
        out = [*linear(prefix, "mlp.gate", self.n_experts, self.hidden, component="router",
                       shard=Shard.REPLICATED, **kw)]
        if self.score_bias:
            out.append(vector(prefix, "mlp.gate", self.n_experts, component="router",
                              layer=layer, dtype="F32", suffix="e_score_correction_bias"))
        for e in range(self.n_experts):
            out += _gated(prefix, f"mlp.experts.{e}.", self.hidden, self.expert_inter,
                          component="routed_expert", expert=True, **kw)
        if self.n_shared:
            out += _gated(prefix, "mlp.shared_experts.", self.hidden,
                          self.expert_inter * self.n_shared, component="shared_expert", **kw)
        return out
