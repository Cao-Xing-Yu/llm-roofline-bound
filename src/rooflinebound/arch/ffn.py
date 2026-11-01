"""Feed-forward components: dense gated MLP and mixture of experts."""

from __future__ import annotations

from dataclasses import dataclass

from .tensors import Quantization, Shard, TensorSpec, linear, vector

__all__ = ["DenseFFN", "MoE"]

GATE_UP_DOWN = ("gate_proj", "up_proj", "down_proj")


def _gated(prefix: str, base: str, hidden: int, inter: int, *, component: str, layer: int,
           quant: Quantization | None, dense_dtype: str, expert: bool = False,
           names: tuple[str, str, str] = GATE_UP_DOWN) -> list[TensorSpec]:
    """Gate and up projections (column-parallel), down projection (row-parallel).

    ``names`` are the checkpoint's names for (gate, up, down), e.g. Kimi's
    ``("w1", "w3", "w2")``.
    """
    kw = dict(component=component, layer=layer, quant=quant, dense_dtype=dense_dtype)

    def shard(s: Shard) -> dict:
        return {"shard": Shard.EXPERT, "expert_shard": s} if expert else {"shard": s}

    gate, up, down = names
    return [
        *linear(prefix, f"{base}{gate}", inter, hidden, **shard(Shard.COL), **kw),
        *linear(prefix, f"{base}{up}", inter, hidden, **shard(Shard.COL), **kw),
        *linear(prefix, f"{base}{down}", hidden, inter, **shard(Shard.ROW), **kw),
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
    auxiliary-loss-free load balancing. With ``latent``, routed experts work in
    a smaller space: a replicated projection takes tokens from ``hidden`` to
    ``latent`` before the experts and back after (optionally normalized).
    ``module`` and ``expert_names`` follow the checkpoint's naming.
    """

    hidden: int
    n_experts: int
    top_k: int
    expert_inter: int
    n_shared: int
    score_bias: bool
    latent: int | None = None
    latent_norm: bool = False
    module: str = "mlp."
    expert_names: tuple[str, str, str] = GATE_UP_DOWN

    @property
    def expert_hidden(self) -> int:
        return self.latent or self.hidden

    def tensors(self, prefix: str, layer: int, quant: Quantization | None,
                dense_dtype: str) -> list[TensorSpec]:
        kw = dict(layer=layer, quant=quant, dense_dtype=dense_dtype)
        m = self.module
        out = [*linear(prefix, f"{m}gate", self.n_experts, self.hidden, component="router",
                       shard=Shard.REPLICATED, **kw)]
        if self.score_bias:
            out.append(vector(prefix, f"{m}gate", self.n_experts, component="router",
                              layer=layer, dtype="F32", suffix="e_score_correction_bias"))
        for e in range(self.n_experts):
            out += _gated(prefix, f"{m}experts.{e}.", self.expert_hidden, self.expert_inter,
                          component="routed_expert", expert=True, names=self.expert_names,
                          **kw)
        if self.latent:
            lat = dict(component="moe_latent", shard=Shard.REPLICATED, **kw)
            out += [*linear(prefix, f"{m}routed_expert_down_proj", self.latent, self.hidden,
                            **lat),
                    *linear(prefix, f"{m}routed_expert_up_proj", self.hidden, self.latent,
                            **lat)]
            if self.latent_norm:
                out.append(vector(prefix, f"{m}routed_expert_norm", self.latent,
                                  component="moe_latent", layer=layer, dtype=dense_dtype))
        if self.n_shared:
            out += _gated(prefix, f"{m}shared_experts.", self.hidden,
                          self.expert_inter * self.n_shared, component="shared_expert", **kw)
        return out
