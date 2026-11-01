"""Vision encoders: weights that are resident but not used by text-only requests."""

from __future__ import annotations

from dataclasses import dataclass

from .tensors import Shard, TensorSpec

__all__ = ["MoonViT3d"]


@dataclass(frozen=True)
class MoonViT3d:
    """Kimi-K3's vision tower and patch-merger projector.

    SGLang runs the tower data-parallel (plain linears), so every tensor is
    replicated on each rank. Names follow the checkpoint.
    """

    hidden: int  # vt_hidden_size
    mlp: int  # vt_intermediate_size
    qkv_hidden: int  # qkv_hidden_size
    layers: int
    patch: int
    pos_h: int
    pos_w: int
    merge: tuple[int, int]
    text_hidden: int
    dtype: str = "BF16"
    in_channels: int = 3

    def tensors(self) -> list[TensorSpec]:
        h, d = self.hidden, self.dtype

        def t(name: str, shape: tuple[int, ...]) -> TensorSpec:
            return TensorSpec(name, shape, d, "vision", Shard.REPLICATED)

        out = [
            t("vision_tower.patch_embed.proj.weight", (h, self.in_channels, self.patch,
                                                       self.patch)),
            t("vision_tower.patch_embed.pos_emb.weight", (self.pos_h, self.pos_w, h)),
            t("vision_tower.encoder.final_layernorm.weight", (h,)),
        ]
        for i in range(self.layers):
            b = f"vision_tower.encoder.blocks.{i}."
            out += [
                t(f"{b}norm0.weight", (h,)),
                t(f"{b}norm1.weight", (h,)),
                t(f"{b}wqkv.weight", (3 * self.qkv_hidden, h)),
                t(f"{b}wo.weight", (h, self.qkv_hidden)),
                t(f"{b}mlp.fc0.weight", (self.mlp, h)),
                t(f"{b}mlp.fc1.weight", (h, self.mlp)),
            ]
        merged = h * self.merge[0] * self.merge[1]
        out += [
            t("mm_projector.proj.0.weight", (merged, merged)),
            t("mm_projector.proj.2.weight", (self.text_hidden, merged)),
            t("mm_projector.post_norm.weight", (self.text_hidden,)),
        ]
        return out
