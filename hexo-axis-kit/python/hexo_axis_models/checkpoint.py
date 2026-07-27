from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from .model import AxisGineCompatNet


@dataclass(frozen=True)
class ConversionReport:
    copied: tuple[str, ...]
    missing_in_source: tuple[str, ...]
    shape_mismatches: tuple[str, ...]
    unused_source: tuple[str, ...]


def extract_state_dict(checkpoint: Mapping[str, Any]) -> dict[str, Tensor]:
    raw: Any = checkpoint
    for key in ("model_state_dict", "model", "state_dict"):
        if isinstance(raw, Mapping) and key in raw and isinstance(raw[key], Mapping):
            raw = raw[key]
            break
    if not isinstance(raw, Mapping):
        raise TypeError("checkpoint does not contain a state dictionary")
    result: dict[str, Tensor] = {}
    for key, value in raw.items():
        if torch.is_tensor(value):
            result[str(key).replace("_orig_mod.", "")] = value.detach().cpu()
    return result


def _conv_weight(linear_weight: Tensor) -> Tensor:
    if linear_weight.ndim != 2:
        raise ValueError(f"expected a Linear weight, got shape {tuple(linear_weight.shape)}")
    return linear_weight.unsqueeze(-1).unsqueeze(-1)


def convert_strix_axis_state_dict(
    source: Mapping[str, Tensor],
    model: AxisGineCompatNet,
) -> tuple[dict[str, Tensor], ConversionReport]:
    """Convert current HeXONet axis-relational weights to the dense model.

    The conversion preserves learned parameters whose algebra has a direct
    counterpart. Persistent-ray modules, extra scalar conditioning, and any
    other newly introduced branches remain at their model initialisation.
    """
    src = {str(k).replace("_orig_mod.", ""): v.detach().cpu() for k, v in source.items()}
    target = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    copied: list[str] = []
    missing: list[str] = []
    mismatches: list[str] = []
    used_source: set[str] = set()

    def copy(source_key: str, target_key: str, *, as_conv: bool = False) -> None:
        if target_key not in target:
            return
        if source_key not in src:
            missing.append(source_key)
            return
        value = _conv_weight(src[source_key]) if as_conv else src[source_key]
        if tuple(value.shape) != tuple(target[target_key].shape):
            mismatches.append(
                f"{source_key} {tuple(value.shape)} -> {target_key} {tuple(target[target_key].shape)}"
            )
            return
        target[target_key] = value.to(dtype=target[target_key].dtype)
        copied.append(target_key)
        used_source.add(source_key)

    copy("representation.input_proj.weight", "input_proj.weight")
    copy("representation.input_proj.bias", "input_proj.bias")

    for i in range(model.config.num_layers):
        src_prefix = f"representation.convs.{i}."
        dst_prefix = f"blocks.{i}."
        copy(src_prefix + "dist_embed.weight", dst_prefix + "distance_embedding.weight")
        copy(src_prefix + "axis_conv.eps", dst_prefix + "axis_eps")
        copy(src_prefix + "axis_conv.lin.weight", dst_prefix + "axis_edge_proj.weight")
        copy(src_prefix + "axis_conv.lin.bias", dst_prefix + "axis_edge_proj.bias")
        copy(src_prefix + "axis_conv.nn.0.weight", dst_prefix + "axis_mlp.fc1.weight", as_conv=True)
        copy(src_prefix + "axis_conv.nn.0.bias", dst_prefix + "axis_mlp.fc1.bias")
        copy(src_prefix + "axis_conv.nn.2.weight", dst_prefix + "axis_mlp.fc2.weight", as_conv=True)
        copy(src_prefix + "axis_conv.nn.2.bias", dst_prefix + "axis_mlp.fc2.bias")

        copy(src_prefix + "global_edge_embed", dst_prefix + "global_edge_embed")
        copy(src_prefix + "global_conv.eps", dst_prefix + "global_eps")
        copy(src_prefix + "global_conv.lin.weight", dst_prefix + "global_edge_proj.weight")
        copy(src_prefix + "global_conv.lin.bias", dst_prefix + "global_edge_proj.bias")
        copy(src_prefix + "global_conv.nn.0.weight", dst_prefix + "global_mlp.fc1.weight", as_conv=True)
        copy(src_prefix + "global_conv.nn.0.bias", dst_prefix + "global_mlp.fc1.bias")
        copy(src_prefix + "global_conv.nn.2.weight", dst_prefix + "global_mlp.fc2.weight", as_conv=True)
        copy(src_prefix + "global_conv.nn.2.bias", dst_prefix + "global_mlp.fc2.bias")

        copy(src_prefix + "node_update.0.weight", dst_prefix + "node_update.fc1.weight", as_conv=True)
        copy(src_prefix + "node_update.0.bias", dst_prefix + "node_update.fc1.bias")
        copy(src_prefix + "node_update.2.weight", dst_prefix + "node_update.fc2.weight", as_conv=True)
        copy(src_prefix + "node_update.2.bias", dst_prefix + "node_update.fc2.bias")

        copy(f"representation.norms.{i}.weight", dst_prefix + "norm.weight")
        copy(f"representation.norms.{i}.bias", dst_prefix + "norm.bias")
        # Current production configs may not have LayerScale. In that case the
        # dense block's all-ones frozen parameter already reproduces no scaling.
        layer_scale_key = f"representation.layer_scales.{i}"
        if layer_scale_key in src:
            copy(layer_scale_key, dst_prefix + "layer_scale")

    copy("representation.final_norm.weight", "final_norm.weight")
    copy("representation.final_norm.bias", "final_norm.bias")
    if "representation.jk_weights" in src:
        copy("representation.jk_weights", "jk_weights")

    for head_name, source_prefix, target_prefix in (
        ("policy", "policy_head.mlp.", "policy_head."),
        ("q", "q_head.mlp.", "q_head."),
    ):
        del head_name
        copy(source_prefix + "0.weight", target_prefix + "fc1.weight", as_conv=True)
        copy(source_prefix + "0.bias", target_prefix + "fc1.bias")
        copy(source_prefix + "2.weight", target_prefix + "fc2.weight", as_conv=True)
        copy(source_prefix + "2.bias", target_prefix + "fc2.bias")

    copy("value_head.mlp.0.weight", "value_head.fc1.weight")
    copy("value_head.mlp.0.bias", "value_head.fc1.bias")
    copy("value_head.mlp.2.weight", "value_head.fc2.weight")
    copy("value_head.mlp.2.bias", "value_head.fc2.bias")

    for j in range(len(getattr(model, "horizon_value_heads", ()))):
        source_prefix = f"horizon_value_heads.{j}.mlp."
        target_prefix = f"horizon_value_heads.{j}."
        copy(source_prefix + "0.weight", target_prefix + "fc1.weight")
        copy(source_prefix + "0.bias", target_prefix + "fc1.bias")
        copy(source_prefix + "2.weight", target_prefix + "fc2.weight")
        copy(source_prefix + "2.bias", target_prefix + "fc2.bias")

    unused = tuple(sorted(set(src) - used_source))
    return target, ConversionReport(
        copied=tuple(copied),
        missing_in_source=tuple(sorted(set(missing))),
        shape_mismatches=tuple(mismatches),
        unused_source=unused,
    )


def load_strix_axis_checkpoint(
    model: AxisGineCompatNet,
    checkpoint: str | Path | Mapping[str, Any],
    *,
    map_location: str | torch.device = "cpu",
) -> ConversionReport:
    if isinstance(checkpoint, (str, Path)):
        loaded = torch.load(checkpoint, map_location=map_location, weights_only=True)
    else:
        loaded = checkpoint
    source = extract_state_dict(loaded)
    converted, report = convert_strix_axis_state_dict(source, model)
    model.load_state_dict(converted, strict=False)
    return report
