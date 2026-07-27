from __future__ import annotations

import torch

from hexo_axis_models.checkpoint import convert_strix_axis_state_dict
from hexo_axis_models.model import AxisGineCompatNet, AxisGineConfig


def test_selected_checkpoint_keys_convert_with_linear_to_conv_reshape():
    cfg = AxisGineConfig(
        hidden_dim=8,
        num_layers=1,
        line_radius=3,
        distance_bins=3,
        policy_hidden=6,
        q_hidden=5,
        value_hidden=4,
        value_bins=5,
        value_horizons=(),
        jk_mode="none",
    )
    model = AxisGineCompatNet(cfg)
    source = {
        "representation.input_proj.weight": torch.randn(8, 8),
        "representation.input_proj.bias": torch.randn(8),
        "representation.convs.0.dist_embed.weight": torch.randn(3, 8),
        "representation.convs.0.axis_conv.nn.0.weight": torch.randn(8, 8),
        "representation.convs.0.axis_conv.nn.0.bias": torch.randn(8),
        "representation.convs.0.axis_conv.nn.2.weight": torch.randn(8, 8),
        "representation.convs.0.axis_conv.nn.2.bias": torch.randn(8),
        "policy_head.mlp.0.weight": torch.randn(6, 8),
        "policy_head.mlp.0.bias": torch.randn(6),
        "policy_head.mlp.2.weight": torch.randn(1, 6),
        "policy_head.mlp.2.bias": torch.randn(1),
    }
    converted, report = convert_strix_axis_state_dict(source, model)
    assert not report.shape_mismatches
    torch.testing.assert_close(
        converted["blocks.0.axis_mlp.fc1.weight"],
        source["representation.convs.0.axis_conv.nn.0.weight"].unsqueeze(-1).unsqueeze(-1),
    )
    torch.testing.assert_close(
        converted["policy_head.fc2.weight"],
        source["policy_head.mlp.2.weight"].unsqueeze(-1).unsqueeze(-1),
    )
