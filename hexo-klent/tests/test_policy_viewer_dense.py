import dataclasses
import importlib.util
import math
from pathlib import Path

import pytest
import torch

from hexo_klent.config import AlgorithmConfig, KlentModelConfig
from hexo_klent.mcts_adapter import KlentMCTSAdapter
from hexo_klent.model import DenseAxisKlentNet, make_klent_net


def _load_policy_viewer():
    path = Path(__file__).parents[2] / "scripts" / "policy_viewer.py"
    spec = importlib.util.spec_from_file_location(
        "policy_viewer_dense_test",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/policy_viewer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_dense_config() -> KlentModelConfig:
    return KlentModelConfig(
        architecture="dense_axis",
        dense_ray_radius=5,
        hidden_dim=8,
        num_layers=1,
        num_heads=1,
        policy_hidden=8,
        q_hidden=4,
        pre_norm=True,
        dropout=0.0,
        graph_type="axis",
        prune_empty_edges=True,
        threat_features=True,
        relative_stone_encoding=True,
        axis_relational=True,
        axis_window=8,
        compact_stone_onehot=True,
        node_coords=False,
        moves_scope="node",
    )


def test_policy_viewer_bridges_dense_klent_checkpoint_everywhere(tmp_path):
    config = _tiny_dense_config()
    algorithm = AlgorithmConfig()
    network = make_klent_net(config)
    assert isinstance(network, DenseAxisKlentNet)
    with torch.no_grad():
        network.q_head.fc2.bias.fill_(math.atanh(0.25))

    checkpoint = tmp_path / "dense-klent.pt"
    torch.save(
        {
            "format": "hexo-klent-v1",
            "iteration": 7,
            "model_state_dict": network.state_dict(),
            "model_config": dataclasses.asdict(config),
            "config": {"algorithm": dataclasses.asdict(algorithm)},
        },
        checkpoint,
    )

    viewer = _load_policy_viewer()
    model, loaded_config = viewer._load_model(str(checkpoint))

    assert isinstance(model, KlentMCTSAdapter)
    assert loaded_config.architecture == "dense_axis"

    analysis = viewer._analyze(
        [[0, 0]],
        str(checkpoint),
        win_length=6,
        placement_radius=1,
        max_moves=12,
    )
    assert len(analysis["legal"]) == 6
    assert sum(analysis["probs"]) == pytest.approx(1.0)
    assert analysis["value"] == pytest.approx(0.25)
    assert analysis["node_info"]

    searched = viewer._analyze(
        [[0, 0]],
        str(checkpoint),
        win_length=6,
        placement_radius=1,
        max_moves=12,
        mcts_sims=4,
        mcts_m_actions=2,
    )
    assert len(searched["q_hat"]) == len(searched["legal"])
    assert searched["improved_policy"] is not None

    move = viewer._ai_move(
        [[0, 0]],
        str(checkpoint),
        win_length=6,
        placement_radius=1,
        max_moves=12,
    )
    assert move["move"] in analysis["legal"]
    assert move["value"] == pytest.approx(0.25)

    trajectory = viewer._analyze_trajectory(
        [[0, 0], [1, 0], [0, 1]],
        str(checkpoint),
        win_length=6,
        placement_radius=1,
        max_moves=12,
    )
    assert trajectory["evaluated_prefixes"] == 3
    assert [entry["value"] for entry in trajectory["trajectory"]] == (
        pytest.approx([0.25, 0.25, 0.25])
    )


def test_policy_viewer_resolves_klent_checkpoint_directory_and_final(tmp_path):
    viewer = _load_policy_viewer()
    output_dir = tmp_path / "dense-run"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "checkpoint_000005.pt").touch()
    (checkpoint_dir / "final.pt").touch()

    resolved = viewer._checkpoint_dir_from_config(
        {"run": {"output_dir": str(output_dir)}},
        None,
    )

    assert resolved == str(checkpoint_dir)
    assert viewer._list_checkpoints(resolved) == [
        str(checkpoint_dir / "checkpoint_000005.pt"),
        str(checkpoint_dir / "final.pt"),
    ]
    assert (
        viewer._checkpoint_dir_from_config(
            {"run": {"output_dir": "ignored"}},
            "explicit/checkpoints",
        )
        == "explicit/checkpoints"
    )
