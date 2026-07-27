from __future__ import annotations

import torch

from hexo_axis_models.model import (
    AxisGineCompatNet,
    AxisGineConfig,
    PersistentRayAxisNet,
    PersistentRayConfig,
    RayRingMixer,
)
from hexo_axis_models.ops import unpack_ray_bits


def synthetic_inputs(batch: int = 2, size: int = 9, radius: int = 3):
    planes = torch.zeros(batch, 8, size, size)
    center = size // 2
    planes[:, 0, center, center] = 1.0
    planes[:, 1, center, center + 1] = 1.0
    planes[:, 2, center - 1 : center + 2, center - 1 : center + 2] = 1.0
    planes[:, 2, center, center] = 0.0
    planes[:, 2, center, center + 1] = 0.0
    planes[:, 3] = planes[:, 2]
    active = (planes[:, 0:1] + planes[:, 1:2] + planes[:, 2:3]) > 0.5
    scalars = torch.tensor([[1.0, 1.0, 1.0, 0.8, -1.0]]).repeat(batch, 1)

    ray_mask = torch.zeros(batch, 6, radius, size, size, dtype=torch.bool)
    # A few valid sources in opposite directions through the centre.
    ray_mask[:, 0, 0, center, center] = True
    ray_mask[:, 3, 0, center, center + 1] = True
    ray_mask[:, 1, 0, center, center] = True
    return planes, scalars, active, ray_mask


def tiny_config(**kwargs) -> AxisGineConfig:
    base = dict(
        hidden_dim=16,
        num_layers=2,
        line_radius=3,
        distance_bins=3,
        policy_hidden=12,
        q_hidden=10,
        value_hidden=8,
        value_bins=5,
        value_horizons=(2, 4),
        jk_mode="cat",
    )
    base.update(kwargs)
    return AxisGineConfig(**base)


def test_unpack_ray_bits_layout():
    bits = torch.zeros(1, 2, 2, dtype=torch.int64)
    bits[0, 0, 0] = (1 << 0) | (1 << (3 * 5 + 1))
    mask = unpack_ray_bits(bits, radius=5)
    assert mask.shape == (1, 6, 5, 2, 2)
    assert bool(mask[0, 0, 0, 0, 0])
    assert bool(mask[0, 3, 1, 0, 0])
    assert int(mask.sum()) == 2


def test_axis_gine_shapes_masks_and_gradients():
    cfg = tiny_config()
    model = AxisGineCompatNet(cfg)
    planes, scalars, active, ray_mask = synthetic_inputs(radius=cfg.line_radius)
    output = model(planes, scalars, active, ray_mask)
    assert output.policy_logits.shape == (2, 9, 9)
    assert output.q_values.shape == (2, 9, 9)
    assert output.value.shape == (2,)
    assert output.value_logits.shape == (2, 5)

    legal = planes[:, 2].bool()
    assert torch.all(output.q_values[~legal] == 0)
    assert torch.all(output.policy_logits[~legal] == torch.finfo(output.policy_logits.dtype).min)

    loss = output.policy_logits[legal].mean() + output.q_values[legal].mean() + output.value.mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_forward_packed_matches_explicit_mask():
    cfg = tiny_config()
    model = AxisGineCompatNet(cfg).eval()
    planes, scalars, active, ray_mask = synthetic_inputs(radius=cfg.line_radius)
    bits = torch.zeros(planes.shape[0], planes.shape[-2], planes.shape[-1], dtype=torch.int64)
    for ray in range(6):
        for distance in range(cfg.line_radius):
            bits |= ray_mask[:, ray, distance].to(torch.int64) << (ray * 5 + distance)
    with torch.no_grad():
        a = model(planes, scalars, active, ray_mask)
        b = model.forward_packed(planes, scalars, active, bits)
    for xa, xb in zip(a, b):
        torch.testing.assert_close(xa, xb)


def test_active_compacted_axis_mlp_matches_padded_path():
    cfg = tiny_config()
    model = AxisGineCompatNet(cfg).eval()
    planes, scalars, active, ray_mask = synthetic_inputs(
        radius=cfg.line_radius
    )
    active_indices = active.reshape(-1).nonzero(
        as_tuple=False
    ).squeeze(1)
    with torch.no_grad():
        padded = model.forward_features(
            planes, scalars, active, ray_mask
        )[0]
        compact = model.forward_features(
            planes,
            scalars,
            active,
            ray_mask,
            active_indices,
        )[0]
    torch.testing.assert_close(compact, padded)


def test_active_compacted_axis_mlp_matches_padded_gradients():
    cfg = tiny_config()
    padded_model = AxisGineCompatNet(cfg)
    compact_model = AxisGineCompatNet(cfg)
    compact_model.load_state_dict(padded_model.state_dict())
    planes, scalars, active, ray_mask = synthetic_inputs(
        radius=cfg.line_radius
    )
    active_indices = active.reshape(-1).nonzero(
        as_tuple=False
    ).squeeze(1)
    weights = torch.randn(
        2,
        cfg.hidden_dim * cfg.num_layers,
        9,
        9,
    )

    padded = padded_model.forward_features(
        planes, scalars, active, ray_mask
    )[0]
    compact = compact_model.forward_features(
        planes,
        scalars,
        active,
        ray_mask,
        active_indices,
    )[0]
    (padded * weights).sum().backward()
    (compact * weights).sum().backward()

    for padded_parameter, compact_parameter in zip(
        padded_model.parameters(),
        compact_model.parameters(),
        strict=True,
    ):
        if padded_parameter.grad is None:
            assert compact_parameter.grad is None
        else:
            torch.testing.assert_close(
                compact_parameter.grad,
                padded_parameter.grad,
            )


def test_persistent_ray_exact_graft_starts_as_base_function():
    base_cfg = tiny_config()
    ray_cfg = PersistentRayConfig(
        hidden_dim=base_cfg.hidden_dim,
        num_layers=base_cfg.num_layers,
        line_radius=base_cfg.line_radius,
        distance_bins=base_cfg.distance_bins,
        policy_hidden=base_cfg.policy_hidden,
        q_hidden=base_cfg.q_hidden,
        value_hidden=base_cfg.value_hidden,
        value_bins=base_cfg.value_bins,
        value_horizons=base_cfg.value_horizons,
        jk_mode=base_cfg.jk_mode,
        ray_channels=6,
        ray_update_hidden=12,
        exact_graft_init=True,
    )
    base = AxisGineCompatNet(base_cfg).eval()
    ray = PersistentRayAxisNet(ray_cfg).eval()
    ray.load_state_dict(base.state_dict(), strict=False)
    planes, scalars, active, ray_mask = synthetic_inputs(radius=base_cfg.line_radius)
    with torch.no_grad():
        a = base(planes, scalars, active, ray_mask)
        b = ray(planes, scalars, active, ray_mask)
    for xa, xb in zip(a, b):
        torch.testing.assert_close(xa, xb, rtol=0.0, atol=0.0)


def test_ring_mixer_commutes_with_rotation_and_reflection():
    torch.manual_seed(7)
    mixer = RayRingMixer(4).eval()
    rays = torch.randn(2, 6, 4, 3, 3)

    rotation = torch.roll(rays, 2, dims=1)
    torch.testing.assert_close(mixer(rotation), torch.roll(mixer(rays), 2, dims=1))

    reflection_index = torch.tensor([0, 5, 4, 3, 2, 1])
    reflected = rays.index_select(1, reflection_index)
    torch.testing.assert_close(
        mixer(reflected), mixer(rays).index_select(1, reflection_index)
    )
