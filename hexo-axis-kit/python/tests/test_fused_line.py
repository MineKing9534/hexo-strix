from __future__ import annotations

import pytest
import torch

from hexo_axis_models.fused_line import (
    axis_line_gather,
    axis_line_gather_compact,
    axis_line_gather_compact_reference,
    axis_line_gather_reference,
    directed_line_gather_compact,
    directed_line_gather_compact_reference,
    triton,
)
from hexo_axis_models.ops import (
    RAY_DIRS,
    pack_ray_mask,
    unpack_ray_bits,
)


def _compact_fixture(
    *,
    device: torch.device,
    channels: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch, size, radius = 2, 9, 3
    active_mask = (
        torch.rand(batch, size, size, device=device) < 0.55
    )
    active_indices = active_mask.reshape(-1).nonzero(
        as_tuple=False
    ).squeeze(1)
    active_lookup = torch.full(
        (batch * size * size,),
        -1,
        dtype=torch.int32,
        device=device,
    )
    active_lookup.index_copy_(
        0,
        active_indices,
        torch.arange(
            active_indices.numel(),
            dtype=torch.int32,
            device=device,
        ),
    )
    x_active = torch.randn(
        active_indices.numel(),
        channels,
        device=device,
    )
    distance = torch.randn(
        radius + 2,
        channels,
        device=device,
    )
    ray_mask = (
        torch.rand(
            batch,
            6,
            radius,
            size,
            size,
            device=device,
        )
        < 0.35
    )
    rows = torch.arange(size, device=device).view(size, 1)
    columns = torch.arange(size, device=device).view(1, size)
    for ray, (dq, dr) in enumerate(RAY_DIRS):
        for distance_index in range(1, radius + 1):
            source_active = torch.roll(
                active_mask,
                shifts=(-dr * distance_index, -dq * distance_index),
                dims=(-2, -1),
            )
            ray_mask[:, ray, distance_index - 1] &= (
                active_mask
                & source_active
                & (rows + dr * distance_index >= 0)
                & (rows + dr * distance_index < size)
                & (columns + dq * distance_index >= 0)
                & (columns + dq * distance_index < size)
            )
    return (
        x_active,
        distance,
        pack_ray_mask(ray_mask),
        active_indices,
        active_lookup,
        ray_mask,
    )


def test_packed_ray_mask_round_trips():
    torch.manual_seed(37)
    mask = torch.rand(2, 6, 4, 7, 7) < 0.4
    packed = pack_ray_mask(mask)
    assert packed.shape == (2, 7, 7)
    assert packed.dtype == torch.int64
    assert torch.equal(unpack_ray_bits(packed, 4), mask)


@pytest.mark.skipif(
    not torch.cuda.is_available() or triton is None,
    reason="fused line gather requires CUDA/ROCm and Triton",
)
def test_fused_line_gather_matches_reference_forward_and_backward():
    torch.manual_seed(19)
    device = torch.device("cuda")
    batch, channels, size, radius = 2, 4, 9, 3
    source_x = torch.randn(
        batch, channels, size, size, device=device
    )
    source_distance = torch.randn(
        radius + 2, channels, device=device
    )
    source_eps = torch.tensor([0.05], device=device)
    ray_mask = (
        torch.rand(
            batch, 6, radius, size, size, device=device
        )
        < 0.35
    )
    rows = torch.arange(size, device=device).view(size, 1)
    columns = torch.arange(size, device=device).view(1, size)
    for ray, (dq, dr) in enumerate(RAY_DIRS):
        for distance in range(1, radius + 1):
            ray_mask[:, ray, distance - 1] &= (
                (rows + dr * distance >= 0)
                & (rows + dr * distance < size)
                & (columns + dq * distance >= 0)
                & (columns + dq * distance < size)
            )
    output_gradient = torch.randn(
        batch * 3,
        channels,
        size,
        size,
        device=device,
    )

    results = []
    for implementation in (
        axis_line_gather_reference,
        axis_line_gather,
    ):
        x = source_x.detach().clone().requires_grad_()
        distance = (
            source_distance.detach().clone().requires_grad_()
        )
        eps = source_eps.detach().clone().requires_grad_()
        output = implementation(
            x, ray_mask, distance, eps, radius
        )
        (output * output_gradient).sum().backward()
        results.append(
            (output.detach(), x.grad, distance.grad, eps.grad)
        )

    reference, fused = results
    for reference_value, fused_value in zip(
        reference, fused, strict=True
    ):
        torch.testing.assert_close(
            fused_value,
            reference_value,
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or triton is None,
    reason="compact fused gathers require CUDA/ROCm and Triton",
)
@pytest.mark.parametrize("kind", ["axis", "directed"])
def test_compact_packed_gathers_match_reference_forward_and_backward(
    kind,
):
    torch.manual_seed(41)
    device = torch.device("cuda")
    channels, size, radius = 5, 9, 3
    (
        source_x,
        source_distance,
        ray_bits,
        active_indices,
        active_lookup,
        _ray_mask,
    ) = _compact_fixture(device=device, channels=channels)

    if kind == "axis":
        source_eps = torch.tensor([0.07], device=device)
        output_gradient = torch.randn(
            active_indices.numel(),
            3,
            channels,
            device=device,
        )
        implementations = (
            axis_line_gather_compact_reference,
            axis_line_gather_compact,
        )
    else:
        source_eps = None
        output_gradient = torch.randn(
            active_indices.numel(),
            6,
            channels,
            device=device,
        )
        implementations = (
            directed_line_gather_compact_reference,
            directed_line_gather_compact,
        )

    results = []
    for implementation in implementations:
        x = source_x.detach().clone().requires_grad_()
        distance = (
            source_distance.detach().clone().requires_grad_()
        )
        if kind == "axis":
            eps = source_eps.detach().clone().requires_grad_()
            output = implementation(
                x,
                ray_bits,
                distance,
                eps,
                active_indices,
                active_lookup,
                size,
                size,
                radius,
            )
            (output * output_gradient).sum().backward()
            results.append(
                (output.detach(), x.grad, distance.grad, eps.grad)
            )
        else:
            output = implementation(
                x,
                ray_bits,
                distance,
                active_indices,
                active_lookup,
                size,
                size,
                radius,
            )
            (output * output_gradient).sum().backward()
            results.append((output.detach(), x.grad, distance.grad))

    reference, fused = results
    for reference_value, fused_value in zip(
        reference,
        fused,
        strict=True,
    ):
        torch.testing.assert_close(
            fused_value,
            reference_value,
            rtol=1e-5,
            atol=1e-5,
        )
