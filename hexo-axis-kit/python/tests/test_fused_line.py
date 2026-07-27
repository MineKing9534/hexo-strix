from __future__ import annotations

import pytest
import torch

from hexo_axis_models.fused_line import (
    axis_line_gather,
    axis_line_gather_reference,
    triton,
)
from hexo_axis_models.ops import RAY_DIRS


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
