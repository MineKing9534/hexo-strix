"""Fused blocker-aware axis-line gather for the dense compatibility model."""

from __future__ import annotations

import torch
from torch import Tensor

from .ops import AXIS_RAY_PAIRS, RAY_DIRS, roll_source

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - the CPU reference remains usable.
    triton = None
    tl = None


def axis_line_gather_reference(
    x: Tensor,
    ray_mask: Tensor,
    distance_table: Tensor,
    axis_eps: Tensor,
    radius: int,
) -> Tensor:
    """Readable destination-gather implementation used on CPU and in tests."""

    axis_inputs: list[Tensor] = []
    for ray_a, ray_b in AXIS_RAY_PAIRS:
        aggregate = torch.zeros_like(x)
        for ray in (ray_a, ray_b):
            dq, dr = RAY_DIRS[ray]
            for distance in range(1, radius + 1):
                source = roll_source(x, dq * distance, dr * distance)
                edge = distance_table[distance - 1].view(1, -1, 1, 1)
                message = torch.relu(source + edge)
                mask = ray_mask[:, ray, distance - 1].unsqueeze(1).to(x.dtype)
                aggregate = aggregate + message * mask
        axis_inputs.append((1.0 + axis_eps) * x + aggregate)

    batch, channels, height, width = x.shape
    return torch.stack(axis_inputs, dim=1).reshape(
        batch * 3,
        channels,
        height,
        width,
    )


if triton is not None:

    @triton.jit
    def _ray_delta(ray: tl.constexpr):
        if ray == 0:
            return 1, 0
        if ray == 1:
            return 0, 1
        if ray == 2:
            return -1, 1
        if ray == 3:
            return -1, 0
        if ray == 4:
            return 0, -1
        return 1, -1


    @triton.jit
    def _axis_line_forward_axis_kernel(
        x_ptr,
        ray_mask_ptr,
        distance_ptr,
        eps_ptr,
        output_ptr,
        batch,
        channels,
        height,
        width,
        AXIS: tl.constexpr,
        RADIUS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        cells = height * width
        elements_per_batch = channels * cells
        total = batch * elements_per_batch
        valid_output = offsets < total

        within_batch = offsets % elements_per_batch
        batch_index = offsets // elements_per_batch
        channel = within_batch // cells
        cell = within_batch % cells
        row = cell // width
        column = cell % width

        x_value = tl.load(x_ptr + offsets, mask=valid_output, other=0.0).to(
            tl.float32
        )
        epsilon = tl.load(eps_ptr).to(tl.float32)
        result = (1.0 + epsilon) * x_value

        for side in tl.static_range(0, 2):
            ray = AXIS + side * 3
            dq, dr = _ray_delta(ray)
            for distance in tl.static_range(1, RADIUS + 1):
                source_row = row + dr * distance
                source_column = column + dq * distance
                valid_source = (
                    valid_output
                    & (source_row >= 0)
                    & (source_row < height)
                    & (source_column >= 0)
                    & (source_column < width)
                )
                mask_offset = (
                    (
                        (
                            (batch_index * 6 + ray) * RADIUS
                            + (distance - 1)
                        )
                        * height
                        + row
                    )
                    * width
                    + column
                )
                admitted = tl.load(
                    ray_mask_ptr + mask_offset,
                    mask=valid_output,
                    other=0,
                )
                source_offset = (
                    (batch_index * channels + channel) * height + source_row
                ) * width + source_column
                source = tl.load(
                    x_ptr + source_offset,
                    mask=valid_source & admitted,
                    other=0.0,
                ).to(tl.float32)
                edge = tl.load(
                    distance_ptr + (distance - 1) * channels + channel,
                    mask=valid_output,
                    other=0.0,
                ).to(tl.float32)
                message = tl.maximum(source + edge, 0.0)
                result += tl.where(valid_source & admitted, message, 0.0)

        output_offset = (
            (batch_index * 3 + AXIS) * elements_per_batch
            + within_batch
        )
        tl.store(output_ptr + output_offset, result, mask=valid_output)


    @triton.jit
    def _axis_line_grad_x_kernel(
        x_ptr,
        ray_mask_ptr,
        distance_ptr,
        eps_ptr,
        grad_output_ptr,
        grad_x_ptr,
        grad_distance_batch_ptr,
        grad_eps_batch_ptr,
        distance_bins,
        batch,
        channels,
        height,
        width,
        RADIUS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        batch_channel = tl.program_id(0)
        batch_index = batch_channel // channels
        channel = batch_channel % channels
        cell = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        cells = height * width
        valid_input = (batch_index < batch) & (cell < cells)
        row = cell // width
        column = cell % width
        x_offset = batch_channel * cells + cell

        x_value = tl.load(x_ptr + x_offset, mask=valid_input, other=0.0).to(
            tl.float32
        )
        epsilon = tl.load(eps_ptr).to(tl.float32)
        grad_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for axis in tl.static_range(0, 3):
            grad_offset = (
                ((batch_index * 3 + axis) * channels + channel) * height + row
            ) * width + column
            grad_sum += tl.load(
                grad_output_ptr + grad_offset,
                mask=valid_input,
                other=0.0,
            ).to(tl.float32)

        grad_x = (1.0 + epsilon) * grad_sum

        for distance in tl.static_range(1, RADIUS + 1):
            edge = tl.load(
                distance_ptr + (distance - 1) * channels + channel,
            ).to(tl.float32)
            grad_distance = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            for ray in tl.static_range(0, 6):
                dq, dr = _ray_delta(ray)
                axis = ray % 3
                destination_row = row - dr * distance
                destination_column = column - dq * distance
                valid_destination = (
                    valid_input
                    & (destination_row >= 0)
                    & (destination_row < height)
                    & (destination_column >= 0)
                    & (destination_column < width)
                )
                mask_offset = (
                    (
                        (
                            (batch_index * 6 + ray) * RADIUS
                            + (distance - 1)
                        )
                        * height
                        + destination_row
                    )
                    * width
                    + destination_column
                )
                admitted = tl.load(
                    ray_mask_ptr + mask_offset,
                    mask=valid_destination,
                    other=0,
                )
                grad_offset = (
                    (
                        (batch_index * 3 + axis) * channels + channel
                    )
                    * height
                    + destination_row
                ) * width + destination_column
                grad_message = tl.load(
                    grad_output_ptr + grad_offset,
                    mask=valid_destination & admitted,
                    other=0.0,
                ).to(tl.float32)
                contribution = tl.where(
                    valid_destination & admitted & (x_value + edge > 0.0),
                    grad_message,
                    0.0,
                )
                grad_x += contribution
                grad_distance += contribution

            distance_partial = tl.sum(grad_distance, axis=0)
            distance_batch_offset = (
                (batch_index * distance_bins + (distance - 1)) * channels
                + channel
            )
            tl.atomic_add(
                grad_distance_batch_ptr + distance_batch_offset,
                distance_partial,
                mask=batch_index < batch,
            )

        tl.store(grad_x_ptr + x_offset, grad_x, mask=valid_input)
        eps_partial = tl.sum(
            tl.where(valid_input, x_value * grad_sum, 0.0),
            axis=0,
        )
        eps_batch_offset = batch_index * channels + channel
        tl.atomic_add(
            grad_eps_batch_ptr + eps_batch_offset,
            eps_partial,
            mask=batch_index < batch,
        )


    @torch.library.custom_op(
        "hexo_axis::axis_line_gather_cuda",
        mutates_args=(),
    )
    def _axis_line_gather_cuda(
        x: Tensor,
        ray_mask: Tensor,
        distance_table: Tensor,
        axis_eps: Tensor,
        radius: int,
    ) -> Tensor:
        if not x.is_cuda:
            raise ValueError("axis_line_gather_cuda requires CUDA/ROCm tensors")
        if not (
            x.is_contiguous()
            and ray_mask.is_contiguous()
            and distance_table.is_contiguous()
        ):
            raise ValueError("axis_line_gather_cuda requires contiguous inputs")
        batch, channels, height, width = x.shape
        output_dtype = torch.promote_types(
            torch.promote_types(x.dtype, distance_table.dtype),
            axis_eps.dtype,
        )
        output = torch.empty(
            (batch * 3, channels, height, width),
            dtype=output_dtype,
            device=x.device,
        )
        grid = lambda meta: (
            triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),
        )
        for axis in range(3):
            _axis_line_forward_axis_kernel[grid](
                x,
                ray_mask,
                distance_table,
                axis_eps,
                output,
                batch,
                channels,
                height,
                width,
                AXIS=axis,
                RADIUS=radius,
                BLOCK_SIZE=128,
            )
        return output


    @_axis_line_gather_cuda.register_fake
    def _axis_line_gather_cuda_fake(
        x: Tensor,
        ray_mask: Tensor,
        distance_table: Tensor,
        axis_eps: Tensor,
        radius: int,
    ) -> Tensor:
        del ray_mask, radius
        batch, channels, height, width = x.shape
        output_dtype = torch.promote_types(
            torch.promote_types(x.dtype, distance_table.dtype),
            axis_eps.dtype,
        )
        return torch.empty(
            (batch * 3, channels, height, width),
            dtype=output_dtype,
            device=x.device,
        )


    @torch.library.custom_op(
        "hexo_axis::axis_line_gather_backward_cuda",
        mutates_args=(),
    )
    def _axis_line_gather_backward_cuda(
        x: Tensor,
        ray_mask: Tensor,
        distance_table: Tensor,
        axis_eps: Tensor,
        grad_output: Tensor,
        radius: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        grad_output = grad_output.contiguous()
        grad_x = torch.empty_like(x)
        grad_distance_batch = torch.zeros(
            (x.shape[0], distance_table.shape[0], x.shape[1]),
            dtype=torch.float32,
            device=x.device,
        )
        grad_eps_batch = torch.zeros(
            (x.shape[0], x.shape[1]),
            dtype=torch.float32,
            device=x.device,
        )
        batch, channels, height, width = x.shape

        x_grid = lambda meta: (
            batch * channels,
            triton.cdiv(height * width, meta["BLOCK_SIZE"]),
        )
        _axis_line_grad_x_kernel[x_grid](
            x,
            ray_mask,
            distance_table,
            axis_eps,
            grad_output,
            grad_x,
            grad_distance_batch,
            grad_eps_batch,
            distance_table.shape[0],
            batch,
            channels,
            height,
            width,
            RADIUS=radius,
            BLOCK_SIZE=256,
        )
        grad_distance_float = grad_distance_batch.sum(dim=0)
        grad_eps = grad_eps_batch.sum().reshape_as(axis_eps)
        return (
            grad_x,
            grad_distance_float.to(distance_table.dtype),
            grad_eps.to(axis_eps.dtype),
        )


    @_axis_line_gather_backward_cuda.register_fake
    def _axis_line_gather_backward_cuda_fake(
        x: Tensor,
        ray_mask: Tensor,
        distance_table: Tensor,
        axis_eps: Tensor,
        grad_output: Tensor,
        radius: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del ray_mask, grad_output, radius
        return (
            torch.empty_like(x),
            torch.empty_like(distance_table),
            torch.empty_like(axis_eps),
        )


    def _axis_line_gather_setup_context(ctx, inputs, output) -> None:
        del output
        x, ray_mask, distance_table, axis_eps, radius = inputs
        ctx.save_for_backward(x, ray_mask, distance_table, axis_eps)
        ctx.radius = radius


    def _axis_line_gather_backward(ctx, grad_output):
        x, ray_mask, distance_table, axis_eps = ctx.saved_tensors
        grad_x, grad_distance, grad_eps = _axis_line_gather_backward_cuda(
            x,
            ray_mask,
            distance_table,
            axis_eps,
            grad_output,
            ctx.radius,
        )
        return grad_x, None, grad_distance, grad_eps, None


    _axis_line_gather_cuda.register_autograd(
        _axis_line_gather_backward,
        setup_context=_axis_line_gather_setup_context,
    )


def axis_line_gather(
    x: Tensor,
    ray_mask: Tensor,
    distance_table: Tensor,
    axis_eps: Tensor,
    radius: int,
) -> Tensor:
    """Dispatch to the fused CUDA/ROCm op or the portable reference path."""

    if ray_mask.ndim != 5 or ray_mask.shape[2] < radius:
        raise ValueError(
            "ray_mask must be [B,6,R,H,W] with R at least radius"
        )
    if distance_table.ndim != 2 or distance_table.shape[0] < radius:
        raise ValueError(
            "distance_table must be [D,C] with D at least radius"
        )
    if x.is_cuda and triton is not None:
        return _axis_line_gather_cuda(
            x.contiguous(),
            ray_mask[:, :, :radius].contiguous(),
            distance_table.contiguous(),
            axis_eps,
            radius,
        )
    return axis_line_gather_reference(
        x,
        ray_mask,
        distance_table,
        axis_eps,
        radius,
    )
