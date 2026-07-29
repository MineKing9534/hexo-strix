"""Fused blocker-aware axis-line gather for the dense compatibility model."""

from __future__ import annotations

import torch
from torch import Tensor

from .ops import (
    AXIS_RAY_PAIRS,
    RAY_DIRS,
    roll_source,
    unpack_ray_bits,
)

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


def directed_line_gather_active_reference(
    x: Tensor,
    ray_mask: Tensor,
    distance_table: Tensor,
    active_flat_indices: Tensor,
    radius: int,
) -> Tensor:
    """Gather six directed ray messages only at active destination cells."""

    directions: list[Tensor] = []
    for ray, (dq, dr) in enumerate(RAY_DIRS):
        aggregate = torch.zeros_like(x)
        for distance in range(1, radius + 1):
            source = roll_source(x, dq * distance, dr * distance)
            edge = distance_table[distance - 1].view(1, -1, 1, 1)
            message = torch.nn.functional.silu(source + edge)
            mask = ray_mask[:, ray, distance - 1].unsqueeze(1).to(x.dtype)
            aggregate = aggregate + message * mask
        directions.append(aggregate)

    batch, channels, height, width = x.shape
    points = (
        torch.stack(directions, dim=1)
        .permute(0, 3, 4, 1, 2)
        .reshape(batch * height * width, 6, channels)
    )
    return points.index_select(0, active_flat_indices.to(torch.long))


def axis_line_gather_compact_reference(
    x_active: Tensor,
    ray_bits: Tensor,
    distance_table: Tensor,
    axis_eps: Tensor,
    active_flat_indices: Tensor,
    active_flat_lookup: Tensor | None,
    height: int,
    width: int,
    radius: int,
) -> Tensor:
    """Readable compact axis gather reconstructed through the dense reference."""

    del active_flat_lookup
    batch = ray_bits.shape[0]
    channels = x_active.shape[1]
    points = x_active.new_zeros((batch * height * width, channels))
    points.index_copy_(
        0,
        active_flat_indices.to(device=x_active.device, dtype=torch.long),
        x_active,
    )
    x = points.reshape(batch, height, width, channels).permute(
        0, 3, 1, 2
    )
    gathered = axis_line_gather_reference(
        x,
        unpack_ray_bits(ray_bits, radius),
        distance_table,
        axis_eps,
        radius,
    )
    gathered_points = gathered.reshape(
        batch, 3, channels, height, width
    ).permute(0, 3, 4, 1, 2).reshape(
        batch * height * width, 3, channels
    )
    return gathered_points.index_select(
        0,
        active_flat_indices.to(
            device=gathered_points.device,
            dtype=torch.long,
        ),
    )


def directed_line_gather_compact_reference(
    x_active: Tensor,
    ray_bits: Tensor,
    distance_table: Tensor,
    active_flat_indices: Tensor,
    active_flat_lookup: Tensor | None,
    height: int,
    width: int,
    radius: int,
) -> Tensor:
    """Readable compact directed gather reconstructed through the dense path."""

    del active_flat_lookup
    batch = ray_bits.shape[0]
    channels = x_active.shape[1]
    points = x_active.new_zeros((batch * height * width, channels))
    points.index_copy_(
        0,
        active_flat_indices.to(device=x_active.device, dtype=torch.long),
        x_active,
    )
    x = points.reshape(batch, height, width, channels).permute(
        0, 3, 1, 2
    )
    return directed_line_gather_active_reference(
        x,
        unpack_ray_bits(ray_bits, radius),
        distance_table,
        active_flat_indices,
        radius,
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
    def _ray_delta_dynamic(ray):
        dq = tl.where(
            (ray == 0) | (ray == 5),
            1,
            tl.where((ray == 2) | (ray == 3), -1, 0),
        )
        dr = tl.where(
            (ray == 1) | (ray == 2),
            1,
            tl.where((ray == 4) | (ray == 5), -1, 0),
        )
        return dq, dr


    @triton.jit
    def _axis_line_compact_forward_kernel(
        x_ptr,
        ray_bits_ptr,
        distance_ptr,
        eps_ptr,
        active_indices_ptr,
        active_lookup_ptr,
        output_ptr,
        active_count,
        channels,
        height,
        width,
        RADIUS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        total = active_count * 3 * channels
        valid_output = offsets < total
        channel = offsets % channels
        axis = (offsets // channels) % 3
        active_index = offsets // (3 * channels)
        cells = height * width
        flat_cell = tl.load(
            active_indices_ptr + active_index,
            mask=valid_output,
            other=0,
        )
        cell = flat_cell % cells
        row = cell // width
        column = cell % width
        word = tl.load(
            ray_bits_ptr + flat_cell,
            mask=valid_output,
            other=0,
        ).to(tl.int32)
        x_value = tl.load(
            x_ptr + active_index * channels + channel,
            mask=valid_output,
            other=0.0,
        ).to(tl.float32)
        epsilon = tl.load(eps_ptr).to(tl.float32)
        result = (1.0 + epsilon) * x_value

        for side in tl.static_range(0, 2):
            ray = axis + side * 3
            dq, dr = _ray_delta_dynamic(ray)
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
                admitted = (
                    (word >> (ray * 5 + (distance - 1))) & 1
                ) != 0
                source_flat = (
                    (flat_cell // cells) * cells
                    + source_row * width
                    + source_column
                )
                source_index = tl.load(
                    active_lookup_ptr + source_flat,
                    mask=valid_source & admitted,
                    other=-1,
                )
                source_active = source_index >= 0
                source = tl.load(
                    x_ptr + source_index * channels + channel,
                    mask=valid_source & admitted & source_active,
                    other=0.0,
                ).to(tl.float32)
                edge = tl.load(
                    distance_ptr
                    + (distance - 1) * channels
                    + channel,
                    mask=valid_output,
                    other=0.0,
                ).to(tl.float32)
                result += tl.where(
                    valid_source & admitted & source_active,
                    tl.maximum(source + edge, 0.0),
                    0.0,
                )

        tl.store(output_ptr + offsets, result, mask=valid_output)


    @triton.jit
    def _axis_line_compact_backward_kernel(
        x_ptr,
        ray_bits_ptr,
        distance_ptr,
        eps_ptr,
        active_indices_ptr,
        active_lookup_ptr,
        grad_output_ptr,
        grad_x_ptr,
        grad_distance_ptr,
        grad_eps_ptr,
        active_count,
        channels,
        height,
        width,
        RADIUS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        channel = tl.program_id(0)
        active_index = (
            tl.program_id(1) * BLOCK_SIZE
            + tl.arange(0, BLOCK_SIZE)
        )
        valid_source = active_index < active_count
        cells = height * width
        flat_cell = tl.load(
            active_indices_ptr + active_index,
            mask=valid_source,
            other=0,
        )
        cell = flat_cell % cells
        row = cell // width
        column = cell % width
        x_offset = active_index * channels + channel
        x_value = tl.load(
            x_ptr + x_offset,
            mask=valid_source,
            other=0.0,
        ).to(tl.float32)
        epsilon = tl.load(eps_ptr).to(tl.float32)
        grad_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for axis in tl.static_range(0, 3):
            grad_sum += tl.load(
                grad_output_ptr
                + (active_index * 3 + axis) * channels
                + channel,
                mask=valid_source,
                other=0.0,
            ).to(tl.float32)
        grad_x = (1.0 + epsilon) * grad_sum

        for distance in tl.static_range(1, RADIUS + 1):
            edge_offset = (distance - 1) * channels + channel
            edge = tl.load(distance_ptr + edge_offset).to(tl.float32)
            grad_distance = tl.zeros(
                (BLOCK_SIZE,),
                dtype=tl.float32,
            )
            for ray in tl.static_range(0, 6):
                dq, dr = _ray_delta(ray)
                destination_row = row - dr * distance
                destination_column = column - dq * distance
                valid_destination = (
                    valid_source
                    & (destination_row >= 0)
                    & (destination_row < height)
                    & (destination_column >= 0)
                    & (destination_column < width)
                )
                destination_flat = (
                    (flat_cell // cells) * cells
                    + destination_row * width
                    + destination_column
                )
                destination_index = tl.load(
                    active_lookup_ptr + destination_flat,
                    mask=valid_destination,
                    other=-1,
                )
                active_destination = (
                    valid_destination & (destination_index >= 0)
                )
                word = tl.load(
                    ray_bits_ptr + destination_flat,
                    mask=active_destination,
                    other=0,
                ).to(tl.int32)
                admitted = (
                    (word >> (ray * 5 + (distance - 1))) & 1
                ) != 0
                axis = ray % 3
                grad_message = tl.load(
                    grad_output_ptr
                    + (destination_index * 3 + axis) * channels
                    + channel,
                    mask=active_destination & admitted,
                    other=0.0,
                ).to(tl.float32)
                contribution = tl.where(
                    active_destination
                    & admitted
                    & (x_value + edge > 0.0),
                    grad_message,
                    0.0,
                )
                grad_x += contribution
                grad_distance += contribution

            distance_partial = tl.sum(
                grad_distance,
                axis=0,
            )
            tl.atomic_add(
                grad_distance_ptr + edge_offset,
                distance_partial,
                mask=channel < channels,
            )

        tl.store(
            grad_x_ptr + x_offset,
            grad_x,
            mask=valid_source,
        )
        eps_partial = tl.sum(
            tl.where(valid_source, x_value * grad_sum, 0.0),
            axis=0,
        )
        tl.atomic_add(
            grad_eps_ptr,
            eps_partial,
            mask=channel < channels,
        )


    @triton.jit
    def _directed_line_compact_forward_kernel(
        x_ptr,
        ray_bits_ptr,
        distance_ptr,
        active_indices_ptr,
        active_lookup_ptr,
        output_ptr,
        active_count,
        channels,
        height,
        width,
        RADIUS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        total = active_count * 6 * channels
        valid_output = offsets < total
        channel = offsets % channels
        ray = (offsets // channels) % 6
        active_index = offsets // (6 * channels)
        cells = height * width
        flat_cell = tl.load(
            active_indices_ptr + active_index,
            mask=valid_output,
            other=0,
        )
        cell = flat_cell % cells
        row = cell // width
        column = cell % width
        word = tl.load(
            ray_bits_ptr + flat_cell,
            mask=valid_output,
            other=0,
        ).to(tl.int32)
        dq, dr = _ray_delta_dynamic(ray)

        result = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
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
            admitted = (
                (word >> (ray * 5 + (distance - 1))) & 1
            ) != 0
            source_flat = (
                (flat_cell // cells) * cells
                + source_row * width
                + source_column
            )
            source_index = tl.load(
                active_lookup_ptr + source_flat,
                mask=valid_source & admitted,
                other=-1,
            )
            source_active = source_index >= 0
            source = tl.load(
                x_ptr + source_index * channels + channel,
                mask=valid_source & admitted & source_active,
                other=0.0,
            ).to(tl.float32)
            edge = tl.load(
                distance_ptr
                + (distance - 1) * channels
                + channel,
                mask=valid_output,
                other=0.0,
            ).to(tl.float32)
            z = source + edge
            result += tl.where(
                valid_source & admitted & source_active,
                z * tl.sigmoid(z),
                0.0,
            )

        tl.store(output_ptr + offsets, result, mask=valid_output)


    @triton.jit
    def _directed_line_compact_backward_kernel(
        x_ptr,
        ray_bits_ptr,
        distance_ptr,
        active_indices_ptr,
        active_lookup_ptr,
        grad_output_ptr,
        grad_x_ptr,
        grad_distance_ptr,
        active_count,
        channels,
        height,
        width,
        RADIUS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        channel = tl.program_id(0)
        active_index = (
            tl.program_id(1) * BLOCK_SIZE
            + tl.arange(0, BLOCK_SIZE)
        )
        valid_source = active_index < active_count
        cells = height * width
        flat_cell = tl.load(
            active_indices_ptr + active_index,
            mask=valid_source,
            other=0,
        )
        cell = flat_cell % cells
        row = cell // width
        column = cell % width
        x_offset = active_index * channels + channel
        x_value = tl.load(
            x_ptr + x_offset,
            mask=valid_source,
            other=0.0,
        ).to(tl.float32)
        grad_x = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        for distance in tl.static_range(1, RADIUS + 1):
            edge_offset = (distance - 1) * channels + channel
            edge = tl.load(distance_ptr + edge_offset).to(tl.float32)
            z = x_value + edge
            sigmoid = tl.sigmoid(z)
            derivative = sigmoid * (1.0 + z * (1.0 - sigmoid))
            grad_distance = tl.zeros(
                (BLOCK_SIZE,),
                dtype=tl.float32,
            )
            for ray in tl.static_range(0, 6):
                dq, dr = _ray_delta(ray)
                destination_row = row - dr * distance
                destination_column = column - dq * distance
                valid_destination = (
                    valid_source
                    & (destination_row >= 0)
                    & (destination_row < height)
                    & (destination_column >= 0)
                    & (destination_column < width)
                )
                destination_flat = (
                    (flat_cell // cells) * cells
                    + destination_row * width
                    + destination_column
                )
                destination_index = tl.load(
                    active_lookup_ptr + destination_flat,
                    mask=valid_destination,
                    other=-1,
                )
                active_destination = (
                    valid_destination & (destination_index >= 0)
                )
                word = tl.load(
                    ray_bits_ptr + destination_flat,
                    mask=active_destination,
                    other=0,
                ).to(tl.int32)
                admitted = (
                    (word >> (ray * 5 + (distance - 1))) & 1
                ) != 0
                grad_message = tl.load(
                    grad_output_ptr
                    + (destination_index * 6 + ray) * channels
                    + channel,
                    mask=active_destination & admitted,
                    other=0.0,
                ).to(tl.float32)
                contribution = tl.where(
                    active_destination & admitted,
                    grad_message * derivative,
                    0.0,
                )
                grad_x += contribution
                grad_distance += contribution

            distance_partial = tl.sum(
                grad_distance,
                axis=0,
            )
            tl.atomic_add(
                grad_distance_ptr + edge_offset,
                distance_partial,
                mask=channel < channels,
            )

        tl.store(
            grad_x_ptr + x_offset,
            grad_x,
            mask=valid_source,
        )


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


    @torch.library.custom_op(
        "hexo_axis::axis_line_gather_compact_cuda",
        mutates_args=(),
    )
    def _axis_line_gather_compact_cuda(
        x_active: Tensor,
        ray_bits: Tensor,
        distance_table: Tensor,
        axis_eps: Tensor,
        active_flat_indices: Tensor,
        active_flat_lookup: Tensor,
        height: int,
        width: int,
        radius: int,
    ) -> Tensor:
        if not x_active.is_cuda:
            raise ValueError(
                "axis_line_gather_compact_cuda requires CUDA/ROCm tensors"
            )
        if not all(
            tensor.is_contiguous()
            for tensor in (
                x_active,
                ray_bits,
                distance_table,
                active_flat_indices,
                active_flat_lookup,
            )
        ):
            raise ValueError(
                "axis_line_gather_compact_cuda requires contiguous inputs"
            )
        output_dtype = torch.promote_types(
            torch.promote_types(
                x_active.dtype,
                distance_table.dtype,
            ),
            axis_eps.dtype,
        )
        output = torch.empty(
            (x_active.shape[0], 3, x_active.shape[1]),
            dtype=output_dtype,
            device=x_active.device,
        )
        grid = lambda meta: (
            triton.cdiv(output.numel(), meta["BLOCK_SIZE"]),
        )
        _axis_line_compact_forward_kernel[grid](
            x_active,
            ray_bits,
            distance_table,
            axis_eps,
            active_flat_indices,
            active_flat_lookup,
            output,
            x_active.shape[0],
            x_active.shape[1],
            height,
            width,
            RADIUS=radius,
            BLOCK_SIZE=128,
        )
        return output


    @_axis_line_gather_compact_cuda.register_fake
    def _axis_line_gather_compact_cuda_fake(
        x_active: Tensor,
        ray_bits: Tensor,
        distance_table: Tensor,
        axis_eps: Tensor,
        active_flat_indices: Tensor,
        active_flat_lookup: Tensor,
        height: int,
        width: int,
        radius: int,
    ) -> Tensor:
        del (
            ray_bits,
            active_flat_indices,
            active_flat_lookup,
            height,
            width,
            radius,
        )
        output_dtype = torch.promote_types(
            torch.promote_types(
                x_active.dtype,
                distance_table.dtype,
            ),
            axis_eps.dtype,
        )
        return torch.empty(
            (x_active.shape[0], 3, x_active.shape[1]),
            dtype=output_dtype,
            device=x_active.device,
        )


    @torch.library.custom_op(
        "hexo_axis::axis_line_gather_compact_backward_cuda",
        mutates_args=(),
    )
    def _axis_line_gather_compact_backward_cuda(
        x_active: Tensor,
        ray_bits: Tensor,
        distance_table: Tensor,
        axis_eps: Tensor,
        active_flat_indices: Tensor,
        active_flat_lookup: Tensor,
        grad_output: Tensor,
        height: int,
        width: int,
        radius: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        grad_x = torch.empty_like(x_active)
        grad_distance_float = torch.zeros_like(
            distance_table,
            dtype=torch.float32,
        )
        grad_eps_float = torch.zeros(
            (),
            dtype=torch.float32,
            device=x_active.device,
        )
        grad_output = grad_output.contiguous()
        grid = lambda meta: (
            x_active.shape[1],
            triton.cdiv(x_active.shape[0], meta["BLOCK_SIZE"]),
        )
        _axis_line_compact_backward_kernel[grid](
            x_active,
            ray_bits,
            distance_table,
            axis_eps,
            active_flat_indices,
            active_flat_lookup,
            grad_output,
            grad_x,
            grad_distance_float,
            grad_eps_float,
            x_active.shape[0],
            x_active.shape[1],
            height,
            width,
            RADIUS=radius,
            BLOCK_SIZE=256,
        )
        return (
            grad_x,
            grad_distance_float.to(distance_table.dtype),
            grad_eps_float.reshape_as(axis_eps).to(axis_eps.dtype),
        )


    @_axis_line_gather_compact_backward_cuda.register_fake
    def _axis_line_gather_compact_backward_cuda_fake(
        x_active: Tensor,
        ray_bits: Tensor,
        distance_table: Tensor,
        axis_eps: Tensor,
        active_flat_indices: Tensor,
        active_flat_lookup: Tensor,
        grad_output: Tensor,
        height: int,
        width: int,
        radius: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del (
            ray_bits,
            active_flat_indices,
            active_flat_lookup,
            grad_output,
            height,
            width,
            radius,
        )
        return (
            torch.empty_like(x_active),
            torch.empty_like(distance_table),
            torch.empty_like(axis_eps),
        )


    def _axis_line_gather_compact_setup_context(
        ctx,
        inputs,
        output,
    ) -> None:
        del output
        (
            x_active,
            ray_bits,
            distance_table,
            axis_eps,
            active_flat_indices,
            active_flat_lookup,
            height,
            width,
            radius,
        ) = inputs
        ctx.save_for_backward(
            x_active,
            ray_bits,
            distance_table,
            axis_eps,
            active_flat_indices,
            active_flat_lookup,
        )
        ctx.height = height
        ctx.width = width
        ctx.radius = radius


    def _axis_line_gather_compact_backward(ctx, grad_output):
        (
            x_active,
            ray_bits,
            distance_table,
            axis_eps,
            active_flat_indices,
            active_flat_lookup,
        ) = ctx.saved_tensors
        grad_x, grad_distance, grad_eps = (
            _axis_line_gather_compact_backward_cuda(
                x_active,
                ray_bits,
                distance_table,
                axis_eps,
                active_flat_indices,
                active_flat_lookup,
                grad_output,
                ctx.height,
                ctx.width,
                ctx.radius,
            )
        )
        return (
            grad_x,
            None,
            grad_distance,
            grad_eps,
            None,
            None,
            None,
            None,
            None,
        )


    _axis_line_gather_compact_cuda.register_autograd(
        _axis_line_gather_compact_backward,
        setup_context=_axis_line_gather_compact_setup_context,
    )


    @torch.library.custom_op(
        "hexo_axis::directed_line_gather_compact_cuda",
        mutates_args=(),
    )
    def _directed_line_gather_compact_cuda(
        x_active: Tensor,
        ray_bits: Tensor,
        distance_table: Tensor,
        active_flat_indices: Tensor,
        active_flat_lookup: Tensor,
        height: int,
        width: int,
        radius: int,
    ) -> Tensor:
        if not x_active.is_cuda:
            raise ValueError(
                "directed_line_gather_compact_cuda requires CUDA/ROCm tensors"
            )
        if not all(
            tensor.is_contiguous()
            for tensor in (
                x_active,
                ray_bits,
                distance_table,
                active_flat_indices,
                active_flat_lookup,
            )
        ):
            raise ValueError(
                "directed_line_gather_compact_cuda requires contiguous inputs"
            )
        output = torch.empty(
            (x_active.shape[0], 6, x_active.shape[1]),
            dtype=torch.promote_types(
                x_active.dtype,
                distance_table.dtype,
            ),
            device=x_active.device,
        )
        grid = lambda meta: (
            triton.cdiv(output.numel(), meta["BLOCK_SIZE"]),
        )
        _directed_line_compact_forward_kernel[grid](
            x_active,
            ray_bits,
            distance_table,
            active_flat_indices,
            active_flat_lookup,
            output,
            x_active.shape[0],
            x_active.shape[1],
            height,
            width,
            RADIUS=radius,
            BLOCK_SIZE=128,
        )
        return output


    @_directed_line_gather_compact_cuda.register_fake
    def _directed_line_gather_compact_cuda_fake(
        x_active: Tensor,
        ray_bits: Tensor,
        distance_table: Tensor,
        active_flat_indices: Tensor,
        active_flat_lookup: Tensor,
        height: int,
        width: int,
        radius: int,
    ) -> Tensor:
        del (
            ray_bits,
            active_flat_indices,
            active_flat_lookup,
            height,
            width,
            radius,
        )
        return torch.empty(
            (x_active.shape[0], 6, x_active.shape[1]),
            dtype=torch.promote_types(
                x_active.dtype,
                distance_table.dtype,
            ),
            device=x_active.device,
        )


    @torch.library.custom_op(
        "hexo_axis::directed_line_gather_compact_backward_cuda",
        mutates_args=(),
    )
    def _directed_line_gather_compact_backward_cuda(
        x_active: Tensor,
        ray_bits: Tensor,
        distance_table: Tensor,
        active_flat_indices: Tensor,
        active_flat_lookup: Tensor,
        grad_output: Tensor,
        height: int,
        width: int,
        radius: int,
    ) -> tuple[Tensor, Tensor]:
        grad_x = torch.empty_like(x_active)
        grad_distance_float = torch.zeros_like(
            distance_table,
            dtype=torch.float32,
        )
        grad_output = grad_output.contiguous()
        grid = lambda meta: (
            x_active.shape[1],
            triton.cdiv(x_active.shape[0], meta["BLOCK_SIZE"]),
        )
        _directed_line_compact_backward_kernel[grid](
            x_active,
            ray_bits,
            distance_table,
            active_flat_indices,
            active_flat_lookup,
            grad_output,
            grad_x,
            grad_distance_float,
            x_active.shape[0],
            x_active.shape[1],
            height,
            width,
            RADIUS=radius,
            BLOCK_SIZE=256,
        )
        return (
            grad_x,
            grad_distance_float.to(distance_table.dtype),
        )


    @_directed_line_gather_compact_backward_cuda.register_fake
    def _directed_line_gather_compact_backward_cuda_fake(
        x_active: Tensor,
        ray_bits: Tensor,
        distance_table: Tensor,
        active_flat_indices: Tensor,
        active_flat_lookup: Tensor,
        grad_output: Tensor,
        height: int,
        width: int,
        radius: int,
    ) -> tuple[Tensor, Tensor]:
        del (
            ray_bits,
            active_flat_indices,
            active_flat_lookup,
            grad_output,
            height,
            width,
            radius,
        )
        return (
            torch.empty_like(x_active),
            torch.empty_like(distance_table),
        )


    def _directed_line_gather_compact_setup_context(
        ctx,
        inputs,
        output,
    ) -> None:
        del output
        (
            x_active,
            ray_bits,
            distance_table,
            active_flat_indices,
            active_flat_lookup,
            height,
            width,
            radius,
        ) = inputs
        ctx.save_for_backward(
            x_active,
            ray_bits,
            distance_table,
            active_flat_indices,
            active_flat_lookup,
        )
        ctx.height = height
        ctx.width = width
        ctx.radius = radius


    def _directed_line_gather_compact_backward(ctx, grad_output):
        (
            x_active,
            ray_bits,
            distance_table,
            active_flat_indices,
            active_flat_lookup,
        ) = ctx.saved_tensors
        grad_x, grad_distance = (
            _directed_line_gather_compact_backward_cuda(
                x_active,
                ray_bits,
                distance_table,
                active_flat_indices,
                active_flat_lookup,
                grad_output,
                ctx.height,
                ctx.width,
                ctx.radius,
            )
        )
        return (
            grad_x,
            None,
            grad_distance,
            None,
            None,
            None,
            None,
            None,
        )


    _directed_line_gather_compact_cuda.register_autograd(
        _directed_line_gather_compact_backward,
        setup_context=_directed_line_gather_compact_setup_context,
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


def _prepare_compact_indices(
    x_active: Tensor,
    ray_bits: Tensor,
    active_flat_indices: Tensor,
    active_flat_lookup: Tensor | None,
    height: int,
    width: int,
) -> tuple[Tensor, Tensor]:
    if x_active.ndim != 2:
        raise ValueError("x_active must be [active, channels]")
    if ray_bits.ndim != 3:
        raise ValueError("ray_bits must be [B,H,W]")
    if ray_bits.shape[1:] != (height, width):
        raise ValueError(
            "ray_bits spatial shape does not match height and width"
        )
    if active_flat_indices.ndim != 1:
        raise ValueError("active_flat_indices must be one-dimensional")
    if active_flat_indices.numel() != x_active.shape[0]:
        raise ValueError(
            "active_flat_indices and x_active must have equal rows"
        )
    active_flat_indices = active_flat_indices.to(
        device=x_active.device,
        dtype=torch.long,
    )
    total_cells = ray_bits.shape[0] * height * width
    if active_flat_lookup is None:
        active_flat_lookup = torch.full(
            (total_cells,),
            -1,
            dtype=torch.int32,
            device=x_active.device,
        )
        active_flat_lookup.index_copy_(
            0,
            active_flat_indices,
            torch.arange(
                active_flat_indices.numel(),
                dtype=torch.int32,
                device=x_active.device,
            ),
        )
    if (
        active_flat_lookup.ndim != 1
        or active_flat_lookup.numel() != total_cells
    ):
        raise ValueError(
            "active_flat_lookup must be one-dimensional with B*H*W elements"
        )
    return (
        active_flat_indices.contiguous(),
        active_flat_lookup.to(
            device=x_active.device,
            dtype=torch.int32,
        ).contiguous(),
    )


def axis_line_gather_compact(
    x_active: Tensor,
    ray_bits: Tensor,
    distance_table: Tensor,
    axis_eps: Tensor,
    active_flat_indices: Tensor,
    active_flat_lookup: Tensor | None,
    height: int,
    width: int,
    radius: int,
) -> Tensor:
    """Gather three blocker-aware axes directly in active-cell order."""

    if distance_table.ndim != 2 or distance_table.shape[0] < radius:
        raise ValueError(
            "distance_table must be [D,C] with D at least radius"
        )
    active_flat_indices, active_flat_lookup = (
        _prepare_compact_indices(
            x_active,
            ray_bits,
            active_flat_indices,
            active_flat_lookup,
            height,
            width,
        )
    )
    ray_bits = ray_bits.to(
        device=x_active.device,
        dtype=torch.int64,
    ).contiguous()
    if x_active.is_cuda and triton is not None:
        return _axis_line_gather_compact_cuda(
            x_active.contiguous(),
            ray_bits,
            distance_table.contiguous(),
            axis_eps,
            active_flat_indices,
            active_flat_lookup,
            height,
            width,
            radius,
        )
    return axis_line_gather_compact_reference(
        x_active,
        ray_bits,
        distance_table,
        axis_eps,
        active_flat_indices,
        active_flat_lookup,
        height,
        width,
        radius,
    )


def directed_line_gather_compact(
    x_active: Tensor,
    ray_bits: Tensor,
    distance_table: Tensor,
    active_flat_indices: Tensor,
    active_flat_lookup: Tensor | None,
    height: int,
    width: int,
    radius: int,
) -> Tensor:
    """Gather six directed messages without a padded source raster."""

    if distance_table.ndim != 2 or distance_table.shape[0] < radius:
        raise ValueError(
            "distance_table must be [D,C] with D at least radius"
        )
    active_flat_indices, active_flat_lookup = (
        _prepare_compact_indices(
            x_active,
            ray_bits,
            active_flat_indices,
            active_flat_lookup,
            height,
            width,
        )
    )
    ray_bits = ray_bits.to(
        device=x_active.device,
        dtype=torch.int64,
    ).contiguous()
    if x_active.is_cuda and triton is not None:
        return _directed_line_gather_compact_cuda(
            x_active.contiguous(),
            ray_bits,
            distance_table.contiguous(),
            active_flat_indices,
            active_flat_lookup,
            height,
            width,
            radius,
        )
    return directed_line_gather_compact_reference(
        x_active,
        ray_bits,
        distance_table,
        active_flat_indices,
        active_flat_lookup,
        height,
        width,
        radius,
    )
