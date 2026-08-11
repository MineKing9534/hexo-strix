"""Dense hexagonal residual CNN with windowed three-axis attention."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class HexConv2d(nn.Conv2d):
    """A depthwise 3x3 convolution over centre plus six axial neighbours."""

    def __init__(self, channels: int, *, bias: bool = True) -> None:
        super().__init__(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=bias,
        )
        mask = torch.ones((1, 1, 3, 3))
        # Raster rows are r and columns are q.  The invalid square-grid
        # diagonals are therefore (-q,-r) and (+q,+r).
        mask[..., 0, 0] = 0.0
        mask[..., 2, 2] = 0.0
        self.register_buffer("hex_mask", mask, persistent=False)

    def forward(self, inputs: Tensor) -> Tensor:
        return F.conv2d(
            inputs,
            self.weight * self.hex_mask,
            self.bias,
            stride=1,
            padding=1,
            groups=self.groups,
        )


class ChannelLayerNorm(nn.LayerNorm):
    """Layer-normalize channels independently at every raster cell."""

    def __init__(self, channels: int) -> None:
        super().__init__(channels)

    def forward(self, inputs: Tensor) -> Tensor:
        channels_last = inputs.permute(0, 2, 3, 1)
        normalized = super().forward(channels_last)
        return normalized.permute(0, 3, 1, 2)


def _three_way_softmax(logits: Tensor) -> Tensor:
    """Stable softmax for the fixed q/r/q+r axis dimension.

    Spelling out all three lanes avoids Inductor's generic split-reduction
    softmax lowering. Half-precision inputs retain float32 accumulation, then
    return to the convolution dtype just like the surrounding autocast path.
    """

    if logits.ndim < 2 or logits.shape[1] != 3:
        raise ValueError("three-way softmax expects exactly three axis logits")
    work = logits.float() if logits.dtype in {torch.float16, torch.bfloat16} else logits
    q_logit, r_logit, s_logit = work.unbind(dim=1)
    maximum = torch.maximum(torch.maximum(q_logit, r_logit), s_logit)
    q_weight = (q_logit - maximum).exp()
    r_weight = (r_logit - maximum).exp()
    s_weight = (s_logit - maximum).exp()
    inverse_sum = (q_weight + r_weight + s_weight).reciprocal()
    weights = torch.stack(
        (
            q_weight * inverse_sum,
            r_weight * inverse_sum,
            s_weight * inverse_sum,
        ),
        dim=1,
    )
    return weights.to(dtype=logits.dtype)


class HexResidualBlock(nn.Module):
    """Pre-normalized residual block using a true hex-local convolution."""

    def __init__(self, channels: int, expansion: int, dropout: float) -> None:
        super().__init__()
        expanded_channels = channels * expansion
        self.norm1 = ChannelLayerNorm(channels)
        self.hex_conv = HexConv2d(channels)
        self.norm2 = ChannelLayerNorm(channels)
        self.channel_expand = nn.Conv2d(
            channels,
            expanded_channels,
            kernel_size=1,
        )
        self.channel_project = nn.Conv2d(
            expanded_channels,
            channels,
            kernel_size=1,
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.layer_scale = nn.Parameter(torch.full((channels,), 1.0e-2))

    def forward(self, inputs: Tensor, active_mask: Tensor) -> Tensor:
        hidden = self.hex_conv(F.silu(self.norm1(inputs)))
        hidden = self.channel_expand(self.norm2(hidden))
        hidden = self.channel_project(F.silu(hidden))
        hidden = self.dropout(hidden)
        output = inputs + self.layer_scale[None, :, None, None] * hidden
        return output * active_mask


class GatedDilatedHexConv(nn.Module):
    """Three axial depthwise branches selected by a per-cell content gate.

    One grouped 3x3 convolution emits q, r, and q+r branches for every
    channel.  Dilation changes the sampled axial distance without changing
    the nine-tap kernel footprint.  Only centre plus the two cells on the
    selected axis have trainable effect in each branch.
    """

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        if channels <= 0 or dilation <= 0:
            raise ValueError("channels and dilation must be positive")
        self.channels = channels
        self.dilation = dilation
        self.axis_conv = nn.Conv2d(
            channels,
            3 * channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
        )
        self.axis_gate = nn.Conv2d(channels, 3, kernel_size=1)
        nn.init.zeros_(self.axis_gate.weight)
        nn.init.zeros_(self.axis_gate.bias)

        masks = torch.zeros((3, 1, 3, 3))
        # Constant r, varying q.
        masks[0, 0, 1, :] = 1.0
        # Constant q, varying r.
        masks[1, 0, :, 1] = 1.0
        # Constant q+r, varying along the valid anti-diagonal.
        masks[2, 0, 0, 2] = 1.0
        masks[2, 0, 1, 1] = 1.0
        masks[2, 0, 2, 0] = 1.0
        self.register_buffer(
            "axis_mask",
            masks.repeat(channels, 1, 1, 1),
            persistent=False,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        batch, channels, height, width = inputs.shape
        branches = F.conv2d(
            inputs,
            self.axis_conv.weight * self.axis_mask,
            self.axis_conv.bias,
            padding=self.dilation,
            dilation=self.dilation,
            groups=self.channels,
        ).reshape(batch, channels, 3, height, width)
        gates = _three_way_softmax(self.axis_gate(inputs))
        return (branches * gates[:, None]).sum(dim=2)


class D6GatedDilatedHexConv(nn.Module):
    """Exactly D6-equivariant content-gated mixing over the three hex axes.

    Rotations and reflections only permute the three undirected axes.  Sharing
    one line kernel and one content-gate kernel across those axes, while tying
    the two ends of every line, therefore makes the weighted sum equivariant by
    construction.  Features remain scalar fields; no orientation channels or
    group expansion are required.
    """

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        if channels <= 0 or dilation <= 0:
            raise ValueError("channels and dilation must be positive")
        self.channels = channels
        self.dilation = dilation
        self.main_center = nn.Parameter(torch.empty(channels))
        self.main_neighbor = nn.Parameter(torch.empty(channels))
        self.main_bias = nn.Parameter(torch.empty(channels))
        # Score the three axes from the line responses already gathered by the
        # main convolution. This retains a content-dependent equivariant gate
        # without paying for a second spatial convolution.
        self.axis_gate = nn.Parameter(torch.empty(channels))

        # Rows are axes and columns are flattened 3x3 raster offsets.  The
        # centre is included separately so the two endpoints can share one
        # reflection-invariant coefficient.
        endpoints = torch.zeros((3, 9))
        endpoints[0, [3, 5]] = 1.0  # constant r
        endpoints[1, [1, 7]] = 1.0  # constant q
        endpoints[2, [2, 6]] = 1.0  # constant q+r
        centre = torch.zeros((3, 9))
        centre[:, 4] = 1.0
        self.register_buffer("axis_endpoints", endpoints, persistent=False)
        self.register_buffer("axis_centres", centre, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Match depthwise Conv2d's scale for a three-tap receptive field.
        bound = 1.0 / math.sqrt(3.0)
        nn.init.uniform_(self.main_center, -bound, bound)
        nn.init.uniform_(self.main_neighbor, -bound, bound)
        nn.init.uniform_(self.main_bias, -bound, bound)
        nn.init.zeros_(self.axis_gate)

    def _main_weight(self) -> Tensor:
        lines = (
            self.axis_centres[:, None] * self.main_center[None, :, None]
            + self.axis_endpoints[:, None] * self.main_neighbor[None, :, None]
        )
        # Grouped convolution expects the three axes for each channel to be
        # contiguous: c0/q,c0/r,c0/s,c1/q,...
        return lines.permute(1, 0, 2).reshape(3 * self.channels, 1, 3, 3)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, channels, height, width = inputs.shape
        branches = F.conv2d(
            inputs,
            self._main_weight(),
            self.main_bias.repeat_interleave(3),
            padding=self.dilation,
            dilation=self.dilation,
            groups=self.channels,
        ).reshape(batch, channels, 3, height, width)
        gate_logits = (
            branches * self.axis_gate[None, :, None, None, None]
        ).sum(dim=1)
        gates = _three_way_softmax(gate_logits)
        return (branches * gates[:, None]).sum(dim=2)


class HexDilatedResidualBlock(nn.Module):
    """Content-gated, dilated axial mixing plus pointwise channel mixing."""

    def __init__(
        self,
        channels: int,
        expansion: int,
        dropout: float,
        dilation: int,
    ) -> None:
        super().__init__()
        expanded_channels = channels * expansion
        self.norm1 = ChannelLayerNorm(channels)
        self.axis_conv = GatedDilatedHexConv(channels, dilation)
        self.norm2 = ChannelLayerNorm(channels)
        self.channel_expand = nn.Conv2d(
            channels,
            expanded_channels,
            kernel_size=1,
        )
        self.channel_project = nn.Conv2d(
            expanded_channels,
            channels,
            kernel_size=1,
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.layer_scale = nn.Parameter(torch.full((channels,), 1.0e-2))

    def forward(self, inputs: Tensor, active_mask: Tensor) -> Tensor:
        # LayerNorm's learned affine bias makes an all-zero inactive cell
        # non-zero. Feeding those values into a spatial convolution exposes
        # the surrounding axial rectangle, whose square corners are not D6
        # invariant. Reapply the semantic-cell mask after normalization so the
        # convolution sees the intended zero extension of the active hex set.
        hidden = F.silu(self.norm1(inputs)) * active_mask
        hidden = self.axis_conv(hidden)
        hidden = self.channel_expand(self.norm2(hidden))
        hidden = self.channel_project(F.silu(hidden))
        hidden = self.dropout(hidden)
        output = inputs + self.layer_scale[None, :, None, None] * hidden
        return output * active_mask


class D6HexDilatedResidualBlock(HexDilatedResidualBlock):
    """Dilated residual block whose spatial mixer is exactly D6-equivariant."""

    def __init__(
        self,
        channels: int,
        expansion: int,
        dropout: float,
        dilation: int,
    ) -> None:
        super().__init__(channels, expansion, dropout, dilation)
        self.axis_conv = D6GatedDilatedHexConv(channels, dilation)


class HexAxialAttention(nn.Module):
    """Content-dependent attention over q, r, and q+r hex lines.

    The three orientations share projections and relative-distance biases.
    Only active cells act as queries or keys. Each line family is converted to
    regular padded sequences, then attended through a sliding window. This
    keeps memory linear in line length rather than materializing an L x L
    attention matrix for every padded raster line.
    """

    def __init__(
        self,
        channels: int,
        heads: int,
        radius: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if heads <= 0 or channels % heads != 0:
            raise ValueError("axial-attention heads must divide channels")
        if radius <= 0:
            raise ValueError("axial-attention radius must be positive")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.radius = radius
        self.dropout = dropout
        self.norm = ChannelLayerNorm(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, kernel_size=1)
        self.output = nn.Conv2d(channels, channels, kernel_size=1)
        self.relative_bias = nn.Parameter(torch.zeros(heads, radius + 1))
        self.layer_scale = nn.Parameter(torch.full((channels,), 1.0e-2))

    @staticmethod
    def _pack_axis(
        tensor: Tensor,
        axis: int,
    ) -> Tensor:
        """Return [B, lines, length, C] sequences for one hex axis."""

        batch, channels, height, width = tensor.shape
        if axis == 0:  # constant r; vary q
            return tensor.permute(0, 2, 3, 1)
        if axis == 1:  # constant q; vary r
            return tensor.permute(0, 3, 2, 1)
        if axis != 2:
            raise ValueError(f"invalid hex axis {axis}")

        # Constant q+r anti-diagonals.  Every line is padded to ``width``;
        # invalid entries gather one appended all-zero token.
        line = torch.arange(
            height + width - 1,
            device=tensor.device,
        )[:, None]
        column = torch.arange(width, device=tensor.device)[None, :]
        row = line - column
        valid = (row >= 0) & (row < height)
        dummy = height * width
        indices = torch.where(valid, row * width + column, dummy).reshape(-1)
        flat = tensor.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        flat = torch.cat((flat, flat.new_zeros((batch, 1, channels))), dim=1)
        return flat.index_select(1, indices).reshape(
            batch,
            height + width - 1,
            width,
            channels,
        )

    @staticmethod
    def _unpack_axis(
        lines: Tensor,
        axis: int,
        *,
        height: int,
        width: int,
    ) -> Tensor:
        """Restore [B,C,H,W] cells from one line-family result."""

        if axis == 0:
            return lines.permute(0, 3, 1, 2)
        if axis == 1:
            return lines.permute(0, 3, 2, 1)
        batch, _line_count, _length, channels = lines.shape
        row = torch.arange(height, device=lines.device)[:, None]
        column = torch.arange(width, device=lines.device)[None, :]
        indices = ((row + column) * width + column).reshape(-1)
        cells = lines.reshape(batch, -1, channels).index_select(1, indices)
        return cells.reshape(batch, height, width, channels).permute(0, 3, 1, 2)

    def _attend(self, q: Tensor, k: Tensor, v: Tensor, active: Tensor) -> Tensor:
        batch, lines, length, channels = q.shape
        merged = batch * lines

        def heads(tensor: Tensor) -> Tensor:
            return tensor.reshape(
                merged,
                length,
                self.heads,
                self.head_dim,
            )

        q_heads = heads(q)
        k_heads = heads(k)
        v_heads = heads(v)
        active = active.reshape(merged, length)

        window = 2 * self.radius + 1
        padded_k = F.pad(k_heads, (0, 0, 0, 0, self.radius, self.radius))
        padded_v = F.pad(v_heads, (0, 0, 0, 0, self.radius, self.radius))
        padded_active = F.pad(active, (self.radius, self.radius), value=False)

        # Do not build [cell, head, window, head_dim] unfolded K/V tensors.
        # Inductor materializes those views for AOT backward; at a 500k-cell
        # budget that expands a bounded-attention FIT into tens of GiB. The
        # offset form retains only [cell, head, window] logits/weights and
        # accumulates shifted values directly, preserving the same operation.
        logits = torch.stack(
            [
                (
                    q_heads
                    * padded_k[:, offset : offset + length]
                ).sum(dim=-1)
                for offset in range(window)
            ],
            dim=-1,
        ) / math.sqrt(self.head_dim)
        active_windows = torch.stack(
            [
                padded_active[:, offset : offset + length]
                for offset in range(window)
            ],
            dim=-1,
        )
        distances = torch.arange(
            -self.radius,
            self.radius + 1,
            device=q.device,
        ).abs()
        bias = self.relative_bias.index_select(1, distances)
        logits = logits + bias[None, None]
        logits = logits.masked_fill(
            ~active_windows[:, :, None, :],
            torch.finfo(logits.dtype).min,
        )
        weights = torch.softmax(logits, dim=-1)
        if self.training and self.dropout > 0.0:
            weights = F.dropout(weights, p=self.dropout)
        attended = sum(
            weights[..., offset, None]
            * padded_v[:, offset : offset + length]
            for offset in range(window)
        ).reshape(
            batch,
            lines,
            length,
            channels,
        )
        return attended * active.reshape(batch, lines, length, 1)

    def forward(self, inputs: Tensor, active_mask: Tensor) -> Tensor:
        batch, _channels, height, width = inputs.shape
        q, k, v = self.qkv(F.silu(self.norm(inputs))).chunk(3, dim=1)
        active = active_mask.to(dtype=torch.bool)
        axis_sum = torch.zeros_like(inputs)
        for axis in range(3):
            q_lines = self._pack_axis(q, axis)
            k_lines = self._pack_axis(k, axis)
            v_lines = self._pack_axis(v, axis)
            mask_lines = self._pack_axis(active, axis).squeeze(-1)
            attended = self._attend(
                q_lines,
                k_lines,
                v_lines,
                mask_lines,
            )
            axis_sum = axis_sum + self._unpack_axis(
                attended,
                axis,
                height=height,
                width=width,
            )
        hidden = self.output(axis_sum / math.sqrt(3.0))
        output = inputs + self.layer_scale[None, :, None, None] * hidden
        return output * active_mask


class HexAxialCNNBackbone(nn.Module):
    """Residual hex CNN with optional axial-attention layers."""

    input_planes = 8
    input_scalars = 5

    def __init__(
        self,
        *,
        channels: int,
        blocks: int,
        heads: int,
        attention_radius: int,
        attention_layers: tuple[int, ...],
        expansion: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if channels <= 0 or blocks <= 0 or expansion <= 0:
            raise ValueError("hex CNN channels and blocks must be positive")
        invalid_layers = [
            layer
            for layer in attention_layers
            if layer < 0 or layer >= blocks
        ]
        if invalid_layers:
            raise ValueError(
                f"axial-attention layers outside [0,{blocks}): {invalid_layers}"
            )
        self.input_projection = nn.Conv2d(
            self.input_planes,
            channels,
            kernel_size=1,
        )
        self.scalar_projection = nn.Sequential(
            nn.Linear(self.input_scalars, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.blocks = nn.ModuleList(
            HexResidualBlock(channels, expansion, dropout)
            for _ in range(blocks)
        )
        attention_set = set(attention_layers)
        self.attention = nn.ModuleDict(
            {
                str(layer): HexAxialAttention(
                    channels,
                    heads,
                    attention_radius,
                    dropout,
                )
                for layer in attention_layers
            }
        )
        self.attention_layers = attention_set
        self.output_norm = ChannelLayerNorm(channels)

    def forward(
        self,
        planes: Tensor,
        scalars: Tensor,
        active_mask: Tensor,
    ) -> Tensor:
        mask = active_mask.to(dtype=planes.dtype)
        hidden = self.input_projection(planes)
        hidden = hidden + self.scalar_projection(scalars)[:, :, None, None]
        hidden = hidden * mask
        for layer, block in enumerate(self.blocks):
            hidden = block(hidden, mask)
            if layer in self.attention_layers:
                hidden = self.attention[str(layer)](hidden, mask)
        return F.silu(self.output_norm(hidden)) * mask


class HexDilatedCNNBackbone(nn.Module):
    """Sparse multiscale hex CNN with content-selected axial branches."""

    input_planes = 8
    input_scalars = 5

    def __init__(
        self,
        *,
        channels: int,
        dilations: tuple[int, ...],
        expansion: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if channels <= 0 or not dilations or expansion <= 0:
            raise ValueError("hex CNN dimensions and dilations must be positive")
        if any(dilation <= 0 for dilation in dilations):
            raise ValueError("hex CNN dilations must be positive")
        self.input_projection = nn.Conv2d(
            self.input_planes,
            channels,
            kernel_size=1,
        )
        self.scalar_projection = nn.Sequential(
            nn.Linear(self.input_scalars, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.blocks = nn.ModuleList(
            HexDilatedResidualBlock(
                channels,
                expansion,
                dropout,
                dilation,
            )
            for dilation in dilations
        )
        self.output_norm = ChannelLayerNorm(channels)

    def forward(
        self,
        planes: Tensor,
        scalars: Tensor,
        active_mask: Tensor,
    ) -> Tensor:
        mask = active_mask.to(dtype=planes.dtype)
        hidden = self.input_projection(planes)
        hidden = hidden + self.scalar_projection(scalars)[:, :, None, None]
        hidden = hidden * mask
        for block in self.blocks:
            hidden = block(hidden, mask)
        return F.silu(self.output_norm(hidden)) * mask


class HexD6DilatedCNNBackbone(HexDilatedCNNBackbone):
    """Multiscale hex CNN with exact rotation/reflection equivariance."""

    def __init__(
        self,
        *,
        channels: int,
        dilations: tuple[int, ...],
        expansion: int,
        dropout: float,
    ) -> None:
        super().__init__(
            channels=channels,
            dilations=dilations,
            expansion=expansion,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            D6HexDilatedResidualBlock(
                channels,
                expansion,
                dropout,
                dilation,
            )
            for dilation in dilations
        )
