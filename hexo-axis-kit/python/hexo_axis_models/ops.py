from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

NUM_RAYS: Final[int] = 6
PACKED_RAY_RADIUS: Final[int] = 5
RAY_DIRS: Final[tuple[tuple[int, int], ...]] = (
    (1, 0),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (0, -1),
    (1, -1),
)
AXIS_RAY_PAIRS: Final[tuple[tuple[int, int], ...]] = ((0, 3), (1, 4), (2, 5))


def unpack_ray_bits(ray_bits: Tensor, radius: int = PACKED_RAY_RADIUS) -> Tensor:
    """Unpack Rust's 30-bit words to ``[B, 6, radius, H, W]`` bool masks.

    ``ray_bits`` may be ``[B, H, W]`` or ``[H, W]``. The fixed packed layout
    always reserves five bits per ray, even when a curriculum stage uses a
    shorter effective radius.
    """
    if not 1 <= radius <= PACKED_RAY_RADIUS:
        raise ValueError(f"radius must be in 1..={PACKED_RAY_RADIUS}, got {radius}")
    squeeze = ray_bits.ndim == 2
    if squeeze:
        ray_bits = ray_bits.unsqueeze(0)
    if ray_bits.ndim != 3:
        raise ValueError(f"ray_bits must have shape [B,H,W] or [H,W], got {tuple(ray_bits.shape)}")
    bits = ray_bits.to(torch.int64)
    bit_ids = torch.arange(NUM_RAYS * PACKED_RAY_RADIUS, device=bits.device, dtype=torch.int64)
    unpacked = ((bits.unsqueeze(-1) >> bit_ids) & 1).to(torch.bool)
    b, h, w, _ = unpacked.shape
    unpacked = unpacked.view(b, h, w, NUM_RAYS, PACKED_RAY_RADIUS)
    unpacked = unpacked[..., :radius].permute(0, 3, 4, 1, 2).contiguous()
    return unpacked.squeeze(0) if squeeze else unpacked


def roll_source(x: Tensor, dq: int, dr: int) -> Tensor:
    """Return source features at ``dest + (dq, dr)`` using axial raster axes.

    Wrap-around values from ``torch.roll`` are harmless because every caller
    multiplies by the exact Rust ray mask, whose boundary entries are zero.
    """
    return torch.roll(x, shifts=(-dr, -dq), dims=(-2, -1))


def masked_sum(x: Tensor, mask: Tensor, dims: tuple[int, ...]) -> Tensor:
    mask_f = mask.to(dtype=x.dtype)
    return (x * mask_f).sum(dim=dims)


def masked_mean(x: Tensor, mask: Tensor, dims: tuple[int, ...], eps: float = 1.0) -> Tensor:
    mask_f = mask.to(dtype=x.dtype)
    numerator = (x * mask_f).sum(dim=dims)
    denominator = mask_f.sum(dim=dims).clamp_min(eps)
    return numerator / denominator


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels independently at every raster cell."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        if not torch.is_grad_enabled():
            y = x.permute(0, 2, 3, 1)
            y = F.layer_norm(
                y,
                (self.channels,),
                self.weight,
                self.bias,
                self.eps,
            )
            return y.permute(0, 3, 1, 2).contiguous()

        # Spell out channel LayerNorm for the raster trunk. On ROCm/gfx1151,
        # Inductor currently lowers the backward of native_layer_norm together
        # with the surrounding masked reductions into an invalid Triton kernel
        # ("operand does not dominate this use"). Keeping the reduction
        # explicit avoids that compiler path and retains fp32 accumulation
        # under bf16 autocast, matching native LayerNorm's numerical policy.
        y = x.permute(0, 2, 3, 1)
        y_float = y.float()
        mean = y_float.mean(dim=-1, keepdim=True)
        centered = y_float - mean
        variance = centered.square().mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.eps)
        normalized = normalized * self.weight + self.bias
        return normalized.to(dtype=x.dtype).permute(0, 3, 1, 2).contiguous()

    def forward_vector(self, x: Tensor) -> Tensor:
        if not torch.is_grad_enabled():
            return F.layer_norm(
                x,
                (self.channels,),
                self.weight,
                self.bias,
                self.eps,
            )
        x_float = x.float()
        mean = x_float.mean(dim=-1, keepdim=True)
        centered = x_float - mean
        variance = centered.square().mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight + self.bias).to(dtype=x.dtype)


class PointwiseMLP(nn.Module):
    """Two-layer per-cell MLP represented as 1x1 convolutions."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.relu(self.fc1(x)))

    def forward_vector(self, x: Tensor) -> Tensor:
        w1 = self.fc1.weight[:, :, 0, 0]
        w2 = self.fc2.weight[:, :, 0, 0]
        x = F.linear(x, w1, self.fc1.bias)
        x = F.relu(x)
        return F.linear(x, w2, self.fc2.bias)


class SpatialMLPHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, *, tanh: bool = False) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.tanh = bool(tanh)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc2(F.relu(self.fc1(x))).squeeze(1)
        return torch.tanh(x) if self.tanh else x


def gather_dense_actions(dense: Tensor, legal_flat_indices: Tensor) -> Tensor:
    """Gather from ``[B,H,W]`` using Rust batch-global flattened indices."""
    if dense.ndim != 3:
        raise ValueError(f"dense must be [B,H,W], got {tuple(dense.shape)}")
    return dense.reshape(-1).index_select(0, legal_flat_indices.to(torch.long))
