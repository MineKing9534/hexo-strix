from __future__ import annotations

from dataclasses import dataclass
import struct

import numpy as np
import torch
from torch import Tensor

from .ops import unpack_ray_bits

_MAGIC = b"HXR1"
_VERSION = 1
_HEADER = struct.Struct("<4s6H2I")


@dataclass
class RasterTensorBatch:
    planes: Tensor                 # [B,P,H,W] float32
    scalars: Tensor                # [B,S] float32
    active_mask: Tensor            # [B,1,H,W] bool
    ray_bits: Tensor               # [B,H,W] int64
    legal_offsets: Tensor          # [B+1] int64
    legal_flat_indices: Tensor     # [N] int64, global over B*H*W
    active_flat_indices: Tensor    # [A] int64, global over B*H*W
    origins: Tensor                # [B,2] int32
    ray_radius: int
    active_flat_lookup: Tensor | None = None  # [B*H*W] int32, -1 if inactive

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "RasterTensorBatch":
        return RasterTensorBatch(
            planes=self.planes.to(device, non_blocking=non_blocking),
            scalars=self.scalars.to(device, non_blocking=non_blocking),
            active_mask=self.active_mask.to(device, non_blocking=non_blocking),
            ray_bits=self.ray_bits.to(device, non_blocking=non_blocking),
            legal_offsets=self.legal_offsets.to(
                device, non_blocking=non_blocking
            ),
            legal_flat_indices=self.legal_flat_indices.to(
                device, non_blocking=non_blocking
            ),
            active_flat_indices=self.active_flat_indices.to(
                device, non_blocking=non_blocking
            ),
            origins=self.origins.to(device, non_blocking=non_blocking),
            ray_radius=self.ray_radius,
            active_flat_lookup=(
                None
                if self.active_flat_lookup is None
                else self.active_flat_lookup.to(
                    device,
                    non_blocking=non_blocking,
                )
            ),
        )

    def slice(self, start: int, end: int) -> "RasterTensorBatch":
        """Take a contiguous graph range and rebase its flattened legal indices."""

        batch_size = self.planes.shape[0]
        if not 0 <= start < end <= batch_size:
            raise IndexError(
                f"invalid raster batch slice [{start}:{end}] for {batch_size} positions"
            )
        cells = self.planes.shape[-2] * self.planes.shape[-1]
        legal_start = int(self.legal_offsets[start])
        legal_end = int(self.legal_offsets[end])
        legal_offsets = (
            self.legal_offsets[start : end + 1] - self.legal_offsets[start]
        )
        legal_flat_indices = (
            self.legal_flat_indices[legal_start:legal_end] - start * cells
        )
        active_start = start * cells
        active_end = end * cells
        selected_active = (
            (self.active_flat_indices >= active_start)
            & (self.active_flat_indices < active_end)
        )
        active_flat_indices = (
            self.active_flat_indices[selected_active] - active_start
        )
        active_flat_lookup = torch.full(
            ((end - start) * cells,),
            -1,
            dtype=torch.int32,
            device=active_flat_indices.device,
        )
        active_flat_lookup.index_copy_(
            0,
            active_flat_indices,
            torch.arange(
                active_flat_indices.numel(),
                dtype=torch.int32,
                device=active_flat_indices.device,
            ),
        )
        return RasterTensorBatch(
            planes=self.planes[start:end],
            scalars=self.scalars[start:end],
            active_mask=self.active_mask[start:end],
            ray_bits=self.ray_bits[start:end],
            legal_offsets=legal_offsets,
            legal_flat_indices=legal_flat_indices,
            active_flat_indices=active_flat_indices,
            origins=self.origins[start:end],
            ray_radius=self.ray_radius,
            active_flat_lookup=active_flat_lookup,
        )

    @property
    def ray_mask(self) -> Tensor:
        return unpack_ray_bits(self.ray_bits, self.ray_radius)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return tuple(self.planes.shape)  # type: ignore[return-value]


def _take_array(
    view: memoryview,
    offset: int,
    dtype: np.dtype,
    count: int,
) -> tuple[np.ndarray, int]:
    nbytes = int(np.dtype(dtype).itemsize) * count
    end = offset + nbytes
    if end > len(view):
        raise ValueError(f"truncated HXR1 payload: need byte {end}, have {len(view)}")
    array = np.frombuffer(view[offset:end], dtype=dtype, count=count).copy()
    return array, end


def decode_hxr1(data: bytes | bytearray | memoryview) -> RasterTensorBatch:
    """Decode ``RasterBatch::encode_hxr1`` output into CPU tensors."""
    view = memoryview(data)
    if len(view) < _HEADER.size:
        raise ValueError(f"HXR1 header needs {_HEADER.size} bytes, got {len(view)}")
    (
        magic,
        version,
        plane_count,
        scalar_count,
        width,
        height,
        ray_radius,
        batch_size,
        total_legal,
    ) = _HEADER.unpack_from(view, 0)
    if magic != _MAGIC:
        raise ValueError(f"bad HXR1 magic {magic!r}")
    if version != _VERSION:
        raise ValueError(f"unsupported HXR1 version {version}")
    if ray_radius < 1 or ray_radius > 5:
        raise ValueError(f"invalid packed ray radius {ray_radius}")

    cells = int(width) * int(height)
    offset = _HEADER.size
    planes_np, offset = _take_array(
        view, offset, np.dtype("<f4"), int(batch_size) * int(plane_count) * cells
    )
    scalars_np, offset = _take_array(
        view, offset, np.dtype("<f4"), int(batch_size) * int(scalar_count)
    )
    active_np, offset = _take_array(
        view, offset, np.dtype("u1"), int(batch_size) * cells
    )
    ray_np, offset = _take_array(
        view, offset, np.dtype("<u4"), int(batch_size) * cells
    )
    offsets_np, offset = _take_array(
        view, offset, np.dtype("<u4"), int(batch_size) + 1
    )
    legal_np, offset = _take_array(view, offset, np.dtype("<u4"), int(total_legal))
    origins_np, offset = _take_array(
        view, offset, np.dtype("<i4"), int(batch_size) * 2
    )
    if offset != len(view):
        raise ValueError(f"HXR1 payload has {len(view) - offset} trailing bytes")
    if offsets_np[-1] != total_legal:
        raise ValueError(
            f"legal_offsets[-1]={int(offsets_np[-1])} != total_legal={total_legal}"
        )

    planes = torch.from_numpy(
        planes_np.reshape(batch_size, plane_count, height, width)
    )
    scalars = torch.from_numpy(scalars_np.reshape(batch_size, scalar_count))
    active = torch.from_numpy(active_np.reshape(batch_size, 1, height, width)).to(torch.bool)
    # int64 avoids uint32 operator gaps in PyTorch while preserving all 30 bits.
    ray_bits = torch.from_numpy(ray_np.astype(np.int64).reshape(batch_size, height, width))
    legal_offsets = torch.from_numpy(offsets_np.astype(np.int64))
    legal_flat = torch.from_numpy(legal_np.astype(np.int64))
    active_flat = torch.from_numpy(
        np.flatnonzero(active_np).astype(np.int64)
    )
    active_lookup_np = np.full(
        int(batch_size) * cells,
        -1,
        dtype=np.int32,
    )
    active_lookup_np[active_flat.numpy()] = np.arange(
        active_flat.numel(),
        dtype=np.int32,
    )
    active_lookup = torch.from_numpy(active_lookup_np)
    origins = torch.from_numpy(origins_np.reshape(batch_size, 2))
    return RasterTensorBatch(
        planes=planes,
        scalars=scalars,
        active_mask=active,
        ray_bits=ray_bits,
        legal_offsets=legal_offsets,
        legal_flat_indices=legal_flat,
        active_flat_indices=active_flat,
        origins=origins,
        ray_radius=int(ray_radius),
        active_flat_lookup=active_lookup,
    )
