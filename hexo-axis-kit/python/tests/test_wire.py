from __future__ import annotations

import struct

import numpy as np
import torch

from hexo_axis_models.wire import decode_hxr1


def build_fixture() -> bytes:
    batch, planes, scalars, width, height, radius = 1, 8, 5, 3, 2, 5
    cells = width * height
    legal = np.array([1, 4], dtype="<u4")
    parts = [
        struct.pack(
            "<4s6H2I",
            b"HXR1",
            1,
            planes,
            scalars,
            width,
            height,
            radius,
            batch,
            len(legal),
        ),
        np.arange(batch * planes * cells, dtype="<f4").tobytes(),
        np.arange(batch * scalars, dtype="<f4").tobytes(),
        np.array([1, 1, 0, 0, 1, 0], dtype="u1").tobytes(),
        np.array([1, 0, 0, 0, 1 << 15, 0], dtype="<u4").tobytes(),
        np.array([0, len(legal)], dtype="<u4").tobytes(),
        legal.tobytes(),
        np.array([[-2, 7]], dtype="<i4").tobytes(),
    ]
    return b"".join(parts)


def test_decode_hxr1():
    batch = decode_hxr1(build_fixture())
    assert batch.planes.shape == (1, 8, 2, 3)
    assert batch.scalars.shape == (1, 5)
    assert batch.active_mask.shape == (1, 1, 2, 3)
    assert batch.legal_offsets.tolist() == [0, 2]
    assert batch.legal_flat_indices.tolist() == [1, 4]
    assert batch.active_flat_indices.tolist() == [0, 1, 4]
    assert batch.active_flat_lookup.tolist() == [0, 1, -1, -1, 2, -1]
    assert batch.active_flat_lookup.dtype == torch.int32
    assert batch.origins.tolist() == [[-2, 7]]
    mask = batch.ray_mask
    assert bool(mask[0, 0, 0, 0, 0])
    assert bool(mask[0, 3, 0, 1, 1])
    assert int(mask.sum()) == 2
    assert batch.ray_bits.dtype == torch.int64


def test_raster_batch_slice_rebases_offsets_and_flat_indices():
    first = decode_hxr1(build_fixture())
    planes = torch.cat([first.planes, first.planes], dim=0)
    scalars = torch.cat([first.scalars, first.scalars], dim=0)
    active = torch.cat([first.active_mask, first.active_mask], dim=0)
    rays = torch.cat([first.ray_bits, first.ray_bits], dim=0)
    batch = type(first)(
        planes=planes,
        scalars=scalars,
        active_mask=active,
        ray_bits=rays,
        legal_offsets=torch.tensor([0, 2, 4]),
        legal_flat_indices=torch.tensor([1, 4, 7, 10]),
        active_flat_indices=torch.tensor([0, 1, 4, 6, 7, 10]),
        origins=torch.cat([first.origins, first.origins], dim=0),
        ray_radius=first.ray_radius,
    )
    second = batch.slice(1, 2)
    assert second.legal_offsets.tolist() == [0, 2]
    assert second.legal_flat_indices.tolist() == [1, 4]
    assert second.active_flat_indices.tolist() == [0, 1, 4]
    assert second.active_flat_lookup.tolist() == [0, 1, -1, -1, 2, -1]
