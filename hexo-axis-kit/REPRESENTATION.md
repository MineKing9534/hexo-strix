# Recommended HeXO raster representation

## Coordinate layout

Use an axial rectangle. Tensor column `x` is axial `q`; tensor row `y` is axial `r`:

```text
q = origin_q + x
r = origin_r + y
flat = y * width + x
```

The crop contains every stone and every currently legal action. It is centred in the smallest configured bucket that fits. Cells which are neither stones nor legal actions remain zero and are not graph nodes.

The representation is translation invariant. D6 augmentation should transform the `GameState` with the engine's existing coordinate transforms before raster construction, then rebuild the crop and legal ordering.

HeXO's placement radius is relative to existing stones, so it does not bound
the total board extent. A wandering trajectory can produce a huge,
mostly-empty axial rectangle. Production users must plan batches from raster
dimensions before allocating planes, impose a per-position cell limit, and
handle that boundary explicitly (KLENT uses frozen-value bootstrapping rather
than inventing a draw). A batch cell budget alone is insufficient if a single
position can exceed it.

## Spatial planes

Channel-first order, `P=8`:

| Index | Plane | Range |
|---:|---|---|
| 0 | side-to-move stones | 0 or 1 |
| 1 | opponent stones | 0 or 1 |
| 2 | legal actions | 0 or 1 |
| 3 | inverse distance to nearest stone | 0 to 1 |
| 4 | own best clean-window line | normalized by WL |
| 5 | opponent best clean-window line | normalized by WL |
| 6 | own axes with at least WL-2 stones | normalized by 3 |
| 7 | opponent axes with at least WL-2 stones | normalized by 3 |

The four threat planes call `hexo_engine::threat::node_threat_features`, so they match the current graph representation.

For checkpoint portability, `AxisGineCompatNet` assembles the current eight-dimensional lean node vector as:

```text
[own, opponent, moves_remaining, inverse_distance, four_threat_features]
```

The legal plane remains available for action masking but is not fed into the compatibility stem.

## Scalar features

`S=5`:

| Index | Scalar |
|---:|---|
| 0 | placements remaining this turn / 2 |
| 1 | win length / 7 |
| 2 | placement radius / 8 |
| 3 | remaining placement budget / maximum budget |
| 4 | +1 for P1 to move, -1 for P2 |

Only scalar 0 is used by the exact compatibility path. The others can enter through the model's zero-initialized conditioning branch after the ported baseline is validated.

## Active mask

One byte per cell. It is one for a stone or legal action and zero elsewhere. Global pooling and residual outputs are masked to active cells.

## Blocker-aware incoming-ray bits

Each cell carries a `u32`; 30 low bits are used:

```text
bit = ray * 5 + (distance - 1)
```

Ray order is cyclic:

| Ray | Offset | Opposite |
|---:|---|---:|
| 0 | (+1, 0) | 3 |
| 1 | (0, +1) | 4 |
| 2 | (-1, +1) | 5 |
| 3 | (-1, 0) | 0 |
| 4 | (0, -1) | 1 |
| 5 | (+1, -1) | 2 |

For destination `x`, a set bit means an incoming edge exists from:

```text
source = x + distance * ray_offset
```

The Rust builder reproduces the current axis graph's real-node semantics:

- maximum distance is `min(spec.ray_radius, WL-1)`;
- a missing intermediate graph node terminates a walk;
- a stone source stops at an opponent stone;
- an empty source stops at the first stone;
- empty-to-empty pairs are omitted when pruning is enabled;
- admitted pairs are bidirectional.

This mask is the key object which lets a dense destination-gather kernel preserve Strix's blocker-aware line graph without `edge_index` or scatter atomics.

## HXR1 batch wire format

`RasterBatch::encode_hxr1` writes:

```text
4B  "HXR1"
u16 version = 1
u16 plane_count
u16 scalar_count
u16 width
u16 height
u16 packed_ray_radius
u32 batch_size
u32 total_legal
f32 planes[B,P,H,W]
f32 scalars[B,S]
u8  active[B,H,W]
u32 ray_bits[B,H,W]
u32 legal_offsets[B+1]
u32 legal_flat_indices[total_legal]
i32 origins[B,2]
```

`hexo_axis_models.wire.decode_hxr1` decodes this directly into model tensors.
