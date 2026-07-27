//! Dense axial-raster representation for HeXO.
//!
//! This crate is intended to sit beside `hexo-engine` in the `hexo-rs`
//! workspace. It converts a [`hexo_engine::GameState`] into:
//!
//! - eight spatial feature planes in side-to-move-relative form;
//! - five graph-level scalar features;
//! - a packed, blocker-aware six-ray mask that reproduces the real-node edges
//!   of Strix's pruned axis-window graph for radii up to five;
//! - legal-action indices aligned with `GameState::legal_moves()` order.
//!
//! The packed ray layout is fixed at 30 bits per cell:
//!
//! ```text
//! bit = ray * 5 + (distance - 1)
//! ray 0: (+1,  0)    ray 3: (-1,  0)
//! ray 1: ( 0, +1)    ray 4: ( 0, -1)
//! ray 2: (-1, +1)    ray 5: (+1, -1)
//! ```
//!
//! For a destination cell `x`, a set bit `(ray, distance)` means there is an
//! incoming graph edge from `x + distance * RAY_DIRS[ray]`.

use std::error::Error;
use std::fmt;

use hexo_engine::hex::hex_distance;
use hexo_engine::threat::node_threat_features;
use hexo_engine::{Coord, GameState, Player};

/// Number of spatial feature planes emitted by [`build_raster`].
pub const NUM_PLANES: usize = 8;
/// Number of graph-level scalar features emitted by [`build_raster`].
pub const NUM_SCALARS: usize = 5;
/// Number of directed rays around a hex cell.
pub const NUM_RAYS: usize = 6;
/// Packed blocker-aware radius. Six rays × five distances = 30 bits.
pub const PACKED_RAY_RADIUS: usize = 5;
/// Number of occupied bits in each cell's packed ray word.
pub const PACKED_RAY_BITS: usize = NUM_RAYS * PACKED_RAY_RADIUS;

/// Cyclic directed-neighbour order. Opposite directions differ by three.
pub const RAY_DIRS: [Coord; NUM_RAYS] = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)];

/// Spatial-plane indices. Data are stored channel-first: `plane * H * W + cell`.
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plane {
    OwnStone = 0,
    OppStone = 1,
    Legal = 2,
    InvNearestStoneDistance = 3,
    OwnMaxCleanLine = 4,
    OppMaxCleanLine = 5,
    OwnThreatAxisCount = 6,
    OppThreatAxisCount = 7,
}

impl Plane {
    pub const ALL: [Plane; NUM_PLANES] = [
        Plane::OwnStone,
        Plane::OppStone,
        Plane::Legal,
        Plane::InvNearestStoneDistance,
        Plane::OwnMaxCleanLine,
        Plane::OppMaxCleanLine,
        Plane::OwnThreatAxisCount,
        Plane::OppThreatAxisCount,
    ];

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Graph-level scalar indices.
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scalar {
    /// `moves_remaining_this_turn / 2`.
    MovesRemaining = 0,
    /// `win_length / win_length_scale`.
    WinLength = 1,
    /// `placement_radius / placement_radius_scale`.
    PlacementRadius = 2,
    /// Fraction of the configured placement budget remaining.
    RemainingBudget = 3,
    /// `+1` for P1 to move, `-1` for P2 to move.
    ToMoveIdentity = 4,
}

impl Scalar {
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Controls crop bucketing and exact line-mask construction.
#[derive(Debug, Clone)]
pub struct RasterSpec {
    /// Candidate `(width, height)` buckets. The smallest fitting area is used.
    /// If none fits, each dimension is rounded to the next `1 + 8k` size.
    pub buckets: Vec<(u16, u16)>,
    /// Maximum exact blocker-aware radius to emit. Must be in `1..=5`.
    /// The effective radius is additionally capped at `win_length - 1`.
    pub ray_radius: u8,
    /// Match Strix's `prune_empty_edges=true` behaviour.
    pub prune_empty_edges: bool,
    /// Normaliser for the win-length scalar.
    pub win_length_scale: f32,
    /// Normaliser for the placement-radius scalar.
    pub placement_radius_scale: f32,
}

impl Default for RasterSpec {
    fn default() -> Self {
        Self {
            buckets: vec![
                (9, 9),
                (17, 17),
                (25, 25),
                (33, 33),
                (41, 41),
                (49, 49),
                (65, 65),
                (81, 81),
                (97, 97),
                (129, 129),
            ],
            ray_radius: PACKED_RAY_RADIUS as u8,
            prune_empty_edges: true,
            win_length_scale: 7.0,
            placement_radius_scale: 8.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RasterError {
    TerminalPosition,
    InvalidRayRadius(u8),
    EmptyPosition,
    CoordinateSpanOverflow,
    InvalidScale(&'static str),
    ShapeMismatch {
        expected: (u16, u16),
        found: (u16, u16),
    },
    PositionCellBudgetExceeded {
        width: u16,
        height: u16,
        cells: usize,
        budget: usize,
    },
    CorruptPosition(&'static str),
}

impl fmt::Display for RasterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RasterError::TerminalPosition => write!(f, "cannot rasterise a terminal position"),
            RasterError::InvalidRayRadius(r) => {
                write!(f, "ray_radius must be in 1..={PACKED_RAY_RADIUS}, got {r}")
            }
            RasterError::EmptyPosition => write!(f, "position contains no stones or legal moves"),
            RasterError::CoordinateSpanOverflow => write!(f, "raster span exceeds u16 dimensions"),
            RasterError::InvalidScale(name) => {
                write!(f, "normalisation scale {name} must be positive")
            }
            RasterError::ShapeMismatch { expected, found } => write!(
                f,
                "cannot batch raster shape {found:?}; expected {expected:?}"
            ),
            RasterError::PositionCellBudgetExceeded {
                width,
                height,
                cells,
                budget,
            } => write!(
                f,
                "raster position {width}x{height} has {cells} padded cells, \
                 exceeding the per-position cell budget {budget}"
            ),
            RasterError::CorruptPosition(msg) => write!(f, "invalid raster position: {msg}"),
        }
    }
}

impl Error for RasterError {}

/// One model-ready HeXO position.
#[derive(Debug, Clone)]
pub struct RasterPosition {
    pub width: u16,
    pub height: u16,
    /// Global axial coordinate represented by local tensor cell `(0, 0)`.
    pub origin: Coord,
    /// Channel-first `NUM_PLANES × height × width` float32 data.
    pub planes: Vec<f32>,
    /// Graph-level features in [`Scalar`] order.
    pub scalars: [f32; NUM_SCALARS],
    /// One byte per cell; `1` for a stone or currently legal action.
    pub active_mask: Vec<u8>,
    /// One packed 30-bit incoming-ray word per cell.
    pub ray_bits: Vec<u32>,
    /// Sorted exactly like `GameState::legal_moves()`.
    pub legal_coords: Vec<Coord>,
    /// Local flattened tensor index corresponding to each `legal_coords` entry.
    pub legal_flat_indices: Vec<u32>,
}

impl RasterPosition {
    #[inline]
    pub fn cells(&self) -> usize {
        usize::from(self.width) * usize::from(self.height)
    }

    #[inline]
    pub fn plane(&self, plane: Plane) -> &[f32] {
        let cells = self.cells();
        let start = plane.index() * cells;
        &self.planes[start..start + cells]
    }

    #[inline]
    pub fn coord_to_index(&self, coord: Coord) -> Option<usize> {
        let x = coord.0 - self.origin.0;
        let y = coord.1 - self.origin.1;
        if x < 0 || y < 0 || x >= i32::from(self.width) || y >= i32::from(self.height) {
            return None;
        }
        Some(y as usize * usize::from(self.width) + x as usize)
    }

    #[inline]
    pub fn index_to_coord(&self, index: usize) -> Option<Coord> {
        if index >= self.cells() {
            return None;
        }
        let width = usize::from(self.width);
        let y = index / width;
        let x = index % width;
        Some((self.origin.0 + x as i32, self.origin.1 + y as i32))
    }

    /// Whether `cell` receives a line message from the requested source offset.
    #[inline]
    pub fn has_ray_source(&self, cell: usize, ray: usize, distance: usize) -> bool {
        if cell >= self.ray_bits.len()
            || ray >= NUM_RAYS
            || !(1..=PACKED_RAY_RADIUS).contains(&distance)
        {
            return false;
        }
        let bit = ray_bit(ray, distance);
        self.ray_bits[cell] & (1u32 << bit) != 0
    }

    /// Unpack to cell-major `[cell, ray, distance]` bytes for a simple wire path.
    pub fn unpack_ray_mask_u8(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.cells() * PACKED_RAY_BITS];
        for (cell, &bits) in self.ray_bits.iter().enumerate() {
            for bit in 0..PACKED_RAY_BITS {
                out[cell * PACKED_RAY_BITS + bit] = ((bits >> bit) & 1) as u8;
            }
        }
        out
    }

    pub fn validate(&self) -> Result<(), RasterError> {
        let cells = self.cells();
        if self.planes.len() != NUM_PLANES * cells {
            return Err(RasterError::CorruptPosition("plane length mismatch"));
        }
        if self.active_mask.len() != cells {
            return Err(RasterError::CorruptPosition("active-mask length mismatch"));
        }
        if self.ray_bits.len() != cells {
            return Err(RasterError::CorruptPosition("ray-bit length mismatch"));
        }
        if self.legal_coords.len() != self.legal_flat_indices.len() {
            return Err(RasterError::CorruptPosition(
                "legal coordinate/index mismatch",
            ));
        }
        if self.legal_flat_indices.iter().any(|&i| i as usize >= cells) {
            return Err(RasterError::CorruptPosition("legal index outside tensor"));
        }
        Ok(())
    }
}

/// A collection suitable for one inference request.
#[derive(Debug, Clone)]
pub struct RasterBatch {
    pub batch_size: u32,
    pub width: u16,
    pub height: u16,
    pub planes: Vec<f32>,
    pub scalars: Vec<f32>,
    pub active_mask: Vec<u8>,
    pub ray_bits: Vec<u32>,
    /// Prefix sums of legal counts, length `batch_size + 1`.
    pub legal_offsets: Vec<u32>,
    /// Indices into a flattened `[batch, height, width]` tensor.
    pub legal_flat_indices: Vec<u32>,
    pub origins: Vec<Coord>,
}

impl RasterBatch {
    pub fn from_positions(positions: &[RasterPosition]) -> Result<Self, RasterError> {
        if positions.is_empty() {
            return Err(RasterError::EmptyPosition);
        }
        let shape = positions
            .iter()
            .map(|position| (position.width, position.height))
            .fold((0, 0), |(max_width, max_height), (width, height)| {
                (max_width.max(width), max_height.max(height))
            });
        let cells = usize::from(shape.0) * usize::from(shape.1);
        let mut planes = Vec::with_capacity(positions.len() * NUM_PLANES * cells);
        let mut scalars = Vec::with_capacity(positions.len() * NUM_SCALARS);
        let mut active_mask = Vec::with_capacity(positions.len() * cells);
        let mut ray_bits = Vec::with_capacity(positions.len() * cells);
        let mut legal_offsets = Vec::with_capacity(positions.len() + 1);
        let mut legal_flat_indices = Vec::new();
        let mut origins = Vec::with_capacity(positions.len());
        legal_offsets.push(0);

        for (batch_index, p) in positions.iter().enumerate() {
            p.validate()?;
            let pad_x = usize::from(shape.0 - p.width) / 2;
            let pad_y = usize::from(shape.1 - p.height) / 2;
            let source_width = usize::from(p.width);
            let source_height = usize::from(p.height);
            let target_width = usize::from(shape.0);

            for plane in Plane::ALL {
                let mut padded = vec![0.0f32; cells];
                let source = p.plane(plane);
                for row in 0..source_height {
                    let source_start = row * source_width;
                    let target_start = (row + pad_y) * target_width + pad_x;
                    padded[target_start..target_start + source_width]
                        .copy_from_slice(&source[source_start..source_start + source_width]);
                }
                planes.extend_from_slice(&padded);
            }
            scalars.extend_from_slice(&p.scalars);

            let mut padded_active = vec![0u8; cells];
            let mut padded_rays = vec![0u32; cells];
            for row in 0..source_height {
                let source_start = row * source_width;
                let target_start = (row + pad_y) * target_width + pad_x;
                padded_active[target_start..target_start + source_width]
                    .copy_from_slice(&p.active_mask[source_start..source_start + source_width]);
                padded_rays[target_start..target_start + source_width]
                    .copy_from_slice(&p.ray_bits[source_start..source_start + source_width]);
            }
            active_mask.extend_from_slice(&padded_active);
            ray_bits.extend_from_slice(&padded_rays);
            origins.push((p.origin.0 - pad_x as i32, p.origin.1 - pad_y as i32));

            let batch_base = batch_index
                .checked_mul(cells)
                .ok_or(RasterError::CoordinateSpanOverflow)?;
            for &local in &p.legal_flat_indices {
                let local = local as usize;
                let source_y = local / source_width;
                let source_x = local % source_width;
                let padded_local = (source_y + pad_y)
                    .checked_mul(target_width)
                    .and_then(|row| row.checked_add(source_x + pad_x))
                    .ok_or(RasterError::CoordinateSpanOverflow)?;
                let global = batch_base
                    .checked_add(padded_local)
                    .ok_or(RasterError::CoordinateSpanOverflow)?;
                legal_flat_indices
                    .push(u32::try_from(global).map_err(|_| RasterError::CoordinateSpanOverflow)?);
            }
            legal_offsets.push(
                u32::try_from(legal_flat_indices.len())
                    .map_err(|_| RasterError::CoordinateSpanOverflow)?,
            );
        }

        Ok(Self {
            batch_size: u32::try_from(positions.len())
                .map_err(|_| RasterError::CoordinateSpanOverflow)?,
            width: shape.0,
            height: shape.1,
            planes,
            scalars,
            active_mask,
            ray_bits,
            legal_offsets,
            legal_flat_indices,
            origins,
        })
    }

    /// Encode a compact little-endian inference request.
    ///
    /// Layout (`HXR1`, version 1):
    ///
    /// ```text
    /// 4B magic, u16 version, u16 planes, u16 scalars, u16 width, u16 height,
    /// u16 packed_ray_radius, u32 batch, u32 total_legal,
    /// f32 planes[B,P,H,W], f32 scalars[B,S], u8 active[B,H,W],
    /// u32 ray_bits[B,H,W], u32 legal_offsets[B+1], u32 legal_flat[N],
    /// i32 origins[B,2]
    /// ```
    pub fn encode_hxr1(&self) -> Vec<u8> {
        const HEADER_BYTES: usize = 24;
        let mut out = Vec::with_capacity(
            HEADER_BYTES
                + self.planes.len() * 4
                + self.scalars.len() * 4
                + self.active_mask.len()
                + self.ray_bits.len() * 4
                + self.legal_offsets.len() * 4
                + self.legal_flat_indices.len() * 4
                + self.origins.len() * 8,
        );
        out.extend_from_slice(b"HXR1");
        push_u16(&mut out, 1);
        push_u16(&mut out, NUM_PLANES as u16);
        push_u16(&mut out, NUM_SCALARS as u16);
        push_u16(&mut out, self.width);
        push_u16(&mut out, self.height);
        push_u16(&mut out, PACKED_RAY_RADIUS as u16);
        push_u32(&mut out, self.batch_size);
        push_u32(&mut out, self.legal_flat_indices.len() as u32);
        for &v in &self.planes {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.scalars {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.active_mask);
        for &v in &self.ray_bits {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.legal_offsets {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.legal_flat_indices {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &(q, r) in &self.origins {
            out.extend_from_slice(&q.to_le_bytes());
            out.extend_from_slice(&r.to_le_bytes());
        }
        out
    }
}

#[derive(Debug, Clone, Copy)]
enum NodeKind {
    Stone(Player),
    Empty,
}

/// Return the bucketed dense dimensions without allocating raster planes.
///
/// This is deliberately cheap enough to use as a planning/guard pass before
/// materialising a batch. HeXO's placement radius is local to existing
/// stones, so a wandering game can have a very large sparse bounding box even
/// when it contains comparatively few active cells.
pub fn raster_dimensions(game: &GameState, spec: &RasterSpec) -> Result<(u16, u16), RasterError> {
    if game.is_terminal() {
        return Err(RasterError::TerminalPosition);
    }

    let stones = game.stones();
    let legal_coords = game.legal_moves();
    let first = stones
        .keys()
        .copied()
        .chain(legal_coords.iter().copied())
        .next()
        .ok_or(RasterError::EmptyPosition)?;
    let (mut min_q, mut max_q) = (first.0, first.0);
    let (mut min_r, mut max_r) = (first.1, first.1);
    for (q, r) in stones.keys().copied().chain(legal_coords.iter().copied()) {
        min_q = min_q.min(q);
        max_q = max_q.max(q);
        min_r = min_r.min(r);
        max_r = max_r.max(r);
    }

    let span_q_i64 = i64::from(max_q) - i64::from(min_q) + 1;
    let span_r_i64 = i64::from(max_r) - i64::from(min_r) + 1;
    let span_q = u16::try_from(span_q_i64).map_err(|_| RasterError::CoordinateSpanOverflow)?;
    let span_r = u16::try_from(span_r_i64).map_err(|_| RasterError::CoordinateSpanOverflow)?;
    Ok(choose_bucket(span_q, span_r, &spec.buckets))
}

/// Build a dense representation of a non-terminal game state.
pub fn build_raster(game: &GameState, spec: &RasterSpec) -> Result<RasterPosition, RasterError> {
    if game.is_terminal() {
        return Err(RasterError::TerminalPosition);
    }
    if spec.ray_radius == 0 || usize::from(spec.ray_radius) > PACKED_RAY_RADIUS {
        return Err(RasterError::InvalidRayRadius(spec.ray_radius));
    }
    if spec.win_length_scale <= 0.0 {
        return Err(RasterError::InvalidScale("win_length_scale"));
    }
    if spec.placement_radius_scale <= 0.0 {
        return Err(RasterError::InvalidScale("placement_radius_scale"));
    }

    let to_move = game.current_player().ok_or(RasterError::TerminalPosition)?;
    let stones = game.stones();
    let legal_coords = game.legal_moves();

    let mut all_coords = Vec::with_capacity(stones.len() + legal_coords.len());
    all_coords.extend(stones.keys().copied());
    all_coords.extend(legal_coords.iter().copied());
    let (&first, rest) = all_coords.split_first().ok_or(RasterError::EmptyPosition)?;
    let (mut min_q, mut max_q) = (first.0, first.0);
    let (mut min_r, mut max_r) = (first.1, first.1);
    for &(q, r) in rest {
        min_q = min_q.min(q);
        max_q = max_q.max(q);
        min_r = min_r.min(r);
        max_r = max_r.max(r);
    }

    let span_q_i64 = i64::from(max_q) - i64::from(min_q) + 1;
    let span_r_i64 = i64::from(max_r) - i64::from(min_r) + 1;
    let span_q = u16::try_from(span_q_i64).map_err(|_| RasterError::CoordinateSpanOverflow)?;
    let span_r = u16::try_from(span_r_i64).map_err(|_| RasterError::CoordinateSpanOverflow)?;
    let (width, height) = choose_bucket(span_q, span_r, &spec.buckets);

    let left_pad = (i32::from(width) - i32::from(span_q)) / 2;
    let top_pad = (i32::from(height) - i32::from(span_r)) / 2;
    let origin = (min_q - left_pad, min_r - top_pad);
    let cells = usize::from(width) * usize::from(height);

    let mut planes = vec![0.0f32; NUM_PLANES * cells];
    let mut active_mask = vec![0u8; cells];
    let mut kinds = vec![None::<NodeKind>; cells];

    let index_of = |coord: Coord| -> Option<usize> {
        let x = coord.0 - origin.0;
        let y = coord.1 - origin.1;
        if x < 0 || y < 0 || x >= i32::from(width) || y >= i32::from(height) {
            return None;
        }
        Some(y as usize * usize::from(width) + x as usize)
    };

    for (&coord, &player) in stones {
        let idx = index_of(coord).ok_or(RasterError::CoordinateSpanOverflow)?;
        active_mask[idx] = 1;
        kinds[idx] = Some(NodeKind::Stone(player));
        let plane = if player == to_move {
            Plane::OwnStone
        } else {
            Plane::OppStone
        };
        set_plane(&mut planes, cells, plane, idx, 1.0);
    }

    let stone_coords: Vec<Coord> = stones.keys().copied().collect();
    let mut legal_flat_indices = Vec::with_capacity(legal_coords.len());
    for &coord in &legal_coords {
        let idx = index_of(coord).ok_or(RasterError::CoordinateSpanOverflow)?;
        active_mask[idx] = 1;
        kinds[idx] = Some(NodeKind::Empty);
        set_plane(&mut planes, cells, Plane::Legal, idx, 1.0);
        let min_distance = stone_coords
            .iter()
            .map(|&stone| hex_distance(coord, stone))
            .min()
            .unwrap_or(1)
            .max(1);
        set_plane(
            &mut planes,
            cells,
            Plane::InvNearestStoneDistance,
            idx,
            1.0 / min_distance as f32,
        );
        legal_flat_indices
            .push(u32::try_from(idx).map_err(|_| RasterError::CoordinateSpanOverflow)?);
    }

    // Match the engine's existing threat feature implementation exactly.
    for (idx, kind) in kinds.iter().enumerate() {
        if kind.is_none() {
            continue;
        }
        let coord = (
            origin.0 + (idx % usize::from(width)) as i32,
            origin.1 + (idx / usize::from(width)) as i32,
        );
        let threat = node_threat_features(stones, coord, to_move, game.config().win_length);
        set_plane(&mut planes, cells, Plane::OwnMaxCleanLine, idx, threat[0]);
        set_plane(&mut planes, cells, Plane::OppMaxCleanLine, idx, threat[1]);
        set_plane(
            &mut planes,
            cells,
            Plane::OwnThreatAxisCount,
            idx,
            threat[2],
        );
        set_plane(
            &mut planes,
            cells,
            Plane::OppThreatAxisCount,
            idx,
            threat[3],
        );
    }

    let mut ray_bits = vec![0u32; cells];
    let effective_radius =
        usize::from(spec.ray_radius).min(usize::from(game.config().win_length.saturating_sub(1)));

    // This is the destination-gather equivalent of axis_graph.rs. Walk from
    // every active source; when a pair is admitted, set both directed incoming
    // bits. Bit-setting naturally deduplicates pairs discovered from both ends.
    for source_idx in 0..cells {
        let Some(source_kind) = kinds[source_idx] else {
            continue;
        };
        let source_coord = (
            origin.0 + (source_idx % usize::from(width)) as i32,
            origin.1 + (source_idx / usize::from(width)) as i32,
        );

        for (ray, &(dq, dr)) in RAY_DIRS.iter().enumerate() {
            for distance in 1..=effective_radius {
                let target_coord = (
                    source_coord.0 + dq * distance as i32,
                    source_coord.1 + dr * distance as i32,
                );
                let Some(target_idx) = index_of(target_coord) else {
                    break;
                };
                let Some(target_kind) = kinds[target_idx] else {
                    // Exact graph behaviour: a missing intermediate node ends
                    // the ray, even if a farther coordinate would be active.
                    break;
                };

                let both_empty = matches!(source_kind, NodeKind::Empty)
                    && matches!(target_kind, NodeKind::Empty);
                if !(spec.prune_empty_edges && both_empty) {
                    set_ray(&mut ray_bits, source_idx, ray, distance);
                    set_ray(&mut ray_bits, target_idx, opposite_ray(ray), distance);
                }

                let should_stop = match source_kind {
                    NodeKind::Stone(source_player) => {
                        matches!(target_kind, NodeKind::Stone(target_player) if target_player != source_player)
                    }
                    NodeKind::Empty => matches!(target_kind, NodeKind::Stone(_)),
                };
                if should_stop {
                    break;
                }
            }
        }
    }

    let config = game.config();
    let remaining = config.max_moves.saturating_sub(game.move_count());
    let mut scalars = [0.0f32; NUM_SCALARS];
    scalars[Scalar::MovesRemaining.index()] = game.moves_remaining_this_turn() as f32 / 2.0;
    scalars[Scalar::WinLength.index()] = config.win_length as f32 / spec.win_length_scale;
    scalars[Scalar::PlacementRadius.index()] =
        config.placement_radius as f32 / spec.placement_radius_scale;
    scalars[Scalar::RemainingBudget.index()] = remaining as f32 / config.max_moves as f32;
    scalars[Scalar::ToMoveIdentity.index()] = if to_move == Player::P1 { 1.0 } else { -1.0 };

    let result = RasterPosition {
        width,
        height,
        origin,
        planes,
        scalars,
        active_mask,
        ray_bits,
        legal_coords,
        legal_flat_indices,
    };
    result.validate()?;
    Ok(result)
}

#[inline]
pub const fn ray_bit(ray: usize, distance: usize) -> usize {
    ray * PACKED_RAY_RADIUS + (distance - 1)
}

#[inline]
pub const fn opposite_ray(ray: usize) -> usize {
    (ray + 3) % NUM_RAYS
}

#[inline]
fn set_ray(bits: &mut [u32], cell: usize, ray: usize, distance: usize) {
    debug_assert!(ray < NUM_RAYS);
    debug_assert!((1..=PACKED_RAY_RADIUS).contains(&distance));
    bits[cell] |= 1u32 << ray_bit(ray, distance);
}

#[inline]
fn set_plane(planes: &mut [f32], cells: usize, plane: Plane, cell: usize, value: f32) {
    planes[plane.index() * cells + cell] = value;
}

fn choose_bucket(span_q: u16, span_r: u16, buckets: &[(u16, u16)]) -> (u16, u16) {
    buckets
        .iter()
        .copied()
        .filter(|&(w, h)| w >= span_q && h >= span_r)
        .min_by_key(|&(w, h)| (u32::from(w) * u32::from(h), w, h))
        .unwrap_or_else(|| (round_1_plus_8k(span_q), round_1_plus_8k(span_r)))
}

#[inline]
fn round_1_plus_8k(n: u16) -> u16 {
    if n <= 1 {
        return 1;
    }
    let k = (u32::from(n - 1) + 7) / 8;
    (1 + 8 * k).min(u32::from(u16::MAX)) as u16
}

#[inline]
fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

#[inline]
fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}
