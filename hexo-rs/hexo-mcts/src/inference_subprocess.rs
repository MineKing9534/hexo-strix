//! `SubprocessModel` — spawns a Python inference subprocess and communicates
//! via a binary protocol over stdin/stdout.

use rustc_hash::FxHashMap as HashMap;
use std::io::{BufRead, BufReader, BufWriter, Write as _};
use std::os::fd::AsRawFd;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, SystemTime};

/// Per-request inference payloads are several MB (features + edges for a
/// 60k-node batch ≈ 5 MB). Linux's default pipe buffer is 64 KB, forcing
/// ~80 read/write cycles per request. Bumping to 1 MB cuts this to ~5
/// cycles and removes the bulk of pipe-blocking overhead.
const TARGET_PIPE_SIZE: libc::c_int = 1 << 20; // 1 MB

fn try_resize_pipe(raw_fd: libc::c_int, label: &str) {
    // SAFETY: raw_fd is the kernel-side pipe FD owned by the child handle;
    // F_SETPIPE_SZ is the documented resize op. Kernel caps at
    // /proc/sys/fs/pipe-max-size; on failure we just log and keep the
    // default — never fatal.
    let rc = unsafe { libc::fcntl(raw_fd, libc::F_SETPIPE_SZ, TARGET_PIPE_SIZE) };
    if rc < 0 {
        let err = std::io::Error::last_os_error();
        eprintln!("warning: failed to resize {label} pipe to {TARGET_PIPE_SIZE} bytes: {err}");
    } else {
        eprintln!("inference subprocess {label} pipe sized to {rc} bytes");
    }
}

use hexo_engine::game::GameState;
use hexo_engine::types::{Coord, Player};

use crate::graph_tensors::GraphTensors;

const MAGIC: u32 = 0x48583034;
/// Protocol version. v2 added a `node_dim: u8` field to the forward-message
/// header (after `has_edge_attr`) so non-8-dim node features (e.g. 12-dim
/// threat features) survive the wire. Both sides always come from the same
/// checkout (the binary spawns the server), so no rolling compat is needed.
const VERSION: u8 = 2;
const MSG_FORWARD: u8 = 0x01;
const MSG_RELOAD: u8 = 0x02;
/// A3 state-payload wire: the client ships board states (~300 B/graph)
/// instead of graph tensors (~15 MB/request); the server rebuilds graphs.
/// Same VERSION — MSG_FORWARD bytes are untouched.
const MSG_FORWARD_STATES: u8 = 0x03;
const MSG_SHUTDOWN: u8 = 0xFF;

// --- MSG_FORWARD_STATES (A3) wire constants --------------------------------

/// builder_flags bits carried in the states request body. The server's
/// checkpoint-derived flags are authoritative; these are a cross-check —
/// a mismatch comes back as an in-band STATES_STATUS_ERROR (plan rev2 #1).
pub const BUILDER_FLAG_PRUNE: u8 = 0x01;
pub const BUILDER_FLAG_THREAT: u8 = 0x02;
pub const BUILDER_FLAG_RELATIVE: u8 = 0x04;

/// Status byte of every MSG_FORWARD_STATES-typed response.
pub const STATES_STATUS_OK: u8 = 0;
pub const STATES_STATUS_ERROR: u8 = 1;
pub const STATES_STATUS_PROBE_ACK: u8 = 2;

const FNV64_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
const FNV64_PRIME: u64 = 0x0000_0100_0000_01B3;

/// Sanity clamp on the length of an in-band STATES_STATUS_ERROR message. The
/// server writes human-readable validation strings (well under 1 KiB); a frame
/// claiming a larger length is either a protocol desync or a corrupt/hostile
/// peer, so we reject it instead of allocating an arbitrarily large buffer.
const MAX_STATES_ERROR_LEN: usize = 1 << 20; // 1 MiB

/// FNV-1a 64 over the legal-move coords, hashed as `(q: i32 LE, r: i32 LE)`
/// per move, in `legal_moves()` order. Must stay bit-identical to
/// `_fnv1a64` over the server's rebuilt legal coords (the order guard:
/// any client/server legal-ordering divergence is a loud hash mismatch).
pub fn legal_coords_fnv1a64(coords: &[Coord]) -> u64 {
    let mut h = FNV64_OFFSET;
    for &(q, r) in coords {
        for b in q.to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(FNV64_PRIME);
        }
        for b in r.to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(FNV64_PRIME);
        }
    }
    h
}

/// Canonical int32 HexKey (perf-plan §1): `(q << 16) | ((r ^ 0x8000) & 0xFFFF)`
/// — q sign-extended i16 in the high half (NOT biased), ONLY r biased, so
/// signed-i32 key order == lexicographic `(q, r)` order. Panics when either
/// coordinate falls outside the i16 wire range (client-side assert per the
/// wire revision; production coords are radius-bounded far below this).
pub fn pack_hexkey(q: i32, r: i32) -> i32 {
    assert!(
        (i16::MIN as i32..=i16::MAX as i32).contains(&q)
            && (i16::MIN as i32..=i16::MAX as i32).contains(&r),
        "--wire-states: coordinate ({q},{r}) does not fit the i16 HexKey wire range"
    );
    (q << 16) | ((r ^ 0x8000) & 0xFFFF)
}

/// Per-request geometry/schema fields of the MSG_FORWARD_STATES body header.
/// Sourced from the self_play run config (the same config that produced the
/// checkpoint), fixed for the lifetime of a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StatesWireConfig {
    pub win_length: u8,
    pub placement_radius: u8,
    pub max_moves: u32,
    /// BUILDER_FLAG_* bits (cross-check only; server checkpoint is authoritative).
    pub builder_flags: u8,
    /// Node-feature dim schema guard (must equal the server model's input width).
    pub node_dim: u8,
}

/// Board-state snapshot captured in the eval closure (where `&GameState` is
/// available). Carries `legal_coords` so response mapping NEVER rebuilds a
/// graph client-side (plan rev1 #6 / feasibility W3), and a precomputed
/// `edge_estimate` so the batcher loops can run edge-budget accumulation
/// without knowing the builder flags.
#[derive(Clone, Debug)]
pub struct StateSnapshot {
    /// P1 stones as canonical int32 HexKeys, sorted ascending (deterministic
    /// wire bytes; signed-key order == lexicographic (q, r) order).
    pub p1_keys: Vec<i32>,
    /// P2 stones as canonical int32 HexKeys, sorted ascending.
    pub p2_keys: Vec<i32>,
    /// 0 = P1, 1 = P2.
    pub current_player: u8,
    /// Placements left this turn; always 1 or 2 (terminal states never reach
    /// leaf eval and are rejected at capture).
    pub moves_remaining: u8,
    /// `legal_moves()` (engine-sorted) — equals the rebuilt graph's legal-node
    /// order, so server logits map back positionally.
    pub legal_coords: Vec<Coord>,
    /// Batching heuristic: edges ≈ nodes × (6 prune / 24 not) × 1.25 with
    /// nodes ≈ stones + legal + 1 (plan rev1 #9).
    pub edge_estimate: usize,
}

impl StateSnapshot {
    /// Capture a snapshot of a non-terminal position. Validates everything
    /// the wire format caps before send (plan rev1 #2 / #10): terminal
    /// states, moves_remaining ∉ {1,2}, u16 stone/legal-count overflow, and
    /// i16 coord overflow (inside [`pack_hexkey`]) all panic with clear
    /// messages — each is a client-side bug, never a recoverable condition.
    pub fn from_game(game: &GameState, prune_empty_edges: bool) -> Self {
        let current_player = match game.current_player() {
            Some(Player::P1) => 0u8,
            Some(Player::P2) => 1u8,
            None => panic!(
                "--wire-states: cannot snapshot a terminal state (terminal \
                 positions must never reach leaf eval)"
            ),
        };
        let moves_remaining = game.moves_remaining_this_turn();
        assert!(
            moves_remaining == 1 || moves_remaining == 2,
            "--wire-states: moves_remaining {moves_remaining} not in {{1, 2}}"
        );

        let mut p1_keys = Vec::new();
        let mut p2_keys = Vec::new();
        for (&(q, r), &player) in game.stones().iter() {
            let key = pack_hexkey(q, r);
            match player {
                Player::P1 => p1_keys.push(key),
                Player::P2 => p2_keys.push(key),
            }
        }
        // StoneMap iterates in hash order; sort for deterministic wire bytes
        // (the server consumes the stone SET, so any order is valid — sorted
        // keeps golden fixtures stable).
        p1_keys.sort_unstable();
        p2_keys.sort_unstable();

        let legal_coords = game.legal_moves();
        assert!(
            p1_keys.len() <= u16::MAX as usize && p2_keys.len() <= u16::MAX as usize,
            "--wire-states: stone count ({} P1 / {} P2) exceeds the u16 wire cap",
            p1_keys.len(),
            p2_keys.len(),
        );
        assert!(
            legal_coords.len() <= u16::MAX as usize,
            "--wire-states: legal-move count {} exceeds the u16 wire cap",
            legal_coords.len(),
        );

        let nodes = p1_keys.len() + p2_keys.len() + legal_coords.len() + 1;
        let per_node = if prune_empty_edges { 6 } else { 24 };
        let edge_estimate = nodes * per_node * 5 / 4; // ×1.25 safety margin

        StateSnapshot {
            p1_keys,
            p2_keys,
            current_player,
            moves_remaining,
            legal_coords,
            edge_estimate,
        }
    }

    /// Client-side legal-order hash, compared against the server's per-graph
    /// FNV-1a over its rebuilt legal coords.
    pub fn legal_hash(&self) -> u64 {
        legal_coords_fnv1a64(&self.legal_coords)
    }
}

/// Encode a complete MSG_FORWARD_STATES request (6-byte protocol header +
/// body). Layout (all little-endian) — must match the server's
/// `_read_forward_states_body` docstring exactly:
///
/// ```text
/// u32 magic | u8 version | u8 msg_type = 0x03
/// u32 num_graphs | u8 win_length | u8 placement_radius | u32 max_moves
/// u8 builder_flags | u8 node_dim
/// per graph:
///   u16 n_p1 | u16 n_p2 | u8 current_player | u8 moves_remaining
///   u16 num_legal | i32 keys[n_p1] | i32 keys[n_p2]
/// ```
///
/// A zero-snapshot request is the startup capability probe (server replies
/// STATES_STATUS_PROBE_ACK, bypassing all guards).
pub fn encode_forward_states_request(
    snapshots: &[StateSnapshot],
    cfg: &StatesWireConfig,
) -> Vec<u8> {
    let body_size: usize = 12
        + snapshots
            .iter()
            .map(|s| 8 + 4 * (s.p1_keys.len() + s.p2_keys.len()))
            .sum::<usize>();
    let mut buf = Vec::with_capacity(6 + body_size);

    buf.extend_from_slice(&MAGIC.to_le_bytes());
    buf.push(VERSION);
    buf.push(MSG_FORWARD_STATES);

    buf.extend_from_slice(&(snapshots.len() as u32).to_le_bytes());
    buf.push(cfg.win_length);
    buf.push(cfg.placement_radius);
    buf.extend_from_slice(&cfg.max_moves.to_le_bytes());
    buf.push(cfg.builder_flags);
    buf.push(cfg.node_dim);

    for s in snapshots {
        // The u16 wire caps are enforced at snapshot capture
        // ([`StateSnapshot::from_game`]); this guards direct encoder callers
        // (tests, future paths) against a silent truncating cast.
        debug_assert!(
            s.p1_keys.len() <= u16::MAX as usize
                && s.p2_keys.len() <= u16::MAX as usize
                && s.legal_coords.len() <= u16::MAX as usize,
            "encode_forward_states_request: stone/legal count exceeds the u16 wire cap"
        );
        buf.extend_from_slice(&(s.p1_keys.len() as u16).to_le_bytes());
        buf.extend_from_slice(&(s.p2_keys.len() as u16).to_le_bytes());
        buf.push(s.current_player);
        buf.push(s.moves_remaining);
        buf.extend_from_slice(&(s.legal_coords.len() as u16).to_le_bytes());
        for &k in &s.p1_keys {
            buf.extend_from_slice(&k.to_le_bytes());
        }
        for &k in &s.p2_keys {
            buf.extend_from_slice(&k.to_le_bytes());
        }
    }

    debug_assert_eq!(buf.len(), 6 + body_size);
    buf
}

/// Decoded MSG_FORWARD_STATES-typed response.
#[derive(Debug)]
pub enum StatesResponse {
    /// STATES_STATUS_OK: flattened logits over all legal moves plus
    /// per-graph legal counts, values, and legal-order FNV-1a hashes.
    Ok {
        logits: Vec<f32>,
        legal_counts: Vec<i32>,
        values: Vec<f32>,
        legal_hashes: Vec<u64>,
    },
    /// STATES_STATUS_ERROR: in-band server-side validation/rebuild failure.
    /// The server stays alive; the CLIENT must treat this as fatal.
    Error(String),
    /// STATES_STATUS_PROBE_ACK: capability ACK for a zero-graph probe.
    ProbeAck,
}

/// Read one MSG_FORWARD_STATES-typed response frame.
pub fn read_states_response(r: &mut impl std::io::Read) -> Result<StatesResponse, String> {
    let resp_magic = read_u32_le(r)?;
    if resp_magic != MAGIC {
        return Err(format!("bad magic in states response: 0x{resp_magic:08X}"));
    }
    let resp_ver = read_u8(r)?;
    if resp_ver != VERSION {
        return Err(format!("bad version in states response: {resp_ver}"));
    }
    let resp_type = read_u8(r)?;
    if resp_type != MSG_FORWARD_STATES {
        return Err(format!(
            "unexpected response type to MSG_FORWARD_STATES: 0x{resp_type:02X}"
        ));
    }
    let status = read_u8(r)?;
    match status {
        STATES_STATUS_OK => {
            let total_legal = read_u32_le(r)? as usize;
            let num_graphs = read_u32_le(r)? as usize;
            let mut logits = vec![0.0f32; total_legal];
            read_f32_slice(r, &mut logits)?;
            let mut legal_counts = vec![0i32; num_graphs];
            read_i32_slice(r, &mut legal_counts)?;
            let mut values = vec![0.0f32; num_graphs];
            read_f32_slice(r, &mut values)?;
            let mut legal_hashes = vec![0u64; num_graphs];
            read_u64_slice(r, &mut legal_hashes)?;
            // Reject negative counts via try_from rather than an `as usize`
            // cast (which would wrap a negative i32 into a huge usize and then
            // panic on slice-out-of-bounds far from the cause).
            let mut counted: usize = 0;
            for (i, &c) in legal_counts.iter().enumerate() {
                let c = usize::try_from(c).map_err(|_| {
                    format!("states response: negative legal_count {c} for graph {i}")
                })?;
                counted += c;
            }
            if counted != total_legal {
                return Err(format!(
                    "states response framing error: legal_counts sum {counted} != total_legal {total_legal}"
                ));
            }
            Ok(StatesResponse::Ok { logits, legal_counts, values, legal_hashes })
        }
        STATES_STATUS_ERROR => {
            let len = read_u32_le(r)? as usize;
            if len > MAX_STATES_ERROR_LEN {
                return Err(format!(
                    "states response: ERROR message length {len} exceeds the \
                     {MAX_STATES_ERROR_LEN}-byte cap (protocol desync or corrupt frame)"
                ));
            }
            let mut bytes = vec![0u8; len];
            read_exact(r, &mut bytes)?;
            Ok(StatesResponse::Error(String::from_utf8_lossy(&bytes).into_owned()))
        }
        STATES_STATUS_PROBE_ACK => Ok(StatesResponse::ProbeAck),
        other => Err(format!("unknown states response status: {other}")),
    }
}

/// Map a decoded MSG_FORWARD_STATES response onto per-snapshot `(policy, value)`
/// results. Split out of [`SubprocessModel::forward_states`] so every rejection
/// branch is unit-testable without a live subprocess:
///
/// * `ProbeAck` to a non-empty request — protocol bug;
/// * in-band `Error` — server-side validation/rebuild failure (fatal to
///   the client under `--wire-states`);
/// * graph-count mismatch — response carries a different number of graphs;
/// * per-graph legal-count mismatch — client/server builder divergence;
/// * legal-order hash mismatch — `legal_moves()` ordering diverged.
///
/// Logits map POSITIONALLY onto each snapshot's captured `legal_coords`; the
/// hash guard makes any ordering divergence a loud error before that mapping.
fn map_states_response(
    response: StatesResponse,
    snapshots: &[StateSnapshot],
) -> Result<(Vec<HashMap<Coord, f64>>, Vec<f64>), String> {
    match response {
        StatesResponse::ProbeAck => {
            Err("unexpected PROBE_ACK to a non-empty MSG_FORWARD_STATES request".into())
        }
        StatesResponse::Error(msg) => Err(format!(
            "inference server rejected states request (in-band ERROR): {msg}"
        )),
        StatesResponse::Ok { logits, legal_counts, values, legal_hashes } => {
            if values.len() != snapshots.len() {
                return Err(format!(
                    "graph count mismatch: sent {}, got {}",
                    snapshots.len(),
                    values.len()
                ));
            }
            let mut policies = Vec::with_capacity(snapshots.len());
            let mut logit_offset = 0usize;
            for (i, snap) in snapshots.iter().enumerate() {
                // Independent safety: reject a negative count via try_from rather
                // than an `as usize` cast (which would wrap into a huge usize and
                // panic on a slice access far from the cause). read_states_response
                // already screens these during framing, but map_states_response is
                // unit-tested in isolation, so it must not trust its input either.
                let count = usize::try_from(legal_counts[i]).map_err(|_| {
                    format!("graph {i}: negative server legal count {}", legal_counts[i])
                })?;
                if count != snap.legal_coords.len() {
                    return Err(format!(
                        "graph {i}: server legal count {count} != client legal count {} \
                         (client/server builder divergence)",
                        snap.legal_coords.len()
                    ));
                }
                let client_hash = snap.legal_hash();
                if client_hash != legal_hashes[i] {
                    return Err(format!(
                        "graph {i}: legal-order hash mismatch — client {client_hash:016x} != \
                         server {:016x}; legal_moves() ordering diverged between the client \
                         snapshot and the server rebuild",
                        legal_hashes[i]
                    ));
                }
                let mut policy = HashMap::with_capacity_and_hasher(count, Default::default());
                for (j, &coord) in snap.legal_coords.iter().enumerate() {
                    policy.insert(coord, logits[logit_offset + j] as f64);
                }
                logit_offset += count;
                policies.push(policy);
            }
            let values_f64: Vec<f64> = values.iter().map(|&v| v as f64).collect();
            Ok((policies, values_f64))
        }
    }
}

/// Encode a complete MSG_FORWARD request (6-byte protocol header + collated
/// graph body) from pre-built graph tensors. Split out of
/// [`SubprocessModel::forward_graphs`] so the exact wire bytes the live graph
/// path writes can be pinned by a cross-language golden fixture — the write is
/// otherwise inseparable from the pipe. `forward_graphs` calls this, so the
/// fixture covers the production encoder path, not a parallel copy.
///
/// Layout: header (20 bytes) + features (N×node_dim×4) + edge_src (E×8) +
/// edge_dst (E×8) + [edge_attr (E×5×4)] + legal_mask (N) + stone_mask (N) +
/// batch (N×4). Edge indices carry per-graph node offsets; batch indices are
/// i32 graph ids.
pub fn encode_forward_request(graphs: &[GraphTensors]) -> Vec<u8> {
    let num_graphs = graphs.len() as u32;
    let mut total_nodes: u32 = 0;
    let mut total_edges: u32 = 0;
    let has_edge_attr = graphs.first().map_or(false, |g| g.edge_attr.is_some());

    for g in graphs {
        total_nodes += g.num_nodes as u32;
        total_edges += g.num_edges as u32;
    }

    // Node-feature dim, derived per batch from the graph tensors. Batch
    // uniformity is guaranteed upstream (all graphs in a request come from
    // the same builder config); the debug_assert catches drift in tests.
    let node_dim = graphs
        .iter()
        .find(|g| g.num_nodes > 0)
        .map_or(8, |g| g.features.len() / g.num_nodes);
    debug_assert!(
        graphs.iter().all(|g| g.features.len() == g.num_nodes * node_dim),
        "node feature dim must be uniform across the batch"
    );

    let feat_bytes = total_nodes as usize * node_dim * 4;
    let edge_idx_bytes = total_edges as usize * 8 * 2; // src + dst
    let edge_attr_bytes = if has_edge_attr { total_edges as usize * 5 * 4 } else { 0 };
    let mask_bytes = total_nodes as usize * 2; // legal + stone
    let batch_bytes = total_nodes as usize * 4;
    let buf_size = 20 + feat_bytes + edge_idx_bytes + edge_attr_bytes + mask_bytes + batch_bytes;

    let mut buf = Vec::with_capacity(buf_size);

    // Header (20 bytes)
    buf.extend_from_slice(&MAGIC.to_le_bytes());
    buf.push(VERSION);
    buf.push(MSG_FORWARD);
    buf.extend_from_slice(&total_nodes.to_le_bytes());
    buf.extend_from_slice(&total_edges.to_le_bytes());
    buf.extend_from_slice(&num_graphs.to_le_bytes());
    buf.push(has_edge_attr as u8);
    buf.push(u8::try_from(node_dim).expect("node_dim exceeds u8"));

    // Features: copy each graph's feature slab as raw bytes
    for g in graphs {
        buf.extend_from_slice(as_u8_slice(&g.features));
    }

    // Edge src with node offsets applied
    let mut node_offset: i64 = 0;
    for g in graphs {
        for &src in &g.edge_src {
            buf.extend_from_slice(&(src + node_offset).to_le_bytes());
        }
        node_offset += g.num_nodes as i64;
    }

    // Edge dst with node offsets applied
    node_offset = 0;
    for g in graphs {
        for &dst in &g.edge_dst {
            buf.extend_from_slice(&(dst + node_offset).to_le_bytes());
        }
        node_offset += g.num_nodes as i64;
    }

    // Edge attr (if present) — raw byte copy, no per-graph offset
    if has_edge_attr {
        for g in graphs {
            if let Some(ref ea) = g.edge_attr {
                buf.extend_from_slice(as_u8_slice(ea));
            }
        }
    }

    // Legal mask + stone mask — pack bools as single bytes
    for g in graphs {
        for &m in &g.legal_mask {
            buf.push(m as u8);
        }
    }
    for g in graphs {
        for &m in &g.stone_mask {
            buf.push(m as u8);
        }
    }

    // Batch indices
    for (batch_idx, g) in graphs.iter().enumerate() {
        let idx = batch_idx as i32;
        let idx_bytes = idx.to_le_bytes();
        for _ in 0..g.num_nodes {
            buf.extend_from_slice(&idx_bytes);
        }
    }

    debug_assert_eq!(buf.len(), buf_size);
    buf
}

pub struct SubprocessModel {
    child: Child,
    #[allow(dead_code)]
    model_args: Vec<String>,
    #[allow(dead_code)]
    python_bin: String,
    model_mtime: Option<SystemTime>,
    stderr_handle: Option<std::thread::JoinHandle<()>>,
}

/// Build the `Command` used to spawn the inference subprocess, without
/// touching stdio/env — those are attached by the caller.
///
/// - `inference_bin = None`: spawns `python_bin -m hexo_a0.inference_server
///   --checkpoint <checkpoint> <extra_args...>` (the historical Python
///   server).
/// - `inference_bin = Some(bin)`: spawns `<bin> --checkpoint <checkpoint>
///   <extra_args...>` (no `-m`) — the native `hexo-infer-server` drop-in.
///   The extra args are forwarded unchanged; the Rust server ignores
///   whatever it doesn't need.
pub fn spawn_command(
    python_bin: &str,
    inference_bin: Option<&Path>,
    checkpoint: &Path,
    extra_args: &[String],
) -> Command {
    let mut cmd = match inference_bin {
        Some(bin) => {
            let mut cmd = Command::new(bin);
            cmd.arg("--checkpoint").arg(checkpoint);
            // These are set on self_play's own process to tame the
            // Python/torch subprocess's internal thread pool (torch/MKL
            // read OMP_NUM_THREADS at init). A native inference binary
            // never uses torch/MKL, but if it happens to link an
            // OpenMP-runtime library (e.g. libgomp, transitively), that
            // runtime reacts to these vars itself — calling
            // sched_setaffinity() to pin the whole process to a single
            // core — even though our Rust code never calls into OpenMP.
            // Scrub them from the child's env unconditionally so a native
            // binary never silently inherits Python-subprocess tuning.
            cmd.env_remove("OMP_NUM_THREADS");
            cmd.env_remove("MKL_NUM_THREADS");
            cmd.env_remove("OMP_PROC_BIND");
            cmd
        }
        None => {
            let mut cmd = Command::new(python_bin);
            cmd.args(["-m", "hexo_a0.inference_server", "--checkpoint"])
                .arg(checkpoint);
            cmd
        }
    };
    cmd.args(extra_args);
    cmd
}

impl SubprocessModel {
    /// Spawn the inference subprocess (stderr tagged `[python]`).
    ///
    /// Waits up to 600 seconds for the "READY" signal on stderr (torch.compile
    /// can take minutes on first run).
    pub fn spawn(
        python_bin: &str,
        inference_bin: Option<&Path>,
        model_path: &str,
        model_args: &[String],
    ) -> Result<Self, String> {
        Self::spawn_labeled(python_bin, inference_bin, model_path, model_args, "python")
    }

    /// Like [`spawn`](Self::spawn) but tags every stderr line `[{label}]`
    /// instead of `[python]`, so logs from multiple concurrent inference
    /// workers (e.g. `python w0`, `python w1`) are distinguishable. Keep
    /// `python` in the label so existing log greps still match.
    pub fn spawn_labeled(
        python_bin: &str,
        inference_bin: Option<&Path>,
        model_path: &str,
        model_args: &[String],
        label: &str,
    ) -> Result<Self, String> {
        let startup_label = label.to_string();
        let drain_label = label.to_string();
        let mut child = spawn_command(python_bin, inference_bin, Path::new(model_path), model_args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("failed to spawn inference subprocess: {e}"))?;

        // Bump both pipes to 1 MB before any traffic — per-request payloads
        // are multi-MB (features+edges for ~60k-node batches), so the 64 KB
        // default forces dozens of write/read syscalls per request.
        if let Some(stdin) = child.stdin.as_ref() {
            try_resize_pipe(stdin.as_raw_fd(), "rust→python (stdin)");
        }
        if let Some(stdout) = child.stdout.as_ref() {
            try_resize_pipe(stdout.as_raw_fd(), "python→rust (stdout)");
        }

        let stderr = child.stderr.take().expect("stderr was piped");

        // Wait for READY on stderr using thread + channel pattern.
        let (tx, rx) = mpsc::channel();
        let startup_thread = std::thread::spawn(move || {
            let mut reader = BufReader::new(stderr);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => break,  // EOF
                    Ok(_) => {
                        let trimmed = line.trim();
                        eprintln!("[{startup_label}] {trimmed}");
                        if trimmed == "READY" {
                            let _ = tx.send(Ok(reader));
                            return;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(format!("stderr read error: {e}")));
                        return;
                    }
                }
            }
            let _ = tx.send(Err("subprocess exited before sending READY".into()));
        });

        let reader = match rx.recv_timeout(Duration::from_secs(600)) {
            Ok(Ok(reader)) => reader,
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                // Timeout — kill the child and the thread
                let _ = child.kill();
                let _ = startup_thread.join();
                return Err("timed out waiting for READY from inference subprocess (600s)".into());
            }
        };

        // Spawn stderr drain thread
        let stderr_handle = std::thread::spawn(move || {
            let mut reader = reader;
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => eprintln!("[{drain_label}] {}", line.trim()),
                    Err(_) => break,
                }
            }
        });

        let model_mtime = std::fs::metadata(model_path)
            .and_then(|m| m.modified())
            .ok();

        Ok(SubprocessModel {
            child,
            model_args: model_args.to_vec(),
            python_bin: python_bin.to_string(),
            model_mtime,
            stderr_handle: Some(stderr_handle),
        })
    }

    /// Send a batch of graphs for inference and return (policy, value) results.
    ///
    /// Policy is returned as a map from `Coord` to logit for each graph.
    pub fn forward_graphs(
        &mut self,
        graphs: Vec<GraphTensors>,
    ) -> Result<(Vec<HashMap<Coord, f64>>, Vec<f64>), String> {
        if !self.is_alive() {
            return Err("inference subprocess is not running".into());
        }

        // Encode the exact wire bytes (header + body) via the pure encoder so
        // the live path and the cross-language golden fixture share one source
        // of truth (the write itself is inseparable from the pipe).
        let buf = encode_forward_request(&graphs);

        // Single write + flush
        let stdin = self.child.stdin.as_mut().expect("stdin was piped");
        stdin.write_all(&buf).map_err(|e| format!("write error: {e}"))?;
        stdin.flush().map_err(|e| format!("flush error: {e}"))?;

        // --- Read response from stdout ---
        let stdout = self.child.stdout.as_mut().expect("stdout was piped");

        let resp_magic = read_u32_le(stdout)?;
        if resp_magic != MAGIC {
            return Err(format!("bad magic in response: 0x{resp_magic:08X}"));
        }
        let resp_ver = read_u8(stdout)?;
        if resp_ver != VERSION {
            return Err(format!("bad version in response: {resp_ver}"));
        }
        let resp_type = read_u8(stdout)?;
        if resp_type != MSG_FORWARD {
            return Err(format!("unexpected response type: 0x{resp_type:02X}"));
        }

        let total_legal = read_u32_le(stdout)? as usize;
        let resp_num_graphs = read_u32_le(stdout)? as usize;
        if resp_num_graphs != graphs.len() {
            return Err(format!(
                "graph count mismatch: sent {}, got {resp_num_graphs}",
                graphs.len()
            ));
        }

        // Logits for all legal moves
        let mut logits = vec![0.0f32; total_legal];
        read_f32_slice(stdout, &mut logits)?;

        // Legal counts per graph
        let mut legal_counts = vec![0i32; resp_num_graphs];
        read_i32_slice(stdout, &mut legal_counts)?;

        // Values per graph
        let mut values = vec![0.0f32; resp_num_graphs];
        read_f32_slice(stdout, &mut values)?;

        // Map logits back to coordinates
        let mut policies = Vec::with_capacity(resp_num_graphs);
        let mut logit_offset = 0usize;
        for (i, g) in graphs.iter().enumerate() {
            let count = legal_counts[i] as usize;
            let mut policy = HashMap::with_capacity_and_hasher(count, Default::default());
            for j in 0..count {
                let coord = g.legal_coords[j];
                policy.insert(coord, logits[logit_offset + j] as f64);
            }
            logit_offset += count;
            policies.push(policy);
        }

        let values_f64: Vec<f64> = values.iter().map(|&v| v as f64).collect();

        Ok((policies, values_f64))
    }

    /// Send a batch of board-state snapshots (MSG_FORWARD_STATES, the A3
    /// wire) and return (policy, value) results in the same shape as
    /// [`forward_graphs`](Self::forward_graphs).
    ///
    /// Response mapping never rebuilds a graph: logits map positionally onto
    /// each snapshot's captured `legal_coords`, guarded by the per-graph
    /// legal-order FNV-1a hash from the server. ANY failure — in-band server
    /// ERROR status, hash mismatch, count mismatch — is returned as `Err`
    /// and must be treated as fatal by the caller (states-mode errors are
    /// never absorbed as empty logits; plan rev2 #6).
    pub fn forward_states(
        &mut self,
        snapshots: &[StateSnapshot],
        cfg: &StatesWireConfig,
    ) -> Result<(Vec<HashMap<Coord, f64>>, Vec<f64>), String> {
        if !self.is_alive() {
            return Err("inference subprocess is not running".into());
        }
        let buf = encode_forward_states_request(snapshots, cfg);

        let stdin = self.child.stdin.as_mut().expect("stdin was piped");
        stdin.write_all(&buf).map_err(|e| format!("write error: {e}"))?;
        stdin.flush().map_err(|e| format!("flush error: {e}"))?;

        let stdout = self.child.stdout.as_mut().expect("stdout was piped");
        let response = read_states_response(stdout)?;
        map_states_response(response, snapshots)
    }

    /// Startup capability probe (plan rev1 #4 / rev2 #4): send a zero-graph
    /// MSG_FORWARD_STATES request and require the dedicated PROBE_ACK. An
    /// EOF, garbage frame, or non-ACK reply means the server predates the
    /// states wire — the error message says so explicitly.
    pub fn probe_states(&mut self, cfg: &StatesWireConfig) -> Result<(), String> {
        if !self.is_alive() {
            return Err("inference subprocess is not running".into());
        }
        let buf = encode_forward_states_request(&[], cfg);
        let stdin = self.child.stdin.as_mut().expect("stdin was piped");
        stdin.write_all(&buf).map_err(|e| format!("write error: {e}"))?;
        stdin.flush().map_err(|e| format!("flush error: {e}"))?;

        let stdout = self.child.stdout.as_mut().expect("stdout was piped");
        match read_states_response(stdout) {
            Ok(StatesResponse::ProbeAck) => Ok(()),
            Ok(other) => Err(format!(
                "inference server too old for --wire-states: expected PROBE_ACK to the \
                 zero-graph MSG_FORWARD_STATES probe, got {other:?}"
            )),
            Err(e) => Err(format!(
                "inference server too old for --wire-states (no MSG_FORWARD_STATES \
                 support — EOF/unknown reply to the capability probe): {e}"
            )),
        }
    }

    /// Try to reload the model checkpoint if the file has been modified.
    /// Returns true if a reload was performed and acknowledged.
    pub fn try_reload(&mut self, path: &str) -> bool {
        let new_mtime = match std::fs::metadata(path).and_then(|m| m.modified()) {
            Ok(t) => t,
            Err(_) => return false,
        };

        if self.model_mtime == Some(new_mtime) {
            return false;
        }

        if !self.is_alive() {
            return false;
        }

        // Send reload message
        let stdin = match self.child.stdin.as_mut() {
            Some(s) => s,
            None => return false,
        };
        let mut w = BufWriter::new(stdin);
        let path_bytes = path.as_bytes();
        let path_len = path_bytes.len() as u32;

        if w.write_all(&MAGIC.to_le_bytes()).is_err()
            || w.write_all(&[VERSION, MSG_RELOAD]).is_err()
            || w.write_all(&path_len.to_le_bytes()).is_err()
            || w.write_all(path_bytes).is_err()
            || w.flush().is_err()
        {
            return false;
        }

        // Read ACK
        let stdout = match self.child.stdout.as_mut() {
            Some(s) => s,
            None => return false,
        };

        let magic = match read_u32_le(stdout) {
            Ok(m) => m,
            Err(_) => return false,
        };
        if magic != MAGIC {
            return false;
        }
        let ver = match read_u8(stdout) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if ver != VERSION {
            return false;
        }
        let msg_type = match read_u8(stdout) {
            Ok(t) => t,
            Err(_) => return false,
        };
        if msg_type != MSG_RELOAD {
            return false;
        }
        let success = match read_u8(stdout) {
            Ok(s) => s,
            Err(_) => return false,
        };

        if success != 0 {
            self.model_mtime = Some(new_mtime);
            true
        } else {
            false
        }
    }

    /// Check if the subprocess is still alive.
    fn is_alive(&mut self) -> bool {
        match self.child.try_wait() {
            Ok(Some(_)) => false,  // exited
            Ok(None) => true,      // still running
            Err(_) => false,
        }
    }
}

impl Drop for SubprocessModel {
    fn drop(&mut self) {
        // Send shutdown message
        if let Some(stdin) = self.child.stdin.as_mut() {
            let mut w = BufWriter::new(stdin);
            let _ = w.write_all(&MAGIC.to_le_bytes());
            let _ = w.write_all(&[VERSION, MSG_SHUTDOWN]);
            let _ = w.flush();
        }
        // Drop stdin to signal EOF
        self.child.stdin.take();

        // Wait up to 5 seconds for graceful exit
        let start = std::time::Instant::now();
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) => {
                    if start.elapsed() > Duration::from_secs(5) {
                        let _ = self.child.kill();
                        let _ = self.child.wait();
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(_) => break,
            }
        }

        // Join the stderr drain thread
        if let Some(handle) = self.stderr_handle.take() {
            let _ = handle.join();
        }
    }
}

// --- Wire-format reading helpers ---

fn read_exact(r: &mut impl std::io::Read, buf: &mut [u8]) -> Result<(), String> {
    r.read_exact(buf).map_err(|e| format!("read error: {e}"))
}

fn read_u8(r: &mut impl std::io::Read) -> Result<u8, String> {
    let mut buf = [0u8; 1];
    read_exact(r, &mut buf)?;
    Ok(buf[0])
}

fn read_u32_le(r: &mut impl std::io::Read) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    read_exact(r, &mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f32_slice(r: &mut impl std::io::Read, out: &mut [f32]) -> Result<(), String> {
    // Read as raw bytes, then convert
    let byte_len = out.len() * 4;
    let mut bytes = vec![0u8; byte_len];
    read_exact(r, &mut bytes)?;
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes(chunk.try_into().unwrap());
    }
    Ok(())
}

fn read_i32_slice(r: &mut impl std::io::Read, out: &mut [i32]) -> Result<(), String> {
    let byte_len = out.len() * 4;
    let mut bytes = vec![0u8; byte_len];
    read_exact(r, &mut bytes)?;
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = i32::from_le_bytes(chunk.try_into().unwrap());
    }
    Ok(())
}

fn read_u64_slice(r: &mut impl std::io::Read, out: &mut [u64]) -> Result<(), String> {
    let byte_len = out.len() * 8;
    let mut bytes = vec![0u8; byte_len];
    read_exact(r, &mut bytes)?;
    for (i, chunk) in bytes.chunks_exact(8).enumerate() {
        out[i] = u64::from_le_bytes(chunk.try_into().unwrap());
    }
    Ok(())
}

/// Reinterpret a `&[f32]` as `&[u8]` for zero-copy writes.
/// Safe on all platforms (f32 has alignment ≥ u8).
fn as_u8_slice(slice: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4)
    }
}

#[cfg(test)]
mod wire_states_tests {
    use super::*;
    use hexo_engine::game::GameConfig;

    fn test_cfg() -> StatesWireConfig {
        StatesWireConfig {
            win_length: 4,
            placement_radius: 4,
            max_moves: 50,
            builder_flags: BUILDER_FLAG_PRUNE | BUILDER_FLAG_RELATIVE,
            node_dim: 8,
        }
    }

    // --- FNV-1a 64 (values cross-checked against the Python server's _fnv1a64) ---

    #[test]
    fn fnv_empty_is_offset_basis() {
        assert_eq!(legal_coords_fnv1a64(&[]), 0xCBF2_9CE4_8422_2325);
    }

    #[test]
    fn fnv_known_vectors_match_python_server() {
        // _fnv1a64(struct.pack("<ii", 0, 0)) and ("<iiii", 1, -2, 3, 4).
        assert_eq!(legal_coords_fnv1a64(&[(0, 0)]), 0xA8C7_F832_281A_39C5);
        assert_eq!(legal_coords_fnv1a64(&[(1, -2), (3, 4)]), 0x2443_889E_E246_9E16);
    }

    #[test]
    fn fnv_is_order_sensitive() {
        assert_ne!(
            legal_coords_fnv1a64(&[(1, -2), (3, 4)]),
            legal_coords_fnv1a64(&[(3, 4), (1, -2)]),
            "the legal hash is an ORDER guard; permutations must not collide"
        );
    }

    // --- pack_hexkey -------------------------------------------------------

    #[test]
    fn pack_hexkey_matches_spec() {
        assert_eq!(pack_hexkey(0, 0), 0x8000);
        assert_eq!(pack_hexkey(-1, 2), 0xFFFF_8002u32 as i32);
        assert_eq!(pack_hexkey(2, -3), 0x0002_7FFD);
        assert_eq!(pack_hexkey(i16::MIN as i32, i16::MIN as i32), i32::MIN);
        assert_eq!(pack_hexkey(i16::MAX as i32, i16::MAX as i32), 0x7FFF_FFFF);
    }

    #[test]
    fn pack_hexkey_signed_order_is_lexicographic() {
        let coords = [(-3, 5), (-3, 6), (0, -1), (0, 0), (2, -7), (2, 0)];
        let keys: Vec<i32> = coords.iter().map(|&(q, r)| pack_hexkey(q, r)).collect();
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        assert_eq!(keys, sorted, "signed-key order must equal lexicographic (q, r) order");
    }

    #[test]
    #[should_panic(expected = "i16 HexKey wire range")]
    fn pack_hexkey_panics_out_of_i16_range() {
        pack_hexkey(40_000, 0);
    }

    // --- Encoder golden bytes ---------------------------------------------

    #[test]
    fn encode_request_golden_bytes() {
        let snap = StateSnapshot {
            p1_keys: vec![pack_hexkey(0, 0)],
            p2_keys: vec![pack_hexkey(-1, 2), pack_hexkey(2, -3)],
            current_player: 1,
            moves_remaining: 2,
            legal_coords: vec![(0, 1), (1, 0)],
            edge_estimate: 0,
        };
        let buf = encode_forward_states_request(&[snap], &test_cfg());
        #[rustfmt::skip]
        let expected: Vec<u8> = vec![
            // header
            0x34, 0x30, 0x58, 0x48,             // magic 0x48583034 LE
            0x02,                               // version
            0x03,                               // MSG_FORWARD_STATES
            // body header
            0x01, 0x00, 0x00, 0x00,             // num_graphs = 1
            0x04,                               // win_length
            0x04,                               // placement_radius
            0x32, 0x00, 0x00, 0x00,             // max_moves = 50
            0x05,                               // builder_flags (prune|relative)
            0x08,                               // node_dim
            // graph 0
            0x01, 0x00,                         // n_p1 = 1
            0x02, 0x00,                         // n_p2 = 2
            0x01,                               // current_player = P2
            0x02,                               // moves_remaining = 2
            0x02, 0x00,                         // num_legal = 2
            0x00, 0x80, 0x00, 0x00,             // key(0,0)   = 0x00008000
            0x02, 0x80, 0xFF, 0xFF,             // key(-1,2)  = 0xFFFF8002
            0xFD, 0x7F, 0x02, 0x00,             // key(2,-3)  = 0x00027FFD
        ];
        assert_eq!(buf, expected);
    }

    #[test]
    fn encode_zero_graph_probe_is_header_plus_body_header_only() {
        let buf = encode_forward_states_request(&[], &test_cfg());
        assert_eq!(buf.len(), 6 + 12);
        assert_eq!(&buf[..4], &MAGIC.to_le_bytes());
        assert_eq!(buf[4], VERSION);
        assert_eq!(buf[5], MSG_FORWARD_STATES);
        assert_eq!(&buf[6..10], &[0, 0, 0, 0], "num_graphs must be 0");
    }

    // --- Response decoder ---------------------------------------------------

    fn resp_header(status: u8) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.push(VERSION);
        b.push(MSG_FORWARD_STATES);
        b.push(status);
        b
    }

    #[test]
    fn decode_ok_response() {
        let mut b = resp_header(STATES_STATUS_OK);
        b.extend_from_slice(&3u32.to_le_bytes()); // total_legal
        b.extend_from_slice(&2u32.to_le_bytes()); // num_graphs
        for v in [0.5f32, -1.0, 2.0] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for c in [2i32, 1] {
            b.extend_from_slice(&c.to_le_bytes());
        }
        for v in [0.25f32, -0.75] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for h in [0xDEAD_BEEFu64, 0x1234_5678_9ABC_DEF0] {
            b.extend_from_slice(&h.to_le_bytes());
        }
        match read_states_response(&mut b.as_slice()).unwrap() {
            StatesResponse::Ok { logits, legal_counts, values, legal_hashes } => {
                assert_eq!(logits, vec![0.5, -1.0, 2.0]);
                assert_eq!(legal_counts, vec![2, 1]);
                assert_eq!(values, vec![0.25, -0.75]);
                assert_eq!(legal_hashes, vec![0xDEAD_BEEF, 0x1234_5678_9ABC_DEF0]);
            }
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn decode_ok_response_rejects_count_sum_mismatch() {
        let mut b = resp_header(STATES_STATUS_OK);
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes());
        for v in [0.5f32, -1.0, 2.0] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(&2i32.to_le_bytes()); // sums to 2 != 3
        b.extend_from_slice(&0.0f32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        let err = read_states_response(&mut b.as_slice()).unwrap_err();
        assert!(err.contains("legal_counts sum"), "got: {err}");
    }

    #[test]
    fn decode_error_response() {
        let msg = "wire builder_flags 0x00 != server checkpoint flags 0x01";
        let mut b = resp_header(STATES_STATUS_ERROR);
        b.extend_from_slice(&(msg.len() as u32).to_le_bytes());
        b.extend_from_slice(msg.as_bytes());
        match read_states_response(&mut b.as_slice()).unwrap() {
            StatesResponse::Error(m) => assert_eq!(m, msg),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[test]
    fn decode_error_response_rejects_oversized_message_len() {
        // A frame claiming an ERROR message far larger than any real validation
        // string must be rejected as a protocol error, not allocate that buffer.
        let mut b = resp_header(STATES_STATUS_ERROR);
        b.extend_from_slice(&((MAX_STATES_ERROR_LEN as u32) + 1).to_le_bytes());
        // No payload bytes follow: the length cap must fire before any read.
        let err = read_states_response(&mut b.as_slice()).unwrap_err();
        assert!(err.contains("exceeds the"), "got: {err}");
        assert!(err.contains("cap"), "got: {err}");
    }

    #[test]
    fn decode_error_response_accepts_len_at_cap() {
        // Exactly at the cap is still a valid (if implausibly large) message.
        let mut b = resp_header(STATES_STATUS_ERROR);
        b.extend_from_slice(&(MAX_STATES_ERROR_LEN as u32).to_le_bytes());
        b.resize(b.len() + MAX_STATES_ERROR_LEN, b'x');
        match read_states_response(&mut b.as_slice()).unwrap() {
            StatesResponse::Error(m) => assert_eq!(m.len(), MAX_STATES_ERROR_LEN),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[test]
    fn decode_probe_ack() {
        let b = resp_header(STATES_STATUS_PROBE_ACK);
        assert!(matches!(
            read_states_response(&mut b.as_slice()).unwrap(),
            StatesResponse::ProbeAck
        ));
    }

    #[test]
    fn decode_rejects_unknown_status_and_wrong_type() {
        let b = resp_header(7);
        assert!(read_states_response(&mut b.as_slice()).unwrap_err().contains("status"));

        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_le_bytes());
        b.push(VERSION);
        b.push(MSG_FORWARD); // graph-mode type in reply to a states request
        assert!(
            read_states_response(&mut b.as_slice())
                .unwrap_err()
                .contains("unexpected response type")
        );
    }

    #[test]
    fn decode_eof_is_an_error() {
        let b = &MAGIC.to_le_bytes()[..2];
        assert!(read_states_response(&mut &b[..]).is_err());
    }

    #[test]
    fn decode_ok_response_rejects_negative_legal_count() {
        // A negative i32 count must be rejected via try_from, not wrap into a
        // huge usize (which would panic on a slice access far from the cause).
        let mut b = resp_header(STATES_STATUS_OK);
        b.extend_from_slice(&0u32.to_le_bytes()); // total_legal
        b.extend_from_slice(&1u32.to_le_bytes()); // num_graphs
        // logits: none
        b.extend_from_slice(&(-1i32).to_le_bytes()); // legal_counts[0] = -1
        b.extend_from_slice(&0.0f32.to_le_bytes()); // values[0]
        b.extend_from_slice(&0u64.to_le_bytes()); // hashes[0]
        let err = read_states_response(&mut b.as_slice()).unwrap_err();
        assert!(err.contains("negative legal_count"), "got: {err}");
    }

    // --- map_states_response Err branches (fake-response, no live subprocess) ---

    fn snap_with_legal(legal: Vec<Coord>) -> StateSnapshot {
        StateSnapshot {
            p1_keys: vec![],
            p2_keys: vec![],
            current_player: 0,
            moves_remaining: 2,
            legal_coords: legal,
            edge_estimate: 0,
        }
    }

    /// Decode a fake OK-response byte stream then map it — exercises the same
    /// path `forward_states` runs after reading real stdout.
    fn read_then_map(
        bytes: Vec<u8>,
        snapshots: &[StateSnapshot],
    ) -> Result<(Vec<HashMap<Coord, f64>>, Vec<f64>), String> {
        let response = read_states_response(&mut bytes.as_slice()).unwrap();
        map_states_response(response, snapshots)
    }

    fn ok_response_bytes(
        logits: &[f32],
        counts: &[i32],
        values: &[f32],
        hashes: &[u64],
    ) -> Vec<u8> {
        let mut b = resp_header(STATES_STATUS_OK);
        b.extend_from_slice(&(logits.len() as u32).to_le_bytes());
        b.extend_from_slice(&(values.len() as u32).to_le_bytes());
        for &v in logits {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for &c in counts {
            b.extend_from_slice(&c.to_le_bytes());
        }
        for &v in values {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for &h in hashes {
            b.extend_from_slice(&h.to_le_bytes());
        }
        b
    }

    #[test]
    fn map_ok_response_maps_logits_onto_legal_coords() {
        let snap = snap_with_legal(vec![(0, 1), (1, 0)]);
        let hash = snap.legal_hash();
        let bytes = ok_response_bytes(&[0.5, -0.5], &[2], &[0.75], &[hash]);
        let (policies, values) = read_then_map(bytes, std::slice::from_ref(&snap)).unwrap();
        assert_eq!(values, vec![0.75]);
        assert_eq!(policies[0][&(0, 1)], 0.5);
        assert_eq!(policies[0][&(1, 0)], -0.5);
    }

    #[test]
    fn map_error_status_is_fatal_err() {
        let mut b = resp_header(STATES_STATUS_ERROR);
        let msg = "wire builder_flags 0x00 != server checkpoint flags 0x01";
        b.extend_from_slice(&(msg.len() as u32).to_le_bytes());
        b.extend_from_slice(msg.as_bytes());
        let err = read_then_map(b, &[snap_with_legal(vec![(0, 0)])]).unwrap_err();
        assert!(err.contains("in-band ERROR"), "got: {err}");
        assert!(err.contains(msg), "server message must be surfaced: {err}");
    }

    #[test]
    fn map_graph_count_mismatch_is_err() {
        // Response carries 2 graphs; we sent 1 snapshot.
        let snap = snap_with_legal(vec![(0, 0)]);
        let bytes = ok_response_bytes(&[0.0, 0.0], &[1, 1], &[0.0, 0.0], &[0, 0]);
        let err = read_then_map(bytes, std::slice::from_ref(&snap)).unwrap_err();
        assert!(err.contains("graph count mismatch"), "got: {err}");
    }

    #[test]
    fn map_per_graph_legal_count_mismatch_is_err() {
        // Server says 3 legal for a snapshot that captured 2.
        let snap = snap_with_legal(vec![(0, 1), (1, 0)]);
        let bytes = ok_response_bytes(&[0.0, 0.0, 0.0], &[3], &[0.0], &[0]);
        let err = read_then_map(bytes, std::slice::from_ref(&snap)).unwrap_err();
        assert!(err.contains("server legal count 3 != client legal count 2"), "got: {err}");
    }

    #[test]
    fn map_hash_mismatch_is_err() {
        // Counts agree, but the server's legal-order hash disagrees.
        let snap = snap_with_legal(vec![(0, 1), (1, 0)]);
        let wrong_hash = snap.legal_hash() ^ 0xFFFF_FFFF_FFFF_FFFF;
        let bytes = ok_response_bytes(&[0.0, 0.0], &[2], &[0.0], &[wrong_hash]);
        let err = read_then_map(bytes, std::slice::from_ref(&snap)).unwrap_err();
        assert!(err.contains("legal-order hash mismatch"), "got: {err}");
    }

    #[test]
    fn map_probe_ack_to_nonempty_is_err() {
        let err = map_states_response(
            StatesResponse::ProbeAck,
            &[snap_with_legal(vec![(0, 0)])],
        )
        .unwrap_err();
        assert!(err.contains("PROBE_ACK"), "got: {err}");
    }

    // --- StateSnapshot capture ----------------------------------------------

    fn small_config() -> GameConfig {
        GameConfig { win_length: 4, placement_radius: 4, max_moves: 50 }
    }

    #[test]
    fn snapshot_captures_stones_turn_and_legal_order() {
        let mut game = GameState::with_config(small_config());
        for &(q, r) in &[(1, 0), (0, 1), (2, 0)] {
            game.apply_move((q, r)).unwrap();
        }
        let snap = StateSnapshot::from_game(&game, false);

        // Stone sets round-trip (sorted keys on the wire).
        let mut expected_p1: Vec<i32> = game
            .stones()
            .iter()
            .filter(|&(_, &p)| p == Player::P1)
            .map(|(&(q, r), _)| pack_hexkey(q, r))
            .collect();
        expected_p1.sort_unstable();
        let mut expected_p2: Vec<i32> = game
            .stones()
            .iter()
            .filter(|&(_, &p)| p == Player::P2)
            .map(|(&(q, r), _)| pack_hexkey(q, r))
            .collect();
        expected_p2.sort_unstable();
        assert_eq!(snap.p1_keys, expected_p1);
        assert_eq!(snap.p2_keys, expected_p2);

        // Turn info.
        let expected_cur = match game.current_player().unwrap() {
            Player::P1 => 0u8,
            Player::P2 => 1u8,
        };
        assert_eq!(snap.current_player, expected_cur);
        assert_eq!(snap.moves_remaining, game.moves_remaining_this_turn());

        // Legal coords captured verbatim in legal_moves() order.
        assert_eq!(snap.legal_coords, game.legal_moves());
        assert_eq!(snap.legal_hash(), legal_coords_fnv1a64(&game.legal_moves()));
    }

    #[test]
    fn snapshot_edge_estimate_formula() {
        let game = GameState::with_config(small_config());
        let stones = game.stones().len();
        let legal = game.legal_moves().len();
        let nodes = stones + legal + 1;

        let dense = StateSnapshot::from_game(&game, false);
        assert_eq!(dense.edge_estimate, nodes * 24 * 5 / 4);
        let pruned = StateSnapshot::from_game(&game, true);
        assert_eq!(pruned.edge_estimate, nodes * 6 * 5 / 4);
    }

    #[test]
    #[should_panic(expected = "terminal")]
    fn snapshot_rejects_terminal_state() {
        let mut game = GameState::with_config(GameConfig {
            win_length: 2,
            placement_radius: 2,
            max_moves: 50,
        });
        // (0,0) is seeded P1; one adjacent P1 stone wins at win_length=2.
        // Play until terminal regardless of turn structure.
        while !game.is_terminal() {
            let mv = game.legal_moves()[0];
            game.apply_move(mv).unwrap();
        }
        let _ = StateSnapshot::from_game(&game, false);
    }

    // --- Cross-language golden fixture (Task 5) -----------------------------
    //
    // Writes/checks the MSG_FORWARD_STATES request byte-stream for fixed
    // positions against hexo-a0/tests/fixtures/wire_states_request.bin, plus
    // the client-side legal-order FNV hashes (wire_states_legal_hashes.bin).
    // The Python side (hexo-a0/tests/test_wire_states_fixture.py) parses the
    // request with `_read_forward_states_body` and asserts field equality +
    // that the server's rebuilt-legal hashes equal the client hashes.
    //
    // Regenerate (from the repo root) with:
    //   CARGO_TARGET_DIR=hexo-rs/target-a3 HEXO_REGEN_WIRE_FIXTURES=1 \
    //     cargo test -p hexo-mcts --manifest-path hexo-rs/Cargo.toml \
    //     wire_states_golden_fixture

    /// Fixed move list; MUST stay in sync with MOVES in
    /// hexo-a0/tests/test_wire_states_fixture.py.
    const FIXTURE_MOVES: [(i32, i32); 7] =
        [(1, 0), (0, 1), (2, 0), (1, 1), (-1, 2), (3, -1), (0, -2)];
    /// Snapshot after this many applied moves (mid-turn + both players).
    const FIXTURE_POINTS: [usize; 4] = [0, 1, 4, 7];

    fn fixture_snapshots() -> Vec<StateSnapshot> {
        let config = GameConfig { win_length: 6, placement_radius: 8, max_moves: 200 };
        FIXTURE_POINTS
            .iter()
            .map(|&k| {
                let mut game = GameState::with_config(config);
                for &(q, r) in &FIXTURE_MOVES[..k] {
                    game.apply_move((q, r)).unwrap();
                }
                assert!(!game.is_terminal());
                StateSnapshot::from_game(&game, false)
            })
            .collect()
    }

    #[test]
    fn wire_states_golden_fixture() {
        let snaps = fixture_snapshots();
        let cfg = StatesWireConfig {
            win_length: 6,
            placement_radius: 8,
            max_moves: 200,
            builder_flags: 0,
            node_dim: 8,
        };
        let request = encode_forward_states_request(&snaps, &cfg);
        let mut hashes = Vec::with_capacity(snaps.len() * 8);
        for s in &snaps {
            hashes.extend_from_slice(&s.legal_hash().to_le_bytes());
        }

        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../hexo-a0/tests/fixtures");
        let req_path = dir.join("wire_states_request.bin");
        let hash_path = dir.join("wire_states_legal_hashes.bin");

        if std::env::var_os("HEXO_REGEN_WIRE_FIXTURES").is_some() {
            std::fs::write(&req_path, &request).expect("write request fixture");
            std::fs::write(&hash_path, &hashes).expect("write hashes fixture");
            eprintln!("regenerated {} and {}", req_path.display(), hash_path.display());
            return;
        }

        let on_disk_req = std::fs::read(&req_path).unwrap_or_else(|e| {
            panic!(
                "missing golden fixture {} ({e}); regenerate with \
                 HEXO_REGEN_WIRE_FIXTURES=1 cargo test -p hexo-mcts wire_states_golden_fixture"
            , req_path.display())
        });
        let on_disk_hashes = std::fs::read(&hash_path).unwrap_or_else(|e| {
            panic!(
                "missing golden fixture {} ({e}); regenerate with \
                 HEXO_REGEN_WIRE_FIXTURES=1 cargo test -p hexo-mcts wire_states_golden_fixture"
            , hash_path.display())
        });
        assert_eq!(
            request, on_disk_req,
            "Rust MSG_FORWARD_STATES encoder no longer matches the checked-in golden \
             fixture; if the wire format changed intentionally, regenerate with \
             HEXO_REGEN_WIRE_FIXTURES=1 and update the Python fixture test"
        );
        assert_eq!(
            hashes, on_disk_hashes,
            "client legal-order FNV hashes no longer match the checked-in golden fixture"
        );
    }

    /// Nit D: a SECOND states golden fixture variant with all builder flags on
    /// (prune|threat|relative == 0x07) and node_dim 11 — pins the cross-check
    /// header bytes for the production lean schema. Stone keys / legal counts /
    /// legal hashes are flag-independent, so only the two header bytes differ
    /// from `wire_states_request.bin`; the Python fixture test rebuilds with the
    /// three flags on and asserts the server hashes still match.
    #[test]
    fn wire_states_golden_fixture_flags7() {
        let snaps = fixture_snapshots();
        let cfg = StatesWireConfig {
            win_length: 6,
            placement_radius: 8,
            max_moves: 200,
            builder_flags: BUILDER_FLAG_PRUNE | BUILDER_FLAG_THREAT | BUILDER_FLAG_RELATIVE,
            node_dim: 11,
        };
        let request = encode_forward_states_request(&snaps, &cfg);

        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../hexo-a0/tests/fixtures");
        let req_path = dir.join("wire_states_request_flags7.bin");

        if std::env::var_os("HEXO_REGEN_WIRE_FIXTURES").is_some() {
            std::fs::write(&req_path, &request).expect("write flags7 request fixture");
            eprintln!("regenerated {}", req_path.display());
            return;
        }

        let on_disk_req = std::fs::read(&req_path).unwrap_or_else(|e| {
            panic!(
                "missing golden fixture {} ({e}); regenerate with \
                 HEXO_REGEN_WIRE_FIXTURES=1 cargo test -p hexo-mcts wire_states_golden_fixture_flags7"
            , req_path.display())
        });
        assert_eq!(
            request, on_disk_req,
            "Rust MSG_FORWARD_STATES encoder (flags7/node_dim11) no longer matches the \
             checked-in golden fixture; regenerate with HEXO_REGEN_WIRE_FIXTURES=1"
        );
    }

    // --- Graph-mode cross-language golden fixture (TQ2-W1) ------------------
    //
    // Pins the MSG_FORWARD request byte-stream from the REAL client encoder
    // (`encode_forward_request`, which `forward_graphs` calls) for the same
    // fixed positions, against hexo-a0/tests/fixtures/wire_graph_request.bin.
    // The Python side (test_wire_states_fixture.py) parses it with
    // `_read_forward_body` and asserts the tensors equal the same positions
    // built via hexo_rs single-graph collation.
    //
    // Regenerate (from the repo root) with:
    //   CARGO_TARGET_DIR=hexo-rs/target-a3 HEXO_REGEN_WIRE_FIXTURES=1 \
    //     cargo test -p hexo-mcts --manifest-path hexo-rs/Cargo.toml \
    //     wire_graph_golden_fixture

    /// Dedicated (tight-radius) config for the graph fixture: the graph wire
    /// carries full features+edges, so the states fixture's radius-8 positions
    /// would bloat the committed .bin to ~1 MB. A radius-2 board keeps the pin
    /// in the KB range while still covering a multi-graph, mid-turn batch. MUST
    /// stay in sync with GRAPH_* in hexo-a0/tests/test_wire_states_fixture.py.
    const GRAPH_FIXTURE_MOVES: [(i32, i32); 5] = [(1, 0), (0, 1), (-1, 1), (1, -1), (0, -1)];
    const GRAPH_FIXTURE_POINTS: [usize; 4] = [0, 1, 3, 5];

    fn fixture_graphs() -> Vec<GraphTensors> {
        let config = GameConfig { win_length: 5, placement_radius: 2, max_moves: 200 };
        GRAPH_FIXTURE_POINTS
            .iter()
            .map(|&k| {
                let mut game = GameState::with_config(config);
                for &(q, r) in &GRAPH_FIXTURE_MOVES[..k] {
                    game.apply_move((q, r)).unwrap();
                }
                assert!(!game.is_terminal());
                // 8-dim absolute axis graph, no prune — matches the Python
                // `game_to_axis_graph_raw` default the fixture test rebuilds with.
                crate::graph_tensors::build_axis_graph_tensors_opts(&game, false)
            })
            .collect()
    }

    #[test]
    fn wire_graph_golden_fixture() {
        let graphs = fixture_graphs();
        let request = encode_forward_request(&graphs);

        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../hexo-a0/tests/fixtures");
        let req_path = dir.join("wire_graph_request.bin");

        if std::env::var_os("HEXO_REGEN_WIRE_FIXTURES").is_some() {
            std::fs::write(&req_path, &request).expect("write graph request fixture");
            eprintln!("regenerated {}", req_path.display());
            return;
        }

        let on_disk_req = std::fs::read(&req_path).unwrap_or_else(|e| {
            panic!(
                "missing golden fixture {} ({e}); regenerate with \
                 HEXO_REGEN_WIRE_FIXTURES=1 cargo test -p hexo-mcts wire_graph_golden_fixture"
            , req_path.display())
        });
        assert_eq!(
            request, on_disk_req,
            "Rust MSG_FORWARD encoder (forward_graphs path) no longer matches the \
             checked-in golden fixture; if the wire format changed intentionally, \
             regenerate with HEXO_REGEN_WIRE_FIXTURES=1 and update the Python fixture test"
        );
    }

    // --- probe_states negative path (R3T-W2): a server too old for the states
    // wire (no MSG_FORWARD_STATES support) must surface a clear "too old"
    // message. Spawns a fake old server (dead_stub_model's script pattern) that
    // reaches READY and replies to the probe with a pre-canned response frame.

    /// Spawn a stub "inference server" that prints READY, writes `resp_bytes`
    /// to stdout as its reply to the FIRST request, then stays alive (so the
    /// client's `is_alive()` check and read both see a live process). Mirrors
    /// `dead_stub_model` in self_play.rs.
    #[cfg(unix)]
    fn old_server_stub(resp_bytes: &[u8]) -> SubprocessModel {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!(
            "hexo_old_stub_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let resp_path = dir.join("resp.bin");
        std::fs::write(&resp_path, resp_bytes).unwrap();
        let script = dir.join("stub.sh");
        // `cat` flushes the canned frame when it closes stdout; `sleep` keeps
        // the script (and thus the stdout pipe) alive past the client's read.
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\necho READY >&2\ncat '{}'\nsleep 30\n",
                resp_path.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        SubprocessModel::spawn_labeled("unused-python", Some(&script), "unused.pt", &[], "oldstub")
            .expect("old-server stub should reach READY")
    }

    #[cfg(unix)]
    #[test]
    fn probe_states_reports_too_old_on_msg_forward_reply() {
        // An old build answers the zero-graph probe with a MSG_FORWARD-typed
        // frame (it never learned MSG_FORWARD_STATES).
        let mut reply = Vec::new();
        reply.extend_from_slice(&MAGIC.to_le_bytes());
        reply.push(VERSION);
        reply.push(MSG_FORWARD);
        let mut model = old_server_stub(&reply);
        let err = model.probe_states(&test_cfg()).unwrap_err();
        assert!(err.contains("too old"), "got: {err}");
    }

    #[cfg(unix)]
    #[test]
    fn probe_states_reports_too_old_on_garbage_reply() {
        // A garbage frame (bad magic) to the probe is also read as "too old".
        let mut model = old_server_stub(&[0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00]);
        let err = model.probe_states(&test_cfg()).unwrap_err();
        assert!(err.contains("too old"), "got: {err}");
    }
}

#[cfg(test)]
mod spawn_command_tests {
    use super::*;

    fn args_of(cmd: &Command) -> Vec<String> {
        cmd.get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn python_path_spawns_module_with_no_inference_bin() {
        let checkpoint = Path::new("/tmp/model.pt");
        let extra = vec!["--graph-type".to_string(), "axis".to_string()];
        let cmd = spawn_command("python3", None, checkpoint, &extra);

        assert_eq!(cmd.get_program(), "python3");
        let args = args_of(&cmd);
        assert_eq!(
            args,
            vec![
                "-m",
                "hexo_a0.inference_server",
                "--checkpoint",
                "/tmp/model.pt",
                "--graph-type",
                "axis",
            ]
        );
    }

    #[test]
    fn inference_bin_spawns_native_binary_directly() {
        let checkpoint = Path::new("/tmp/model.pt");
        let extra = vec!["--graph-type".to_string(), "axis".to_string()];
        let bin = Path::new("/usr/local/bin/hexo-infer-server");
        let cmd = spawn_command("python3", Some(bin), checkpoint, &extra);

        assert_eq!(cmd.get_program(), bin.as_os_str());
        let args = args_of(&cmd);
        assert_eq!(
            args,
            vec!["--checkpoint", "/tmp/model.pt", "--graph-type", "axis"]
        );
    }

    #[test]
    fn inference_bin_omits_python_module_flag() {
        let checkpoint = Path::new("/tmp/model.pt");
        let bin = Path::new("/usr/local/bin/hexo-infer-server");
        let cmd = spawn_command("python3", Some(bin), checkpoint, &[]);

        let args = args_of(&cmd);
        assert!(!args.iter().any(|a| a == "-m"));
        assert!(!args.iter().any(|a| a == "hexo_a0.inference_server"));
    }

    /// `Command::get_envs()` yields `(key, None)` for a key explicitly
    /// cleared via `env_remove`, distinct from a key never mentioned at all
    /// (which simply doesn't appear in the iterator). This lets us assert
    /// the *constructed* `Command`'s env overrides directly, without
    /// actually spawning a child and inspecting its real environment.
    fn is_env_removed(cmd: &Command, key: &str) -> bool {
        cmd.get_envs()
            .any(|(k, v)| k.to_string_lossy() == key && v.is_none())
    }

    #[test]
    fn inference_bin_scrubs_openmp_env_vars() {
        let checkpoint = Path::new("/tmp/model.pt");
        let bin = Path::new("/usr/local/bin/hexo-infer-server");
        let cmd = spawn_command("python3", Some(bin), checkpoint, &[]);

        for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OMP_PROC_BIND"] {
            assert!(
                is_env_removed(&cmd, key),
                "native inference_bin Command must env_remove {key}"
            );
        }
    }

    #[test]
    fn python_path_leaves_openmp_env_vars_untouched() {
        let checkpoint = Path::new("/tmp/model.pt");
        let cmd = spawn_command("python3", None, checkpoint, &[]);

        // The Python branch must not add any env overrides at all — it
        // relies on inheriting the parent's OMP/MKL tuning vars unchanged.
        assert_eq!(
            cmd.get_envs().count(),
            0,
            "python branch Command must have zero env overrides"
        );
        for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OMP_PROC_BIND"] {
            assert!(
                !is_env_removed(&cmd, key),
                "python branch must not env_remove {key}"
            );
        }
    }
}
