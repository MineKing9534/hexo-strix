//! Optional VCF evaluation at newly selected MCTS leaves.
//!
//! A proof for the leaf's side to move is an exact +1 value at that leaf. The
//! normal MCTS backup performs all player-perspective sign changes. `No` is
//! only "no VCF in the restricted forcing tree", and `BudgetExceeded` is
//! unresolved, so neither may replace the network value.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use hexo_engine::{GameState, Player};

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use super::forcing;

static CALLS: AtomicU64 = AtomicU64::new(0);
static WINS: AtomicU64 = AtomicU64::new(0);
static NO: AtomicU64 = AtomicU64::new(0);
static BUDGET_EXCEEDED: AtomicU64 = AtomicU64::new(0);
static ELAPSED_NS: AtomicU64 = AtomicU64::new(0);
static CAPTURED: AtomicU64 = AtomicU64::new(0);
static CAPTURE_ERRORS: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// Each MCTS game worker solves leaves synchronously, so thread-local
    /// scratch reuses allocations without synchronization or cross-game state.
    static SCRATCH: RefCell<forcing::VerdictScratch> =
        RefCell::new(forcing::VerdictScratch::default());
}

/// Process-wide counters for leaf-forcing experiments. They are deliberately
/// independent of MCTS results so production search structures stay compact.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub calls: u64,
    pub wins: u64,
    pub no: u64,
    pub budget_exceeded: u64,
    pub elapsed_ns: u64,
    pub captured: u64,
    pub capture_errors: u64,
}

/// One independent leaf evaluation within an already network-evaluated batch.
#[derive(Clone, Copy)]
pub struct Request<'a> {
    pub game: &'a GameState,
    pub network_value: f64,
    pub mcts_depth: usize,
    pub root_player: Player,
}

/// Apply leaf forcing to a whole evaluation batch, optionally in parallel.
///
/// Requests are independent and results retain input order. Rayon uses its
/// shared bounded pool; each worker receives its own thread-local solver
/// scratch. Small batches stay serial to avoid scheduling overhead.
pub fn override_batch(
    requests: &[Request<'_>],
    depth_cap: u8,
    node_budget: u64,
    parallel_min_batch: usize,
    tight: bool,
) -> Vec<f64> {
    let solve = |request: &Request<'_>| {
        override_value(
            request.game,
            request.network_value,
            depth_cap,
            node_budget,
            request.mcts_depth,
            request.root_player,
            tight,
        )
    };

    #[cfg(not(target_arch = "wasm32"))]
    if parallel_min_batch > 0 && requests.len() >= parallel_min_batch.max(2) {
        return requests.par_iter().map(solve).collect();
    }

    requests.iter().map(solve).collect()
}

/// Solve one leaf and return the value MCTS should back up.
///
/// The network value is returned unchanged unless a forced win is proven.
pub fn override_value(
    game: &GameState,
    network_value: f64,
    depth_cap: u8,
    node_budget: u64,
    mcts_depth: usize,
    root_player: Player,
    tight: bool,
) -> f64 {
    if node_budget == 0 {
        return network_value;
    }

    #[cfg(not(target_arch = "wasm32"))]
    let started = std::time::Instant::now();
    let verdict = SCRATCH.with(|scratch| {
        forcing::solve_verdict_with_scratch(
            game,
            depth_cap,
            node_budget,
            !tight,
            &mut scratch.borrow_mut(),
        )
    });
    #[cfg(not(target_arch = "wasm32"))]
    let elapsed_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
    #[cfg(target_arch = "wasm32")]
    let elapsed_ns = 0;

    CALLS.fetch_add(1, Ordering::Relaxed);
    ELAPSED_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
    let value = match verdict {
        forcing::ForcingVerdict::Win { .. } => {
            WINS.fetch_add(1, Ordering::Relaxed);
            1.0
        }
        forcing::ForcingVerdict::No => {
            NO.fetch_add(1, Ordering::Relaxed);
            network_value
        }
        forcing::ForcingVerdict::BudgetExceeded => {
            BUDGET_EXCEEDED.fetch_add(1, Ordering::Relaxed);
            network_value
        }
    };

    capture(
        game,
        verdict,
        depth_cap,
        node_budget,
        elapsed_ns,
        mcts_depth,
        root_player,
    );
    value
}

/// Read the aggregate counters. With `reset=true`, atomically starts a fresh
/// measurement window (individual fields may straddle a concurrent solve).
pub fn stats(reset: bool) -> Stats {
    let read = |counter: &AtomicU64| {
        if reset {
            counter.swap(0, Ordering::Relaxed)
        } else {
            counter.load(Ordering::Relaxed)
        }
    };
    Stats {
        calls: read(&CALLS),
        wins: read(&WINS),
        no: read(&NO),
        budget_exceeded: read(&BUDGET_EXCEEDED),
        elapsed_ns: read(&ELAPSED_NS),
        captured: read(&CAPTURED),
        capture_errors: read(&CAPTURE_ERRORS),
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod native_capture {
    use std::fs::{File, OpenOptions};
    use std::io::{self, BufWriter, Write};
    use std::path::Path;
    use std::sync::{Mutex, OnceLock};

    use hexo_engine::{GameState, Player};
    use serde_json::json;

    use super::{AtomicBool, CAPTURE_ERRORS, CAPTURED, Ordering, forcing};

    pub(super) static ENABLED: AtomicBool = AtomicBool::new(false);

    struct Capture {
        writer: BufWriter<File>,
        remaining: u64,
    }

    fn state() -> &'static Mutex<Option<Capture>> {
        static STATE: OnceLock<Mutex<Option<Capture>>> = OnceLock::new();
        STATE.get_or_init(|| Mutex::new(None))
    }

    pub(super) fn configure(path: &Path, limit: u64) -> io::Result<()> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let mut guard = state().lock().unwrap_or_else(|e| e.into_inner());
        *guard = Some(Capture {
            writer: BufWriter::new(file),
            remaining: limit,
        });
        ENABLED.store(limit > 0, Ordering::Release);
        Ok(())
    }

    pub(super) fn disable() -> io::Result<()> {
        ENABLED.store(false, Ordering::Release);
        let mut guard = state().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(capture) = guard.as_mut() {
            capture.writer.flush()?;
        }
        *guard = None;
        Ok(())
    }

    pub(super) fn record(
        game: &GameState,
        verdict: forcing::ForcingVerdict,
        depth_cap: u8,
        node_budget: u64,
        elapsed_ns: u64,
        mcts_depth: usize,
        root_player: Player,
    ) {
        if !ENABLED.load(Ordering::Acquire) {
            return;
        }
        let mut guard = state().lock().unwrap_or_else(|e| e.into_inner());
        let Some(capture) = guard.as_mut() else {
            return;
        };
        if capture.remaining == 0 {
            ENABLED.store(false, Ordering::Release);
            return;
        }

        let mut stones: Vec<_> = game
            .stones()
            .iter()
            .map(|(&(q, r), &p)| (q, r, p))
            .collect();
        stones.sort_unstable_by_key(|&(q, r, p)| (q, r, player_order(p)));
        let stones: Vec<_> = stones
            .into_iter()
            .map(|(q, r, p)| json!([q, r, player_tag(p)]))
            .collect();
        let (verdict, proof_depth) = match verdict {
            forcing::ForcingVerdict::Win { depth } => ("WIN", Some(depth)),
            forcing::ForcingVerdict::No => ("NO", None),
            forcing::ForcingVerdict::BudgetExceeded => ("BUDGET_EXCEEDED", None),
        };
        let Some(attacker) = game.current_player() else {
            return;
        };
        let cfg = game.config();
        let record = json!({
            "source": "mcts_leaf",
            "position": {
                "stones": stones,
                "attacker": player_tag(attacker),
                "placements_remaining": game.moves_remaining_this_turn(),
                "config": {
                    "win_length": cfg.win_length,
                    "placement_radius": cfg.placement_radius,
                    "max_moves": cfg.max_moves,
                },
            },
            "verdict": verdict,
            "depth": proof_depth,
            "depth_cap": depth_cap,
            "node_budget": node_budget,
            "elapsed_ns": elapsed_ns,
            "mcts_depth": mcts_depth,
            "root_player": player_tag(root_player),
            "leaf_player_same_as_root": attacker == root_player,
        });

        let result = match serde_json::to_writer(&mut capture.writer, &record) {
            Ok(()) => capture.writer.write_all(b"\n"),
            Err(err) => Err(std::io::Error::other(err)),
        };
        match result {
            Ok(()) => {
                capture.remaining -= 1;
                CAPTURED.fetch_add(1, Ordering::Relaxed);
                if capture.remaining == 0 {
                    let _ = capture.writer.flush();
                    ENABLED.store(false, Ordering::Release);
                }
            }
            Err(_) => {
                CAPTURE_ERRORS.fetch_add(1, Ordering::Relaxed);
                ENABLED.store(false, Ordering::Release);
            }
        }
    }

    fn player_order(player: Player) -> u8 {
        match player {
            Player::P1 => 1,
            Player::P2 => 2,
        }
    }

    fn player_tag(player: Player) -> &'static str {
        match player {
            Player::P1 => "P1",
            Player::P2 => "P2",
        }
    }
}

/// Append up to `limit` encountered leaf positions as replayable JSONL.
#[cfg(not(target_arch = "wasm32"))]
pub fn configure_capture(path: &std::path::Path, limit: u64) -> std::io::Result<()> {
    native_capture::configure(path, limit)
}

/// Flush and disable leaf capture.
#[cfg(not(target_arch = "wasm32"))]
pub fn finish_capture() -> std::io::Result<()> {
    native_capture::disable()
}

#[cfg(not(target_arch = "wasm32"))]
fn capture(
    game: &GameState,
    verdict: forcing::ForcingVerdict,
    depth_cap: u8,
    node_budget: u64,
    elapsed_ns: u64,
    mcts_depth: usize,
    root_player: Player,
) {
    native_capture::record(
        game,
        verdict,
        depth_cap,
        node_budget,
        elapsed_ns,
        mcts_depth,
        root_player,
    );
}

#[cfg(target_arch = "wasm32")]
fn capture(
    _game: &GameState,
    _verdict: forcing::ForcingVerdict,
    _depth_cap: u8,
    _node_budget: u64,
    _elapsed_ns: u64,
    _mcts_depth: usize,
    _root_player: Player,
) {
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::{Coord, GameConfig, Player};

    #[test]
    fn only_a_proven_win_overrides_the_network_value() {
        let stones: Vec<(Coord, Player)> = [
            (0, 0),
            (1, 0),
            (2, 0),
            (100, 100),
            (100, 101),
            (98, 104),
            (99, 103),
        ]
        .into_iter()
        .map(|c| (c, Player::P1))
        .collect();
        let win = GameState::from_state(&stones, Player::P1, 2, GameConfig::FULL_HEXO);
        assert_eq!(override_value(&win, -0.75, 6, 2_000, 3, Player::P1, false), 1.0);
        assert_eq!(override_value(&win, -0.75, 6, 2_000, 3, Player::P1, true), 1.0);

        let quiet = GameState::with_config(GameConfig::FULL_HEXO);
        assert_eq!(
            override_value(&quiet, -0.25, 6, 2_000, 1, Player::P2, false),
            -0.25,
        );
        assert_eq!(override_value(&win, 0.33, 6, 0, 1, Player::P1, false), 0.33);
    }

    #[test]
    fn parallel_batch_matches_serial_values_in_input_order() {
        let win_stones: Vec<(Coord, Player)> =
            [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
                .into_iter()
                .map(|c| (c, Player::P1))
                .collect();
        let win = GameState::from_state(
            &win_stones,
            Player::P1,
            2,
            GameConfig::FULL_HEXO,
        );
        let quiet = GameState::with_config(GameConfig::FULL_HEXO);
        let games = [&win, &quiet, &quiet, &win, &quiet, &win, &win, &quiet];
        let requests: Vec<_> = games
            .iter()
            .enumerate()
            .map(|(i, &game)| Request {
                game,
                network_value: -0.8 + i as f64 * 0.1,
                mcts_depth: i % 3 + 1,
                root_player: if i % 2 == 0 { Player::P1 } else { Player::P2 },
            })
            .collect();

        let serial = override_batch(&requests, 8, 500, 0, false);
        let parallel = override_batch(&requests, 8, 500, 2, false);
        assert_eq!(parallel, serial);
        assert_eq!(parallel[0], 1.0);
        assert_eq!(parallel[1], requests[1].network_value);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn capture_is_replayable_solver_jsonl() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "hexo-leaf-forcing-{}-{nonce}.jsonl",
            std::process::id(),
        ));
        configure_capture(&path, 1).unwrap();
        let game = GameState::with_config(GameConfig::FULL_HEXO);
        let _ = override_value(&game, 0.25, 6, 500, 4, Player::P1, false);
        finish_capture().unwrap();

        let line = std::fs::read_to_string(&path).unwrap();
        let record: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(record["source"], "mcts_leaf");
        assert_eq!(record["position"]["attacker"], "P2");
        assert_eq!(record["position"]["placements_remaining"], 2);
        assert_eq!(record["node_budget"], 500);
        assert_eq!(record["mcts_depth"], 4);
        assert_eq!(record["root_player"], "P1");
        assert_eq!(record["leaf_player_same_as_root"], false);
        assert!(record["position"]["stones"].is_array());
        std::fs::remove_file(path).unwrap();
    }
}
