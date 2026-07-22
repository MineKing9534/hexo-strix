//! Replay benchmark for verdict-only forcing probes captured at MCTS leaves.
//!
//! Usage:
//!   bench_leaf_forcing <capture.jsonl> [--depth-cap=N] [--node-budget=N]
//!       [--max=N] [--repeats=N] [--wide] [--legacy-order]
//!       [--incremental] [--compare-ordering] [--compare-incremental]

use hexo_engine::{GameConfig, GameState, Player};
use hexo_rs::mcts::forcing::{self, ForcingVerdict, VerdictScratch};
use serde_json::Value;
use std::env;
use std::fs::File;
use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::time::Instant;

#[derive(Clone)]
struct CapturedPosition {
    stones: Vec<((i32, i32), Player)>,
    attacker: Player,
    placements_remaining: u8,
    config: GameConfig,
    captured_verdict: String,
}

fn player(value: &Value) -> Result<Player, String> {
    match value.as_str() {
        Some("P1") => Ok(Player::P1),
        Some("P2") => Ok(Player::P2),
        other => Err(format!("bad player tag {other:?}")),
    }
}

fn parse_record(line: &str) -> Result<CapturedPosition, String> {
    let value: Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
    let position = value.get("position").ok_or("missing position")?;
    let stones = position["stones"]
        .as_array()
        .ok_or("position.stones must be an array")?
        .iter()
        .map(|stone| {
            let parts = stone.as_array().ok_or("stone must be [q,r,player]")?;
            if parts.len() != 3 {
                return Err("stone must contain exactly three elements".to_string());
            }
            let q = parts[0].as_i64().ok_or("stone q must be an integer")? as i32;
            let r = parts[1].as_i64().ok_or("stone r must be an integer")? as i32;
            Ok(((q, r), player(&parts[2])?))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let config = &position["config"];
    Ok(CapturedPosition {
        stones,
        attacker: player(&position["attacker"]).map_err(|e| e.to_string())?,
        placements_remaining: position["placements_remaining"]
            .as_u64()
            .ok_or("placements_remaining must be an integer")? as u8,
        config: GameConfig {
            win_length: config["win_length"].as_u64().ok_or("missing win_length")? as u8,
            placement_radius: config["placement_radius"]
                .as_i64()
                .ok_or("missing placement_radius")? as i32,
            max_moves: config["max_moves"].as_u64().ok_or("missing max_moves")? as u32,
        },
        captured_verdict: value["verdict"].as_str().unwrap_or("UNKNOWN").to_string(),
    })
}

fn load(path: &str, max: usize) -> Result<Vec<CapturedPosition>, String> {
    let file = File::open(path).map_err(|e| format!("opening {path}: {e}"))?;
    BufReader::new(file)
        .lines()
        .take(max)
        .enumerate()
        .map(|(i, line)| {
            let line = line.map_err(|e| format!("reading line {}: {e}", i + 1))?;
            parse_record(&line).map_err(|e| format!("parsing line {}: {e}", i + 1))
        })
        .collect()
}

fn game(position: &CapturedPosition) -> GameState {
    GameState::from_state(
        &position.stones,
        position.attacker,
        position.placements_remaining,
        position.config,
    )
}

fn percentile(sorted: &[u64], numerator: usize, denominator: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let index = ((sorted.len() - 1) * numerator + denominator / 2) / denominator;
    sorted[index]
}

fn tag(verdict: ForcingVerdict) -> &'static str {
    match verdict {
        ForcingVerdict::Win { .. } => "WIN",
        ForcingVerdict::No => "NO",
        ForcingVerdict::BudgetExceeded => "BUDGET_EXCEEDED",
    }
}

fn verdict_index(verdict: ForcingVerdict) -> usize {
    match verdict {
        ForcingVerdict::Win { .. } => 0,
        ForcingVerdict::No => 1,
        ForcingVerdict::BudgetExceeded => 2,
    }
}

fn solve_position(
    state: &GameState,
    depth_cap: u8,
    node_budget: u64,
    wide: bool,
    proof_ordering: bool,
    incremental: bool,
    scratch: &mut VerdictScratch,
) -> ForcingVerdict {
    if incremental {
        forcing::solve_verdict_with_scratch_incremental(
            state,
            depth_cap,
            node_budget,
            wide,
            proof_ordering,
            scratch,
        )
    } else if proof_ordering {
        forcing::solve_verdict_with_scratch_proof_ordered(
            state,
            depth_cap,
            node_budget,
            wide,
            scratch,
        )
    } else {
        forcing::solve_verdict_with_scratch(state, depth_cap, node_budget, wide, scratch)
    }
}

fn compare_ordering(positions: &[CapturedPosition], depth_cap: u8, node_budget: u64, wide: bool) {
    let mut legacy_scratch = VerdictScratch::default();
    let mut proof_scratch = VerdictScratch::default();
    let mut transitions = [[0usize; 3]; 3];
    let mut win_depth_changes = 0usize;
    for position in positions {
        let state = game(position);
        let legacy = forcing::solve_verdict_with_scratch(
            &state,
            depth_cap,
            node_budget,
            wide,
            &mut legacy_scratch,
        );
        let proof = forcing::solve_verdict_with_scratch_proof_ordered(
            &state,
            depth_cap,
            node_budget,
            wide,
            &mut proof_scratch,
        );
        transitions[verdict_index(legacy)][verdict_index(proof)] += 1;
        if let (ForcingVerdict::Win { depth: a }, ForcingVerdict::Win { depth: b }) =
            (legacy, proof)
            && a != b
        {
            win_depth_changes += 1;
        }
    }
    println!("ordering transition matrix (legacy rows -> proof columns)");
    println!("                 WIN       NO       BUDGET_EXCEEDED");
    for (label, row) in ["WIN", "NO", "BUDGET_EXCEEDED"]
        .into_iter()
        .zip(transitions)
    {
        println!("{label:>15} {:>8} {:>8} {:>21}", row[0], row[1], row[2]);
    }
    println!("both-win proof-depth changes={win_depth_changes}");
}

fn compare_incremental(
    positions: &[CapturedPosition],
    depth_cap: u8,
    node_budget: u64,
    wide: bool,
    proof_ordering: bool,
) {
    let mut legacy_scratch = VerdictScratch::default();
    let mut incremental_scratch = VerdictScratch::default();
    let mut transitions = [[0usize; 3]; 3];
    let mut win_depth_changes = 0usize;
    for position in positions {
        let state = game(position);
        let legacy = solve_position(
            &state,
            depth_cap,
            node_budget,
            wide,
            proof_ordering,
            false,
            &mut legacy_scratch,
        );
        let incremental = solve_position(
            &state,
            depth_cap,
            node_budget,
            wide,
            proof_ordering,
            true,
            &mut incremental_scratch,
        );
        transitions[verdict_index(legacy)][verdict_index(incremental)] += 1;
        if let (ForcingVerdict::Win { depth: a }, ForcingVerdict::Win { depth: b }) =
            (legacy, incremental)
            && a != b
        {
            win_depth_changes += 1;
        }
    }
    println!("incremental transition matrix (legacy rows -> incremental columns)");
    println!("                 WIN       NO       BUDGET_EXCEEDED");
    for (label, row) in ["WIN", "NO", "BUDGET_EXCEEDED"]
        .into_iter()
        .zip(transitions)
    {
        println!("{label:>15} {:>8} {:>8} {:>21}", row[0], row[1], row[2]);
    }
    println!("both-win proof-depth changes={win_depth_changes}");
}

fn main() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or(
        "usage: bench_leaf_forcing <capture.jsonl> [--depth-cap=N] [--node-budget=N] [--max=N] [--repeats=N] [--wide]",
    )?;
    let mut depth_cap = 16u8;
    let mut node_budget = 1_000u64;
    let mut max = usize::MAX;
    let mut repeats = 1usize;
    let mut wide = false;
    let mut proof_ordering = true;
    let mut incremental = false;
    let mut compare_order = false;
    let mut compare_windows = false;
    for arg in args {
        if let Some(value) = arg.strip_prefix("--depth-cap=") {
            depth_cap = value
                .parse()
                .map_err(|_| format!("bad depth cap {value:?}"))?;
        } else if let Some(value) = arg.strip_prefix("--node-budget=") {
            node_budget = value
                .parse()
                .map_err(|_| format!("bad node budget {value:?}"))?;
        } else if let Some(value) = arg.strip_prefix("--max=") {
            max = value.parse().map_err(|_| format!("bad max {value:?}"))?;
        } else if let Some(value) = arg.strip_prefix("--repeats=") {
            repeats = value
                .parse()
                .map_err(|_| format!("bad repeats {value:?}"))?;
        } else if arg == "--wide" {
            wide = true;
        } else if arg == "--legacy-order" {
            proof_ordering = false;
        } else if arg == "--incremental" {
            incremental = true;
        } else if arg == "--compare-ordering" {
            compare_order = true;
        } else if arg == "--compare-incremental" {
            compare_windows = true;
        } else {
            return Err(format!("unknown argument {arg:?}"));
        }
    }
    if repeats == 0 {
        return Err("--repeats must be positive".to_string());
    }

    let positions = load(&path, max)?;
    if positions.is_empty() {
        return Err("capture contains no positions".to_string());
    }
    let mut scratch = VerdictScratch::default();

    if compare_order {
        compare_ordering(&positions, depth_cap, node_budget, wide);
        return Ok(());
    }
    if compare_windows {
        compare_incremental(&positions, depth_cap, node_budget, wide, proof_ordering);
        return Ok(());
    }

    // Warm instruction/data caches without including setup in the measurement.
    for position in positions.iter().take(100) {
        let state = game(position);
        black_box(solve_position(
            &state,
            depth_cap,
            node_budget,
            wide,
            proof_ordering,
            incremental,
            &mut scratch,
        ));
    }

    let mut elapsed_ns = Vec::with_capacity(positions.len() * repeats);
    let mut wins = 0usize;
    let mut no = 0usize;
    let mut exceeded = 0usize;
    let mut captured_mismatches = 0usize;
    let started_all = Instant::now();
    for _ in 0..repeats {
        for position in &positions {
            let state = game(position);
            let started = Instant::now();
            let verdict = solve_position(
                &state,
                depth_cap,
                node_budget,
                wide,
                proof_ordering,
                incremental,
                &mut scratch,
            );
            elapsed_ns.push(started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
            match verdict {
                ForcingVerdict::Win { .. } => wins += 1,
                ForcingVerdict::No => no += 1,
                ForcingVerdict::BudgetExceeded => exceeded += 1,
            }
            if tag(verdict) != position.captured_verdict {
                captured_mismatches += 1;
            }
            black_box(verdict);
        }
    }
    let wall = started_all.elapsed();
    elapsed_ns.sort_unstable();
    let calls = elapsed_ns.len();
    let sum_ns: u128 = elapsed_ns.iter().map(|&x| x as u128).sum();
    println!("capture={path}");
    println!(
        "positions={} repeats={} calls={} width={} ordering={} windows={} depth_cap={} node_budget={}",
        positions.len(),
        repeats,
        calls,
        if wide { "wide" } else { "tight" },
        if proof_ordering { "proof" } else { "legacy" },
        if incremental { "incremental" } else { "legacy" },
        depth_cap,
        node_budget,
    );
    println!("verdicts: win={wins} no={no} budget_exceeded={exceeded}");
    println!("captured_verdict_mismatches={captured_mismatches}");
    println!(
        "solver_us: mean={:.1} p50={:.1} p95={:.1} p99={:.1}",
        sum_ns as f64 / calls as f64 / 1_000.0,
        percentile(&elapsed_ns, 50, 100) as f64 / 1_000.0,
        percentile(&elapsed_ns, 95, 100) as f64 / 1_000.0,
        percentile(&elapsed_ns, 99, 100) as f64 / 1_000.0,
    );
    println!(
        "wall_s={:.3} calls_per_s={:.1}",
        wall.as_secs_f64(),
        calls as f64 / wall.as_secs_f64()
    );
    Ok(())
}
