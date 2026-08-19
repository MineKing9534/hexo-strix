//! C ABI JSON wrapper for the standalone forcing solver.
//!
//! This crate intentionally exposes a tiny, stable boundary for JVM callers:
//! UTF-8 JSON in, heap-allocated UTF-8 JSON out. Kotlin/JNA must release returned
//! strings with [`hexo_free_string`].

use hexo_engine::types::Player;
use hexo_solver::forcing::Outcome;
use hexo_solver::{
    SolverEngine, SolverPosition, is_game_valid_board, solve_defense_from_position,
    solve_from_position, solve_wide_from_position,
};
use serde::{Deserialize, Serialize};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Duration;

#[derive(Debug, Deserialize)]
struct SolveRequest {
    win_length: u8,
    placement_radius: i32,
    max_moves: u32,
    to_move: u8,
    moves_remaining: u8,
    depth_cap: u8,
    node_budget: u64,
    #[serde(default)]
    wide: bool,
    #[serde(default)]
    time_limit_ms: Option<u64>,
    stones: Vec<StoneRequest>,
}

#[derive(Debug, Deserialize)]
struct StoneRequest {
    q: i32,
    r: i32,
    player: u8,
}

#[derive(Debug, Serialize)]
struct SolveResponse {
    kind: &'static str,
    depth: u8,
    pv: Vec<TurnResponse>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct CoordResponse {
    q: i32,
    r: i32,
}

#[derive(Debug, Serialize)]
struct TurnResponse {
    turn: u32,
    player: u8,
    cells: Vec<CoordResponse>,
}

#[derive(Debug, Serialize)]
struct DefenseResponse {
    kind: &'static str,
    threat: Option<ThreatResponse>,
    killers: Vec<CoordResponse>,
    pair_anchors: Vec<PairAnchorResponse>,
    best_delay: Option<CoordResponse>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct ThreatResponse {
    depth: u8,
    pv: Vec<TurnResponse>,
}

#[derive(Debug, Serialize)]
struct PairAnchorResponse {
    first: CoordResponse,
    second: CoordResponse,
}

/// Solve a position from a JSON request.
///
/// # Safety
///
/// `input` must either be null or point to a valid NUL-terminated C string for
/// the duration of the call. The returned pointer must be freed exactly once
/// with [`hexo_free_string`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hexo_solve_json(input: *const c_char) -> *mut c_char {
    let response = match catch_unwind(AssertUnwindSafe(|| solve_json(input))) {
        Ok(response) => response,
        Err(panic) => error_json(&format!("internal panic: {}", panic_message(panic))),
    };
    into_c_string(response)
}

/// Run defensive analysis for the side to move from a JSON request.
///
/// # Safety
///
/// `input` must either be null or point to a valid NUL-terminated C string for
/// the duration of the call. The returned pointer must be freed exactly once
/// with [`hexo_free_string`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hexo_solve_defense_json(input: *const c_char) -> *mut c_char {
    let response = match catch_unwind(AssertUnwindSafe(|| solve_defense_json(input))) {
        Ok(response) => response,
        Err(panic) => defense_error_json(&format!("internal panic: {}", panic_message(panic))),
    };
    into_c_string(response)
}

/// Free a string returned by [`hexo_solve_json`] or [`hexo_solve_defense_json`].
///
/// # Safety
///
/// `ptr` must be null or a pointer previously returned by [`hexo_solve_json`]
/// that has not already been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hexo_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            drop(CString::from_raw(ptr));
        }
    }
}

fn solve_json(input: *const c_char) -> String {
    let request = match parse_request(input) {
        Ok(request) => request,
        Err(err) => return error_json(&err),
    };

    match solve_request(request) {
        Ok(response) => to_json(&response),
        Err(err) => error_json(&err),
    }
}

fn solve_defense_json(input: *const c_char) -> String {
    let request = match parse_request(input) {
        Ok(request) => request,
        Err(err) => return defense_error_json(&err),
    };

    match defense_request(request) {
        Ok(response) => to_json(&response),
        Err(err) => defense_error_json(&err),
    }
}

fn parse_request(input: *const c_char) -> Result<SolveRequest, String> {
    if input.is_null() {
        return Err("input pointer is null".to_owned());
    }

    let input = unsafe { CStr::from_ptr(input) }
        .to_str()
        .map_err(|_| "input is not valid utf-8".to_owned())?;

    serde_json::from_str::<SolveRequest>(input)
        .map_err(|err| format!("invalid request json: {err}"))
}

fn solve_request(request: SolveRequest) -> Result<SolveResponse, String> {
    let wide = request.wide;
    let depth_cap = request.depth_cap;
    let node_budget = request.node_budget;
    let moves_remaining = request.moves_remaining;
    let attacker = request.to_move;
    let position = request.into_position()?;

    let outcome = if wide {
        solve_wide_from_position(&position, SolverEngine::Idtt, depth_cap, node_budget)
    } else {
        solve_from_position(&position, SolverEngine::Idtt, depth_cap, node_budget)
    };

    Ok(match outcome {
        Outcome::Win(win) => SolveResponse {
            kind: "win",
            depth: win.depth,
            pv: chunk_pv(win.pv, moves_remaining, attacker),
            error: None,
        },
        Outcome::No => SolveResponse {
            kind: "no",
            depth: 0,
            pv: Vec::new(),
            error: None,
        },
        Outcome::BudgetExceeded => SolveResponse {
            kind: "budget_exceeded",
            depth: 0,
            pv: Vec::new(),
            error: None,
        },
    })
}

fn defense_request(request: SolveRequest) -> Result<DefenseResponse, String> {
    let depth_cap = request.depth_cap;
    let node_budget = request.node_budget;
    let time_limit = Duration::from_millis(request.time_limit_ms.unwrap_or(10_000));
    let threat_attacker = opponent_u8(request.to_move)?;
    let position = request.into_position()?;

    if !is_game_valid_board(&position.stones) {
        return Err(
            "solve_defense requires a game-valid position: the (0,0,P1) origin stone must be present, with no duplicate or contradictory coords"
                .to_owned(),
        );
    }
    // Defense builds a GameState internally, which materializes the full move
    // neighborhood for the given placement radius. Keep the same native guard
    // as the wasm surface so untrusted callers cannot force an oversized
    // allocation.
    if !(1..=64).contains(&position.placement_radius) {
        return Err(
            "solve_defense requires 1 <= placement_radius <= 64"
                .to_owned(),
        );
    }

    match solve_defense_from_position(&position, depth_cap, node_budget, time_limit) {
        None => Ok(DefenseResponse {
            kind: "no_threat",
            threat: None,
            killers: Vec::new(),
            pair_anchors: Vec::new(),
            best_delay: None,
            error: None,
        }),
        Some(analysis) => Ok(DefenseResponse {
            kind: "threat_found",
            threat: Some(ThreatResponse {
                depth: analysis.threat_depth,
                pv: chunk_pv(analysis.threat_pv, 2, threat_attacker),
            }),
            killers: coords_response(analysis.killers),
            pair_anchors: analysis
                .pair_anchors
                .into_iter()
                .map(|(first, second)| PairAnchorResponse {
                    first: coord_response(first),
                    second: coord_response(second),
                })
                .collect(),
            best_delay: analysis.best_delay.map(coord_response),
            error: None,
        }),
    }
}

impl SolveRequest {
    fn into_position(self) -> Result<SolverPosition, String> {
        let to_move = parse_player(self.to_move, "to_move")?;
        let stones = self
            .stones
            .into_iter()
            .map(|stone| {
                Ok((
                    (stone.q, stone.r),
                    parse_player(stone.player, "stone.player")?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;

        Ok(SolverPosition {
            win_length: self.win_length,
            placement_radius: self.placement_radius,
            max_moves: self.max_moves,
            to_move,
            moves_remaining: self.moves_remaining,
            stones,
        })
    }
}

fn parse_player(value: u8, field: &str) -> Result<Player, String> {
    match value {
        1 => Ok(Player::P1),
        2 => Ok(Player::P2),
        _ => Err(format!("{field} must be 1 or 2")),
    }
}

fn coord_response((q, r): (i32, i32)) -> CoordResponse {
    CoordResponse { q, r }
}

fn coords_response(coords: Vec<(i32, i32)>) -> Vec<CoordResponse> {
    coords.into_iter().map(coord_response).collect()
}

/// Chunk the flat solver PV into wasm-compatible turn records.
///
/// Turn 0 belongs to the attacker and has `moves_remaining` cells. Every later
/// turn alternates player and has up to 2 cells.
fn chunk_pv(pv: Vec<(i32, i32)>, moves_remaining: u8, attacker: u8) -> Vec<TurnResponse> {
    if pv.is_empty() {
        return Vec::new();
    }

    let mut turns = Vec::new();
    let mut idx = 0usize;
    let mut turn = 0u32;
    let mut player = attacker;

    let first_len = (moves_remaining as usize).min(pv.len());
    turns.push(TurnResponse {
        turn,
        player,
        cells: coords_response(pv[idx..idx + first_len].to_vec()),
    });
    idx += first_len;
    turn += 1;
    player = opponent_u8(player).expect("validated player");

    while idx < pv.len() {
        let len = 2.min(pv.len() - idx);
        turns.push(TurnResponse {
            turn,
            player,
            cells: coords_response(pv[idx..idx + len].to_vec()),
        });
        idx += len;
        turn += 1;
        player = opponent_u8(player).expect("validated player");
    }

    turns
}

fn opponent_u8(player: u8) -> Result<u8, String> {
    match player {
        1 => Ok(2),
        2 => Ok(1),
        _ => Err("to_move must be 1 or 2".to_owned()),
    }
}

fn error_json(message: &str) -> String {
    to_json(&SolveResponse {
        kind: "error",
        depth: 0,
        pv: Vec::new(),
        error: Some(message.to_owned()),
    })
}

fn defense_error_json(message: &str) -> String {
    to_json(&DefenseResponse {
        kind: "error",
        threat: None,
        killers: Vec::new(),
        pair_anchors: Vec::new(),
        best_delay: None,
        error: Some(message.to_owned()),
    })
}

fn to_json<T: Serialize>(response: &T) -> String {
    serde_json::to_string(response).unwrap_or_else(|err| {
        format!(r#"{{"kind":"error","error":"response serialization failed: {err}"}}"#)
    })
}

fn into_c_string(response: String) -> *mut c_char {
    let sanitized = response.replace('\0', "\\u0000");
    CString::new(sanitized)
        .expect("NUL bytes were sanitized")
        .into_raw()
}

fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_owned())
}
