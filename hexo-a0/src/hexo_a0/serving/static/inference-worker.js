// Browser-local neural inference and Gumbel MCTS. Heavy synchronous WASM work
// stays off the UI thread; terminating this worker cancels an analysis.

let apiPromise;
let botPromise;

function versioned(relative) {
  const url = new URL(relative, import.meta.url);
  url.search = self.location.search;
  return url;
}

async function loadBot(modelUrl) {
  if (!apiPromise) {
    apiPromise = (async () => {
      const api = await import(versioned("./solver/hexo_wasm.js").href);
      await api.default({module_or_path: versioned("./solver/hexo_wasm_bg.wasm")});
      return api;
    })();
  }
  if (!botPromise) {
    botPromise = (async () => {
      const [api, response] = await Promise.all([apiPromise, fetch(modelUrl)]);
      if (!response.ok) throw new Error(`model download failed: HTTP ${response.status}`);
      const weights = new Uint8Array(await response.arrayBuffer());
      return new api.StrixBot(weights);
    })();
  }
  return botPromise;
}

function hasWin(stones, player, winLength) {
  const occupied = new Set(stones.filter(s => s.player === player).map(s => `${s.q},${s.r}`));
  for (const stone of stones) {
    if (stone.player !== player) continue;
    for (const [dq, dr] of [[1, 0], [0, 1], [1, -1]]) {
      if (occupied.has(`${stone.q-dq},${stone.r-dr}`)) continue;
      let count = 1;
      while (occupied.has(`${stone.q+dq*count},${stone.r+dr*count}`)) count++;
      if (count >= winLength) return true;
    }
  }
  return false;
}

function positionAt(moves, end, config) {
  const stones = [];
  for (let i = 0; i <= end; i++) {
    const player = i === 0 ? 1 : (Math.floor((i - 1) / 2) % 2 === 0 ? 2 : 1);
    stones.push({q: moves[i][0], r: moves[i][1], player});
  }
  const placements = Math.max(0, end);
  const toMove = Math.floor(placements / 2) % 2 === 0 ? 2 : 1;
  const movesRemaining = placements % 2 === 0 ? 2 : 1;
  return {config, stones, to_move: toMove, moves_remaining: movesRemaining};
}

function terminalEntry(position) {
  const winner = hasWin(position.stones, 1, position.config.win_length) ? "P1"
    : hasWin(position.stones, 2, position.config.win_length) ? "P2" : null;
  const draw = position.stones.length - 1 >= position.config.max_moves;
  if (!winner && !draw) return null;
  // Match serving.analysis._current_player_at: terminal GameState does not
  // expose a meaningful next mover, so report the opponent of the last mover.
  const lastMover = position.stones[position.stones.length - 1].player;
  const current = lastMover === 1 ? "P2" : "P1";
  return {
    value: winner ? (winner === current ? 1 : -1) : 0,
    current_player: current, terminal: true, winner, legal: [],
    stones: position.stones.map(s => [[s.q, s.r], s.player === 1 ? "P1" : "P2"]),
  };
}

function analysisEntry(result, position) {
  const visits = result.visit_counts || [];
  const total = visits.reduce((sum, n) => sum + n, 0);
  const searchProbs = total ? visits.map(n => n / total) : result.probs;
  return {
    value: result.value,
    current_player: position.to_move === 1 ? "P1" : "P2",
    terminal: false, winner: null,
    legal: result.legal, probs: searchProbs,
    q_hat: result.q_hat, improved_policy: result.improved_policy,
    candidate_set: result.candidate_set,
    stones: position.stones.map(s => [[s.q, s.r], s.player === 1 ? "P1" : "P2"]),
  };
}

function solverPosition(api, position) {
  return new api.Position(
    position.config.win_length, position.config.placement_radius, position.config.max_moves,
    position.to_move === 1 ? api.Player.P1 : api.Player.P2,
    position.moves_remaining,
    new Int32Array(position.stones.flatMap(s => [s.q, s.r, s.player])),
  );
}

function forcingResult(api, outcome, attackerIsMover, position) {
  if (outcome.kind !== api.SolveKind.Win) return null;
  const pv = [], pvOwners = [];
  for (const turn of outcome.pv) {
    const owner = turn.player === api.Player.P1 ? "P1" : "P2";
    for (const cell of turn.cells) {
      pv.push([cell.q, cell.r]);
      pvOwners.push(owner);
      cell.free();
    }
    turn.free();
  }
  const mover = position.to_move === 1 ? "P1" : "P2";
  const winner = attackerIsMover ? mover : (mover === "P1" ? "P2" : "P1");
  return {
    winner, attacker_is_mover: attackerIsMover,
    first_move: pv[0] || null, depth: outcome.depth || null,
    pv, line_placements: pv.length, pv_len: pv.length, pv_owners: pvOwners,
    wide: true, defense: null,
  };
}

function solveForcing(api, position, depthCap = 10, nodeBudget = 20000n) {
  let solver, solverPos, limits, outcome;
  try {
    solver = new api.StrixSolver();
    solverPos = solverPosition(api, position);
    limits = new api.SolverLimits(depthCap, nodeBudget, api.SolverEngineEnum.Idtt);
    outcome = solver.solve_wide(solverPos, limits);
    let result = forcingResult(api, outcome, true, position);
    outcome.free(); outcome = null;
    if (!result) {
      outcome = solver.solve_threat_wide(solverPos, limits);
      result = forcingResult(api, outcome, false, position);
    }
    return result;
  } catch (_error) {
    return null;
  } finally {
    try { outcome?.free(); } catch (_error) {}
    try { limits?.free(); } catch (_error) {}
    try { solverPos?.free(); } catch (_error) {}
    try { solver?.free(); } catch (_error) {}
  }
}

function forcingSettings(message) {
  const strength = message.strength || {};
  const depth = Number.isInteger(strength.forcingDepth) ? strength.forcingDepth : 10;
  const rawBudget = String(strength.forcingBudget || "20000");
  const budget = /^\d+$/.test(rawBudget) ? BigInt(rawBudget) : 20000n;
  return {depth, budget};
}

function annotateMissedWins(api, moves, positions, trajectory) {
  for (let i = 0; i + 1 < trajectory.length; i++) {
    const forcing = trajectory[i].forcing;
    if (!forcing?.attacker_is_mover || !moves[i + 1]) continue;
    if (forcing.first_move?.[0] === moves[i + 1][0] && forcing.first_move?.[1] === moves[i + 1][1]) continue;
    const mover = forcing.winner;
    let exempt = null;
    for (let j = i + 1; j < trajectory.length; j++) {
      const entry = trajectory[j];
      if (entry.terminal) {
        exempt = entry.winner === mover;
        break;
      }
      if (entry.current_player === mover) {
        let kept = entry.forcing;
        exempt = Boolean(kept?.attacker_is_mover && kept.winner === mover);
        if (!exempt) {
          kept = solveForcing(api, positions[j], 12, 250000n);
          if (kept?.attacker_is_mover && kept.winner === mover) {
            entry.forcing = kept;
            exempt = true;
          }
        }
        break;
      }
    }
    if (exempt !== false) continue;
    trajectory[i + 1].missed_win = {
      by: mover, at_prefix: i, first_move: forcing.first_move,
      depth: forcing.depth, pv: forcing.pv,
      line_placements: forcing.line_placements, pv_len: forcing.pv_len,
      pv_owners: forcing.pv_owners,
    };
  }
}

self.onmessage = async event => {
  const msg = event.data || {};
  const requestId = msg.requestId;
  let stage = "loading model";
  try {
    const bot = await loadBot(msg.modelUrl);
    const api = await apiPromise;
    if (msg.type === "bestMove") {
      stage = "local bot search";
      const result = JSON.parse(bot.best_move(JSON.stringify(msg.position), msg.sims, msg.mActions, 0n));
      self.postMessage({type: "bestMove", requestId, result});
      return;
    }
    if (msg.type === "analyzePosition") {
      stage = "local position analysis";
      const result = JSON.parse(bot.best_move(JSON.stringify(msg.position), msg.sims, msg.mActions, 0n));
      const entry = analysisEntry(result, msg.position);
      const settings = forcingSettings(msg);
      entry.forcing = solveForcing(api, msg.position, settings.depth, settings.budget);
      self.postMessage({type: "position", requestId, result: entry});
      return;
    }
    if (msg.type !== "analyzeGame") return;
    stage = "local trajectory analysis";
    const trajectory = [];
    const positions = [];
    const settings = forcingSettings(msg);
    for (let i = 0; i < msg.moves.length; i++) {
      const position = positionAt(msg.moves, i, msg.config);
      positions.push(position);
      const terminal = terminalEntry(position);
      stage = `local MCTS prefix ${i + 1}/${msg.moves.length}`;
      const entry = terminal || analysisEntry(
        JSON.parse(bot.best_move(JSON.stringify(position), msg.sims, msg.mActions, 0n)), position);
      if (!terminal) {
        stage = `local forcing prefix ${i + 1}/${msg.moves.length}`;
        entry.forcing = solveForcing(api, position, settings.depth, settings.budget);
      }
      trajectory.push(entry);
      self.postMessage({type: "progress", requestId, done: i + 1, total: msg.moves.length});
      if (terminal) break;
    }
    annotateMissedWins(api, msg.moves, positions, trajectory);
    const boundaryIndices = trajectory.map((entry, i) =>
      i === 0 || i === trajectory.length - 1 || entry.current_player !== trajectory[i - 1].current_player ? i : -1
    ).filter(i => i >= 0);
    self.postMessage({type: "game", requestId, result: {
      trajectory, boundary_indices: [...new Set(boundaryIndices)],
      evaluated_prefixes: trajectory.length, completed_prefixes: trajectory.length,
      mcts_sims: msg.sims, cancelled: false,
    }});
  } catch (error) {
    self.postMessage({type: "error", requestId,
      error: `${stage}: ${error?.message || String(error)}`});
  }
};
