// Cancellable browser-side bridge to the standalone Strix forced-win solver.
// Search is synchronous inside WASM, so cancellation is implemented by the UI
// terminating this worker; the main thread and the serving host remain free.

let apiPromise;

function versioned(relative) {
  const url = new URL(relative, import.meta.url);
  url.search = self.location.search;
  return url;
}

async function loadApi() {
  if (!apiPromise) {
    apiPromise = (async () => {
      const api = await import(versioned("./solver/hexo_wasm.js").href);
      await api.default({module_or_path: versioned("./solver/hexo_wasm_bg.wasm")});
      return api;
    })();
  }
  return apiPromise;
}

function engineValue(api, engine) {
  const engines = {
    idtt: api.SolverEngineEnum.Idtt,
    pdspn: api.SolverEngineEnum.Pdspn,
    "pdspn-shortest": api.SolverEngineEnum.Pdspn,
  };
  if (!(engine in engines)) throw new Error(`Unknown solver engine: ${engine}`);
  return engines[engine];
}

function playerValue(api, player) {
  if (player === "P1") return api.Player.P1;
  if (player === "P2") return api.Player.P2;
  throw new Error(`Invalid side to move: ${player}`);
}

function playerName(api, player) {
  return player === api.Player.P1 ? "P1" : "P2";
}

function kindName(api, kind) {
  if (kind === api.SolveKind.Win) return "win";
  if (kind === api.SolveKind.No) return "no";
  return "budget_exceeded";
}

function serializePv(api, outcome) {
  const turnsOut = [];
  const placements = [];
  const owners = [];
  const turns = outcome.pv;
  for (const turn of turns) {
    const turnNumber = turn.turn;
    const owner = playerName(api, turn.player);
    const cellsOut = [];
    const cells = turn.cells;
    try {
      for (const cell of cells) {
        const coord = [cell.q, cell.r];
        cellsOut.push(coord);
        placements.push(coord);
        owners.push(owner);
        cell.free();
      }
    } finally {
      turn.free();
    }
    turnsOut.push({turn: turnNumber, player: owner, cells: cellsOut});
  }
  return {turns: turnsOut, placements, owners};
}

function serializeMinimumDefenses(api, outcome) {
  const covers = [];
  const values = outcome.covers;
  try {
    for (const cover of values) {
      const first = cover.first;
      const second = cover.second;
      try {
        covers.push([[first.q, first.r], [second.q, second.r]]);
      } finally {
        freeQuietly(first);
        freeQuietly(second);
        freeQuietly(cover);
      }
    }
  } finally {
    // wasm-bindgen returns a copied JS array; its elements are freed above.
  }
  const kind = outcome.kind === api.MinimumDefenseKind.Covers ? "covers"
    : outcome.kind === api.MinimumDefenseKind.AttackerWin ? "attacker_win"
      : "not_forcing";
  return {kind, covers};
}

function freeQuietly(value) {
  if (!value) return;
  try { value.free(); }
  catch (_error) {
    // A Rust panic/trap can leave wasm-bindgen's dynamic borrow guard raised.
    // Preserve the original solver error; the UI terminates this worker after
    // receiving it, which reclaims the whole instance anyway.
  }
}

function stagedShortestBranch(api, solver, position, settings) {
  let limits, optimizeLimits, proofOutcome, outcome;
  const started = performance.now();
  let initialNodes = 0n;
  let optimized = false;
  try {
    limits = new api.SolverLimits(
      settings.depthCap,
      BigInt(settings.nodeBudget),
      api.SolverEngineEnum.Pdspn,
    );
    limits.pn2_nodes = BigInt(settings.leafNodeBudget || "50000");
    proofOutcome = settings.width === "wide"
      ? solver.solve_wide(position, limits)
      : solver.solve(position, limits);
    const certificateJson = proofOutcome.certificate_json;
    if (certificateJson && settings.optimize !== false) {
      initialNodes = BigInt(proofOutcome.nodes);
      const totalBudget = BigInt(settings.nodeBudget);
      const remainingBudget = totalBudget > initialNodes ? totalBudget - initialNodes : 0n;
      if (remainingBudget > 0n) {
        optimizeLimits = new api.SolverLimits(
          settings.depthCap,
          remainingBudget,
          api.SolverEngineEnum.Pdspn,
        );
        optimizeLimits.pn2_nodes = BigInt(settings.leafNodeBudget || "50000");
        outcome = solver.optimize_certificate(
          position, optimizeLimits, certificateJson, settings.width === "wide",
        );
        optimized = true;
      }
    }
    if (!outcome) {
      outcome = proofOutcome;
      proofOutcome = null;
    }
    const pv = serializePv(api, outcome);
    const finalCertificateJson = outcome.certificate_json;
    const certificate = finalCertificateJson ? JSON.parse(finalCertificateJson) : null;
    return {
      kind: kindName(api, outcome.kind),
      depth: outcome.depth > 0 ? outcome.depth : null,
      nodes: (optimized ? initialNodes + BigInt(outcome.nodes) : BigInt(outcome.nodes)).toString(),
      elapsedMs: performance.now() - started,
      turns: pv.turns,
      pv: pv.placements,
      pvOwners: pv.owners,
      certificate,
      certificateSummary: certificate ? {
        dagNodes: Number(outcome.certificate_nodes),
        proofEdges: Number(outcome.certificate_edges),
        maxAttackerTurns: outcome.certificate_max_attacker_turns,
      } : null,
      bestUpperDepth: optimized && outcome.best_upper_depth > 0
        ? outcome.best_upper_depth : null,
      excludedThroughDepth: optimized && outcome.excluded_through_depth > 0
        ? outcome.excluded_through_depth : 0,
      thresholdProbes: optimized ? Number(outcome.threshold_probes) : 0,
      shortestCertified: optimized && Boolean(outcome.shortest_certified),
    };
  } finally {
    freeQuietly(proofOutcome);
    freeQuietly(outcome);
    freeQuietly(optimizeLimits);
    freeQuietly(limits);
  }
}

function canonicalCover(cover) {
  return cover.map(cell => [Number(cell[0]), Number(cell[1])])
    .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
}

function sameCover(left, right) {
  return JSON.stringify(canonicalCover(left)) === JSON.stringify(canonicalCover(right));
}

function takeDefensePairs(values) {
  return values.map(pair => {
    const first = pair.first;
    const second = pair.second;
    try { return [[first.q, first.r], [second.q, second.r]]; }
    finally {
      freeQuietly(first);
      freeQuietly(second);
      freeQuietly(pair);
    }
  });
}

self.onmessage = async (event) => {
  const message = event.data || {};
  if (!["solve", "verify", "minimum-defenses", "rank-defenses"].includes(message.type)) return;
  const requestId = message.requestId;
  let solver, position, limits, optimizeLimits, outcome, proofOutcome, verification;
  try {
    const api = await loadApi();
    const input = message.position;
    position = new api.Position(
      input.winLength,
      input.placementRadius,
      input.maxMoves,
      playerValue(api, input.toMove),
      input.movesRemaining,
      new Int32Array(input.stonesFlat),
    );
    solver = new api.StrixSolver();

    if (message.type === "minimum-defenses") {
      const defenses = solver.minimum_defenses_after_attack(
        position,
        message.width === "wide",
      );
      try {
        self.postMessage({
          type: "minimum-defenses",
          requestId,
          ...serializeMinimumDefenses(api, defenses),
        });
      } finally {
        freeQuietly(defenses);
      }
      return;
    }

    if (message.type === "rank-defenses") {
      const defenses = solver.minimum_defenses_after_attack(
        position,
        message.width === "wide",
      );
      let serialized;
      try {
        serialized = serializeMinimumDefenses(api, defenses);
      } finally {
        freeQuietly(defenses);
      }
      const alternatives = serialized.kind === "covers"
        ? serialized.covers.filter(
          cover => !sameCover(cover, message.playedCover || []),
        ) : [];
      const defenderName = input.toMove === "P1" ? "P2" : "P1";
      const defenderNumber = input.toMove === "P1" ? 2 : 1;
      let best = null;
      let unresolved = false;
      let evaluated = 0;
      let nodesUsed = 0n;
      const totalBudget = BigInt(message.nodeBudget);

      // The defender moves now. A separate swapped-root proof starts from
      // this real pre-defence state; no cover is pre-applied and nobody gets an
      // extra turn. Run it when exact-cover enumeration has no legal witness;
      // ordinary cover comparisons retain the full run allowance.
      // Certificate-backed counter-wins are expensive in browser WASM.
      // Standard effort keeps the established backward review responsive;
      // Thorough and above opt into the swapped-root PDS-PN proof.
      const counterBudget = serialized.kind !== "covers" && totalBudget >= 1_000_000n
        ? totalBudget : 0n;
      let counterCharge = 0n;
      const proveCounterWin = () => {
        let counterPosition, screenLimits, defenseOutcome;
        try {
          counterPosition = new api.Position(
            input.winLength, input.placementRadius, input.maxMoves,
            playerValue(api, defenderName), 2, new Int32Array(input.stonesFlat),
          );
          // Shared IDTT defense analysis is a cheap initiative witness. Do not
          // launch the much deeper swapped PDS-PN proof unless it found an
          // independently verified offensive pair.
          const screenBudget = counterBudget < 5_000n ? counterBudget : 5_000n;
          if (screenBudget === 0n) return null;
          counterCharge = screenBudget;
          screenLimits = new api.SolverLimits(
            message.depthCap, screenBudget, api.SolverEngineEnum.Idtt,
          );
          defenseOutcome = message.width === "wide"
            ? solver.solve_defense_wide(counterPosition, screenLimits)
            : solver.solve_defense(counterPosition, screenLimits);
          const counterHints = takeDefensePairs(defenseOutcome.counter_threats || []);
          if (!counterHints.length) return null;
          const proofBudget = counterBudget > screenBudget
            ? counterBudget - screenBudget : 0n;
          if (proofBudget === 0n) {
            unresolved = true;
            return null;
          }
          counterCharge = counterBudget;
          const counterResult = stagedShortestBranch(api, solver, counterPosition, {
            width: message.width,
            depthCap: message.depthCap,
            nodeBudget: proofBudget.toString(),
            leafNodeBudget: message.leafNodeBudget,
            optimize: false,
          });
          const rootCover = counterResult.turns?.[0]?.cells || [];
          const matchingCover = alternatives.find(cover => sameCover(cover, rootCover));
          const suggestedCover = matchingCover || canonicalCover(rootCover);
          const isInitiativeRoot = counterHints.some(pair => sameCover(pair, suggestedCover));
          if (counterResult.kind === "budget_exceeded") unresolved = true;
          if (counterResult.kind !== "win" || !counterResult.certificate
              || !isInitiativeRoot
              || suggestedCover.length < 1 || suggestedCover.length > 2
              || sameCover(suggestedCover, message.playedCover || [])) return null;
          const upper = Number(counterResult.bestUpperDepth
            || counterResult.certificateSummary?.maxAttackerTurns
            || counterResult.depth || 0);
          return {
            cover: suggestedCover, classification: "counter_win",
            upper, lower: Number(counterResult.excludedThroughDepth || 0),
            result: counterResult,
          };
        } finally {
          freeQuietly(defenseOutcome);
          freeQuietly(screenLimits);
          freeQuietly(counterPosition);
        }
      };
      const postCounterWin = candidate => {
        self.postMessage({
          type: "defense-progress", requestId, evaluated: evaluated + 1,
          total: alternatives.length, cover: candidate.cover,
          classification: "counter_win", upper: candidate.upper, lower: candidate.lower,
        });
        self.postMessage({
          type: "defense-result", requestId, status: "counter_win",
          evaluated: evaluated + 1, total: alternatives.length,
          nodes: nodesUsed.toString(), best: candidate,
        });
      };

      if (serialized.kind !== "covers") {
        const counter = proveCounterWin();
        // Charge the reserved slice, not the outer PDS counter: nested leaf
        // searches are bounded by this slice but not all appear in `nodes`.
        nodesUsed += counterCharge;
        if (counter) {
          postCounterWin(counter);
          return;
        }
        self.postMessage({
          type: "defense-result", requestId,
          status: serialized.kind, evaluated: 0, total: 0,
          nodes: nodesUsed.toString(), best: null,
        });
        return;
      }

      const coverBudget = totalBudget > counterBudget ? totalBudget - counterBudget : 0n;
      const perCoverBudget = alternatives.length
        ? (coverBudget / BigInt(alternatives.length)).toString()
        : coverBudget.toString();
      for (const cover of alternatives) {
        let branchPosition;
        try {
          const stonesFlat = [...input.stonesFlat];
          for (const [q, r] of cover) stonesFlat.push(q, r, defenderNumber);
          branchPosition = new api.Position(
            input.winLength, input.placementRadius, input.maxMoves,
            playerValue(api, input.toMove), 2, new Int32Array(stonesFlat),
          );
          const result = stagedShortestBranch(api, solver, branchPosition, {
            width: message.width,
            depthCap: message.depthCap,
            nodeBudget: perCoverBudget,
            leafNodeBudget: message.leafNodeBudget,
          });
          evaluated++;
          nodesUsed += BigInt(result.nodes || "0");
          const upper = Number(result.bestUpperDepth
            || (result.certificateSummary && result.certificateSummary.maxAttackerTurns)
            || result.depth || 0);
          const lower = Number(result.excludedThroughDepth || 0);
          let classification = "unresolved";
          if (result.kind === "no") classification = "refutes";
          else if (lower >= Number(message.baselineRemainingTurns)) classification = "extends";
          else if (result.kind === "win" && upper > 0
              && upper <= Number(message.baselineRemainingTurns)) classification = "holds";
          else unresolved = true;
          const candidate = {cover, classification, upper, lower, result};
          if (classification === "refutes"
              || (classification === "extends" && (!best
                || best.classification !== "refutes"
                && (lower > best.lower || (lower === best.lower && upper > best.upper))))) {
            best = candidate;
          }
          self.postMessage({
            type: "defense-progress", requestId, evaluated,
            total: alternatives.length, cover, classification, upper, lower,
          });
          if (classification === "refutes") break;
        } finally {
          freeQuietly(branchPosition);
        }
      }
      if (!best && counterBudget > 0n) {
        const counter = proveCounterWin();
        nodesUsed += counterCharge;
        if (counter) {
          postCounterWin(counter);
          return;
        }
      }
      self.postMessage({
        type: "defense-result", requestId,
        status: best ? best.classification : unresolved ? "unresolved" : "no_improvement",
        evaluated, total: alternatives.length, nodes: nodesUsed.toString(), best,
      });
      return;
    }

    if (message.type === "verify") {
      verification = solver.verify_certificate(
        position,
        JSON.stringify(message.certificate),
      );
      self.postMessage({
        type: "verified",
        requestId,
        summary: {
          dagNodes: Number(verification.dag_nodes),
          proofEdges: Number(verification.proof_edges),
          maxAttackerTurns: verification.max_attacker_turns,
        },
      });
      return;
    }

    limits = new api.SolverLimits(
      message.depthCap,
      BigInt(message.nodeBudget),
      engineValue(api, message.engine),
    );
    limits.pn2_nodes = BigInt(message.leafNodeBudget || "50000");
    const started = performance.now();
    let initialNodes = 0n;
    let optimized = false;
    if (message.engine === "pdspn-shortest") {
      let certificateJson = message.certificate
        ? JSON.stringify(message.certificate)
        : "";
      if (!certificateJson) {
        proofOutcome = message.width === "wide"
          ? solver.solve_wide(position, limits)
          : solver.solve(position, limits);
        certificateJson = proofOutcome.certificate_json;
        if (!certificateJson) {
          // The prerequisite proof did not resolve. Return that honest result;
          // no shortest optimization can begin without a verified upper bound.
          outcome = proofOutcome;
          proofOutcome = null;
        } else {
          initialNodes = BigInt(proofOutcome.nodes);
        }
      }
      if (certificateJson) {
        const totalBudget = BigInt(message.nodeBudget);
        const remainingBudget = totalBudget > initialNodes ? totalBudget - initialNodes : 0n;
        if (remainingBudget === 0n && proofOutcome) {
          // Stage one used the selected effort. Preserve its verified WIN rather
          // than silently spending the same budget again in the shortest pass.
          outcome = proofOutcome;
          proofOutcome = null;
        } else {
          optimizeLimits = new api.SolverLimits(
            message.depthCap,
            remainingBudget,
            engineValue(api, message.engine),
          );
          optimizeLimits.pn2_nodes = BigInt(message.leafNodeBudget || "50000");
          outcome = solver.optimize_certificate(
            position,
            optimizeLimits,
            certificateJson,
            message.width === "wide",
          );
          optimized = true;
        }
      }
    } else {
      outcome = message.width === "wide"
        ? solver.solve_wide(position, limits)
        : solver.solve(position, limits);
    }
    const elapsedMs = performance.now() - started;
    const pv = serializePv(api, outcome);
    const certificateJson = outcome.certificate_json;
    const certificate = certificateJson ? JSON.parse(certificateJson) : null;
    const certificateSummary = certificate ? {
      dagNodes: Number(outcome.certificate_nodes),
      proofEdges: Number(outcome.certificate_edges),
      maxAttackerTurns: outcome.certificate_max_attacker_turns,
    } : null;
    self.postMessage({
      type: "result",
      requestId,
      result: {
        kind: kindName(api, outcome.kind),
        depth: outcome.depth > 0 ? outcome.depth : null,
        nodes: (initialNodes + BigInt(outcome.nodes)).toString(),
        elapsedMs,
        turns: pv.turns,
        pv: pv.placements,
        pvOwners: pv.owners,
        certificate,
        certificateSummary,
        bestUpperDepth: optimized && outcome.best_upper_depth > 0
          ? outcome.best_upper_depth : null,
        excludedThroughDepth: optimized && outcome.excluded_through_depth > 0
          ? outcome.excluded_through_depth : 0,
        thresholdProbes: optimized ? Number(outcome.threshold_probes) : 0,
        shortestCertified: optimized && Boolean(outcome.shortest_certified),
      },
    });
  } catch (error) {
    self.postMessage({
      type: "error",
      requestId,
      error: error && error.message ? error.message : String(error),
    });
  } finally {
    freeQuietly(verification);
    freeQuietly(proofOutcome);
    freeQuietly(outcome);
    freeQuietly(solver);
    freeQuietly(optimizeLimits);
    freeQuietly(limits);
    freeQuietly(position);
  }
};
