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
    pns: api.SolverEngineEnum.Pns,
    dfpn: api.SolverEngineEnum.Dfpn,
    pdspn: api.SolverEngineEnum.Pdspn,
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

function freeQuietly(value) {
  if (!value) return;
  try { value.free(); }
  catch (_error) {
    // A Rust panic/trap can leave wasm-bindgen's dynamic borrow guard raised.
    // Preserve the original solver error; the UI terminates this worker after
    // receiving it, which reclaims the whole instance anyway.
  }
}

self.onmessage = async (event) => {
  const message = event.data || {};
  if (message.type !== "solve" && message.type !== "verify") return;
  const requestId = message.requestId;
  let solver, position, limits, outcome, verification;
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
    outcome = message.width === "wide"
      ? solver.solve_wide(position, limits)
      : solver.solve(position, limits);
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
        nodes: outcome.nodes.toString(),
        elapsedMs,
        turns: pv.turns,
        pv: pv.placements,
        pvOwners: pv.owners,
        certificate,
        certificateSummary,
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
    freeQuietly(outcome);
    freeQuietly(solver);
    freeQuietly(limits);
    freeQuietly(position);
  }
};
