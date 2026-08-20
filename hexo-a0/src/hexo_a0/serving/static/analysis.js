// Analysis screen: HDS import, SSE loader, move tree, verdicts, eval bar,
// analysis board, and the app entry point (DOMContentLoaded). Loaded last.

function openHdsImport() {
  const dialog = document.getElementById("hds-import-dialog");
  if (!dialog || dialog.open) return;
  document.getElementById("hds-status").textContent = "";
  dialog.showModal();
  document.getElementById("hds-input").focus();
}

function closeHdsImport() {
  const dialog = document.getElementById("hds-import-dialog");
  if (dialog?.open) dialog.close();
  document.getElementById("hds-import-trigger")?.focus();
}

async function convertHds() {
  const src = document.getElementById("hds-input").value.trim();
  const statusEl = document.getElementById("hds-status");
  if (!src) { statusEl.textContent = "Paste a sandbox link or code."; return; }
  const submit = document.querySelector("#hds-import-dialog button[type=submit]");
  statusEl.textContent = "Importing position…";
  if (submit) submit.disabled = true;
  try {
    const resp = await fetch(URL_PREFIX + "/convert_hds", {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({src}),
    });
    const body = await resp.json();
    if (!resp.ok) { statusEl.textContent = body.error || "We could not import that position. Check the link or code and try again."; return; }
    document.getElementById("analysis-htttx").value = body.htttx;
    closeHdsImport();
    loadGame();
  } catch (e) {
    statusEl.textContent = "We could not reach the server. Check your connection and try again.";
  } finally {
    if (submit) submit.disabled = false;
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  restoreAnalysisPreferences();
  updateForcingSolverUi();
  // The path selects the screen; the hash carries the line/game. Old share links
  // were "/#c=<moves>" (analysis hash on the play path) — upgrade them in place to
  // "/analysis#c=..." so path and content agree.
  const analysisHash = location.hash.startsWith("#c=") || location.hash.startsWith("#a=");
  let view = currentPathView();
  if (view === "play" && analysisHash) {
    view = "analysis";
    try { history.replaceState(null, "", viewPath("analysis") + location.hash); } catch (_e) {}
  }

  if (view === "analysis") {
    setView("analysis");
    const savedProofId = savedProofIdFromPath();
    if (savedProofId) {
      await loadSavedProof(savedProofId);
      return;
    }
    // Deep link into a specific line: #c=<compact> (or legacy #a=<decimal>).
    let deepMoves = null;
    if (location.hash.startsWith("#c=")) deepMoves = decodeMovesCompact(location.hash.slice(3));
    else if (location.hash.startsWith("#a=")) deepMoves = decodeMovesHash(location.hash.slice(3));
    if (deepMoves && deepMoves.length) {
      document.getElementById("analysis-htttx").value = serializeHtttx([[0, 0], ...deepMoves]);
      loadGame();
    }
    return;
  }

  // Play view.
  setView("play");
  let gid = null;
  if (location.hash.startsWith("#g=")) gid = location.hash.slice(3);
  else gid = localStorage.getItem("hexo_game_id");
  if (gid) {
    try {
      const resp = await fetch(URL_PREFIX + "/state", {
        method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({game_id: gid}),
      });
      if (resp.ok) {
        const body = await resp.json();
        applyState(body.state);
        return;
      }
    } catch (_e) { /* fall through to new-game modal */ }
    localStorage.removeItem("hexo_game_id");
    location.hash = "";
  }
  setStatus("Click 'New game' to begin.");
  openModal();
});

// --- Analysis view ---
// Trajectory values are native side-to-move perspective (the model's value
// head), so they flip sign every ply. We display everything from P1's fixed
// perspective so the eval bar / per-position value reads as a single
// consistent advantage line (positive = P1 winning), like policy_viewer.
let analysisMode = false;
let analysisMoves = [];          // loaded mainline move list incl. [0,0] seed
let analysisTrajectory = null;   // /analyze_game result (drives the eval bar)
const analysisCfg = window.__HEXO_CFG__ || {};
const DEFAULT_ANALYSIS_FORCING_DEPTH = 12;
const MAX_ANALYSIS_FORCING_DEPTH = 60;
const ANALYSIS_STRENGTHS = {
  network: {sims: 0, forcingDepth: 10, forcingBudget: "20000"},
  quick: {sims: 16, forcingDepth: 8, forcingBudget: "20000"},
  standard: {sims: 64, forcingDepth: 10, forcingBudget: "20000"},
  strong: {sims: 128, forcingDepth: 12, forcingBudget: "250000"},
  deep: {sims: 256, forcingDepth: 16, forcingBudget: "1000000"},
};
const FORCING_ENGINE_INFO = {
  idtt: {
    label: "IDTT",
    help: "Finds the shortest win, up to the maximum number of turns you set. It counts only turns taken by the side trying to win.",
  },
  dfpn: {
    label: "DFPN",
    help: "Looks for any forced win. Search effort controls how long it tries. The turn limit affects only the example line shown after a win is found.",
  },
  pdspn: {
    label: "PDS-PN",
    help: "Looks for a forced win and saves the opponent's replies so you can explore them. Search effort controls how long it tries. The turn limit affects only the example line.",
  },
  "pdspn-shortest": {
    label: "PDS-PN shortest",
    help: "First finds a forced win, then checks whether the same player can force it sooner. It calls the result shortest only after ruling out every shorter win.",
  },
  pns: {
    label: "PNS",
    help: "Provides a second yes-or-no check. It does not show a winning line or say how many turns the win takes.",
  },
};
let forcingUiEngine = "idtt";
let forcingIdttDepth = DEFAULT_ANALYSIS_FORCING_DEPTH;
let forcingDeepRecoveryDepth = MAX_ANALYSIS_FORCING_DEPTH;
let forcingShortestDepth = 25;
let forcingWorker = null;
let forcingRun = null;
let forcingRequestSerial = 0;
let analysisView = { x: 0, y: 0, scale: 1 };
let analysisPanning = false, analysisPanStart = { x: 0, y: 0, vx: 0, vy: 0 };
let analysisEvalHoverIdx = null;
let analysisEvalScrubbing = false;
function updateAnalysisHash() {
  // Mainline navigation is view state, not a new game: keep the full loaded
  // record in the URL while clicking or dragging through earlier positions.
  // A played side line remains shareable as its own branch.
  const onMainline = analysisCurrent && analysisMain[analysisCurrent.depth] === analysisCurrent;
  const line = analysisCurrent && !onMainline ? lineOf(analysisCurrent) : analysisMoves;
  const newHash = `#c=${encodeMovesCompact(line || [[0, 0]])}`;
  if (location.hash !== newHash) {
    try { history.replaceState(null, "", newHash); } catch (_e) { location.hash = newHash; }
  }
}

// HTTTX text uses explore.htttx.io's coordinate convention, a mirror of our
// internal axial frame: internal (q, r) <-> HTTTX (q + r, -r). Self-inverse,
// so parseHtttx and serializeHtttx apply the same map (mirrors serving/htttx.py).
function mirrorAxial(q, r) { return [q + r, -r]; }

function parseHtttx(text) {
  const re = /\[(-?\d+),(-?\d+)\]/g;
  const out = [];
  let m;
  while ((m = re.exec(text)) !== null) {
    out.push(mirrorAxial(parseInt(m[1]), parseInt(m[2])));
  }
  return out;
}

// --- Move tree (PGN-style with side-line variations) -----------------------
// analysisTree: root node {move:null,...}. analysisMain: the loaded mainline
// spine (array of nodes). analysisCurrent: the node the board currently shows.
// Each node: {move:[q,r]|null, player, parent, children:[], result, depth}.
// `result` is the per-position analysis: a mainline node points at its
// /analyze_game trajectory entry; a side-line node carries its own /analyze
// result (fetched on demand).
let analysisTree = null;
let analysisMain = [];
let analysisCurrent = null;
let analysisCancel = null;  // AbortController for an in-flight /analyze_game
let inferenceWorker = null;
let inferenceRequestSerial = 0;
const inferencePending = new Map();
let defenseHydrationTimer = null;
let defenseHydrationSerial = 0;
let qualityHydrationTimer = null;
let qualityHydrationSerial = 0;

function setProofLabOpen(open) {
  const drawer = document.getElementById("proof-lab-drawer");
  const proofTab = document.getElementById("proof-lab-launch");
  const analysisTab = document.getElementById("analysis-mode-analysis");
  const analysisBody = document.getElementById("analysis-controls-body");
  const analysisInfo = document.getElementById("analysis-info");
  if (!drawer || !proofTab || !analysisTab || !analysisBody) return;
  drawer.hidden = !open;
  analysisBody.hidden = open;
  if (analysisInfo) analysisInfo.hidden = open;
  proofTab.setAttribute("aria-selected", String(open));
  analysisTab.setAttribute("aria-selected", String(!open));
  document.getElementById("analysis-panel")?.classList.toggle("proof-lab-open", open);
  if (open) {
    updateProofLabPosition();
    drawer.querySelector("select, input, button")?.focus();
  }
}

function openProofLab() {
  if (!analysisCurrent) return;
  if (window.innerWidth <= 768) setAnalysisSheetOpen(true);
  setProofLabOpen(true);
}

function closeProofLab() {
  setProofLabOpen(false);
  document.getElementById("analysis-mode-analysis")?.focus();
}

function updateProofLabPosition() {
  const label = document.getElementById("proof-lab-position");
  if (!label || !analysisCurrent) return;
  const side = analysisCurrent.result?.current_player || "side to move";
  label.textContent = `Position ${analysisCurrent.depth + 1} · ${side} to move`;
}

document.addEventListener("keydown", event => {
  if (event.key === "Escape" && !document.getElementById("proof-lab-drawer")?.hidden)
    closeProofLab();
});

function analysisStrength() {
  const select = document.getElementById("analysis-strength");
  const key = select?.value || localStorage.getItem("hexo_analysis_strength") || "standard";
  const safe = ANALYSIS_STRENGTHS[key] ? key : "standard";
  if (select) select.value = safe;
  return {name: safe, ...ANALYSIS_STRENGTHS[safe]};
}

function saveAnalysisStrength() {
  const selected = analysisStrength();
  localStorage.setItem("hexo_analysis_strength", selected.name);
  updateAnalysisSettingsStatus();
}

function automaticAnalysisEnabled() {
  return Boolean(document.getElementById("analysis-auto-branch")?.checked);
}

function saveAutomaticAnalysis() {
  localStorage.setItem("hexo_analysis_auto_branch", String(automaticAnalysisEnabled()));
  updateAnalysisSettingsStatus();
}

function automaticForcingEnabled() {
  return Boolean(document.getElementById("analysis-auto-forcing")?.checked);
}

function saveAutomaticForcing() {
  localStorage.setItem("hexo_analysis_auto_forcing", String(automaticForcingEnabled()));
  if (automaticForcingEnabled()) scheduleThreatDefense(analysisCurrent);
}

function saveDisplayPreferences() {
  const preferences = [
    ["analysis-heatmap", "hexo_analysis_show_heatmap"],
    ["analysis-forcing", "hexo_analysis_show_forcing"],
    ["analysis-threats", "hexo_analysis_show_threats"],
  ];
  for (const [id, key] of preferences) {
    const input = document.getElementById(id);
    if (input) localStorage.setItem(key, String(input.checked));
  }
  rerenderCurrentAnalysis();
}

function updateAnalysisSettingsStatus() {
  const status = document.getElementById("analysis-settings-status");
  if (!status) return;
  const option = document.getElementById("analysis-strength")?.selectedOptions?.[0];
  const strength = option?.textContent?.split("·")[0]?.trim() || "Standard";
  status.textContent = `${strength} · auto ${automaticAnalysisEnabled() ? "on" : "off"}`;
}

function restoreAnalysisPreferences() {
  const savedStrength = localStorage.getItem("hexo_analysis_strength");
  const select = document.getElementById("analysis-strength");
  if (select && ANALYSIS_STRENGTHS[savedStrength]) select.value = savedStrength;
  const savedForcing = localStorage.getItem("hexo_analysis_auto_forcing");
  const toggle = document.getElementById("analysis-auto-forcing");
  if (toggle && savedForcing !== null) toggle.checked = savedForcing !== "false";
  const savedAutomatic = localStorage.getItem("hexo_analysis_auto_branch");
  const automatic = document.getElementById("analysis-auto-branch");
  if (automatic && savedAutomatic !== null) automatic.checked = savedAutomatic === "true";
  const displayPreferences = [
    ["analysis-heatmap", "hexo_analysis_show_heatmap"],
    ["analysis-forcing", "hexo_analysis_show_forcing"],
    ["analysis-threats", "hexo_analysis_show_threats"],
  ];
  for (const [id, key] of displayPreferences) {
    const saved = localStorage.getItem(key);
    const input = document.getElementById(id);
    if (input && saved !== null) input.checked = saved === "true";
  }
  updateAnalysisSettingsStatus();
}

function inferenceWorkerUrl() {
  const version = (window.__HEXO_CFG__ || {}).assetVersion || "";
  return `${URL_PREFIX}/static/inference-worker.js?v=${version}`;
}

function getInferenceWorker() {
  if (inferenceWorker) return inferenceWorker;
  inferenceWorker = new Worker(inferenceWorkerUrl(), {type: "module", name: "strix-inference"});
  inferenceWorker.onmessage = event => {
    const message = event.data || {};
    const pending = inferencePending.get(message.requestId);
    if (!pending) return;
    if (message.type === "progress") {
      pending.progress?.(message.done, message.total, message);
    } else if (message.type === "estimate") {
      pending.estimate?.(message.result);
    } else if (message.type === "error") {
      inferencePending.delete(message.requestId);
      pending.reject(new Error(message.error || "local inference failed"));
    } else if (message.type === "game" || message.type === "position" ||
               message.type === "bestMove" || message.type === "defense") {
      inferencePending.delete(message.requestId);
      pending.resolve(message.result);
    }
  };
  inferenceWorker.onerror = event => {
    const error = new Error(event.message || "local inference worker failed");
    for (const pending of inferencePending.values()) pending.reject(error);
    inferencePending.clear();
    inferenceWorker?.terminate();
    inferenceWorker = null;
  };
  return inferenceWorker;
}

function cancelLocalInference() {
  if (!inferenceWorker) return;
  inferenceWorker.terminate();
  inferenceWorker = null;
  for (const pending of inferencePending.values()) pending.reject(new DOMException("Cancelled", "AbortError"));
  inferencePending.clear();
}

function localInference(type, payload, progress, estimate) {
  const requestId = ++inferenceRequestSerial;
  return new Promise((resolve, reject) => {
    inferencePending.set(requestId, {resolve, reject, progress, estimate});
    getInferenceWorker().postMessage({
      type, requestId, modelUrl: (window.__HEXO_CFG__ || {}).modelUrl,
      ...payload,
    });
  });
}

function _newNode(move, player, parent, result) {
  return {move, player, parent, children: [], result,
          depth: parent ? parent.depth + 1 : 0};
}

function lineOf(node) {
  // Full move list (incl. the [0,0] seed at depth 0) from root to `node`.
  const out = [];
  let n = node;
  while (n) { if (n.move) out.unshift(n.move); n = n.parent; }
  return [[0, 0], ...out];
}

function playerAtDepth(depth) {
  // Seed (depth 0) is P1; then P2,P2,P1,P1,P2,P2,... (2 placements/turn).
  if (depth === 0) return "P1";
  return ((depth - 1) % 4) < 2 ? "P2" : "P1";
}

function forcingDepthFromUi() {
  const input = document.getElementById("analysis-forcing-depth");
  let depth = Number(input ? input.value : DEFAULT_ANALYSIS_FORCING_DEPTH);
  if (!Number.isInteger(depth)) depth = DEFAULT_ANALYSIS_FORCING_DEPTH;
  depth = Math.max(1, Math.min(MAX_ANALYSIS_FORCING_DEPTH, depth));
  if (input) input.value = String(depth);
  return depth;
}

function updateForcingSolverUi() {
  const engineSelect = document.getElementById("analysis-forcing-engine");
  const depthInput = document.getElementById("analysis-forcing-depth");
  if (!engineSelect || !depthInput) return;
  const next = engineSelect.value;
  if (next !== forcingUiEngine) {
    if (forcingUiEngine === "idtt") forcingIdttDepth = forcingDepthFromUi();
    else if (forcingUiEngine === "pdspn-shortest") forcingShortestDepth = forcingDepthFromUi();
    else if (forcingUiEngine !== "pns") forcingDeepRecoveryDepth = forcingDepthFromUi();
    depthInput.value = String(next === "idtt" ? forcingIdttDepth
      : next === "pdspn-shortest" ? forcingShortestDepth : forcingDeepRecoveryDepth);
    forcingUiEngine = next;
  }
  const isPns = next === "pns";
  const isIdtt = next === "idtt";
  const isShortest = next === "pdspn-shortest";
  const isPds = next === "pdspn" || isShortest;
  document.getElementById("analysis-forcing-depth-row").hidden = isPns;
  document.getElementById("analysis-forcing-leaf-row").hidden = !isPds;
  document.getElementById("analysis-forcing-depth-label").textContent = isShortest
    ? "Longest win to check"
    : isIdtt ? "Maximum turns by the winning side" : "Maximum turns in the example line";
  document.getElementById("analysis-forcing-budget-label").textContent = "Search effort";
  document.getElementById("analysis-solver-help").textContent = FORCING_ENGINE_INFO[next].help;

  // Best-first PNS retains its whole tree. Very large budgets can exhaust a
  // browser tab before the counter is reached; DFPN/PDS-PN are the long-run
  // engines and keep the larger steps available.
  const budget = document.getElementById("analysis-forcing-budget");
  for (const option of budget.options) {
    option.disabled = isPns && Number(option.value) > 1_000_000;
  }
  if (isPns && Number(budget.value) > 1_000_000) budget.value = "1000000";
}

function setForcingStatus(message, state = "") {
  const status = document.getElementById("analysis-forcing-status");
  if (!status) return;
  status.textContent = message;
  if (state) status.dataset.state = state;
  else delete status.dataset.state;
}

function restoreForcingStatusForNode(node) {
  const search = node && node.result && node.result.forcing_search;
  const hasCertificate = Boolean(node && node.result && node.result.forcing_certificate);
  for (const id of ["analysis-explore-certificate-btn", "analysis-share-certificate-btn",
                    "analysis-download-certificate-btn"]) {
    const button = document.getElementById(id);
    if (button) button.hidden = !hasCertificate;
  }
  if (!search) {
    setForcingStatus("Ready. This search runs on your device.");
    return;
  }
  const engine = FORCING_ENGINE_INFO[search.engine];
  const label = engine ? engine.label : search.engine;
  const width = search.width === "wide" ? "Broad search" : "Direct-only search";
  const elapsed = formatSolverElapsed(search.elapsed_ms);
  const work = formatSolverNodes(search.nodes, search.engine);
  if (search.kind === "win") {
    if (search.engine === "pdspn-shortest" && search.shortest_certified) {
      setForcingStatus(
        `Shortest forced win: ${search.best_upper_depth} turns by the winning side. The search ruled out every win in ${search.excluded_through_depth} turns or fewer. ${width} · ${work} · ${elapsed}.`,
        "win",
      );
      return;
    }
    const proof = search.proof_depth != null
      ? formatForcingLine(search.engine, search.proof_depth, search.line_placements)
      : "result only; no line shown";
    const certificate = search.certificate_summary;
    const certified = certificate
      ? ` · every saved reply checked; win within ${certificate.maxAttackerTurns} turns by the winning side`
      : "";
    setForcingStatus(`Forced win proved (${proof}). ${label} · ${width} · ${work}${certified} · ${elapsed}.`, "win");
  } else if (search.kind === "no") {
    setForcingStatus(`No forced win exists within the selected settings. ${label} · ${width} · ${work} · ${elapsed}.`, "no");
  } else {
    if (search.engine === "pdspn-shortest" && search.best_upper_depth) {
      const lower = Number(search.excluded_through_depth || 0);
      const range = lower > 0
        ? `The shortest win takes between ${lower + 1} and ${search.best_upper_depth} turns by the winning side`
        : `A win within ${search.best_upper_depth} turns is proved, but it may be possible sooner`;
      setForcingStatus(`The search stopped before it could prove the exact shortest win. ${range}. ${width} · ${work} · ${elapsed}.`, "budget");
    } else {
      setForcingStatus(`The search used all the selected effort without reaching a conclusion. Choose more effort or another method. ${label} · ${width} · ${work} · ${elapsed}.`, "budget");
    }
  }
}

function setForcingControlsRunning(running) {
  const button = document.getElementById("analysis-solve-forcing-btn");
  const cancel = document.getElementById("analysis-cancel-forcing-btn");
  button.disabled = running;
  cancel.hidden = !running;
  for (const id of ["analysis-explore-certificate-btn", "analysis-share-certificate-btn",
                    "analysis-download-certificate-btn"]) {
    const button = document.getElementById(id);
    if (button) button.disabled = running;
  }
  for (const id of ["analysis-forcing-engine", "analysis-forcing-width",
                    "analysis-forcing-depth", "analysis-forcing-budget",
                    "analysis-forcing-leaf-budget"]) {
    document.getElementById(id).disabled = running;
  }
}

function forcingPosition(node) {
  const result = node && node.result;
  if (!result) throw new Error("Load a position first.");
  if (result.terminal) throw new Error("The selected game is already over.");
  if (!result.current_player || !Array.isArray(result.stones)) {
    throw new Error("This position is still loading. Try again in a moment.");
  }
  const stonesFlat = [];
  for (const stone of result.stones) {
    if (!Array.isArray(stone) || !Array.isArray(stone[0])) continue;
    stonesFlat.push(Number(stone[0][0]), Number(stone[0][1]), stone[1] === "P1" ? 1 : 2);
  }
  const placementsAfterOrigin = Math.max(0, lineOf(node).length - 1);
  const rules = window.__HEXO_CFG__ || {};
  return {
    winLength: Number(rules.winLength || 6),
    placementRadius: Number(rules.placementRadius || 8),
    maxMoves: Number(rules.maxMoves || 400),
    toMove: result.current_player,
    movesRemaining: placementsAfterOrigin % 2 === 0 ? 2 : 1,
    stonesFlat,
  };
}

function portableProofPosition(position) {
  const stones = [];
  for (let i = 0; i < position.stonesFlat.length; i += 3) {
    stones.push([
      position.stonesFlat[i],
      position.stonesFlat[i + 1],
      position.stonesFlat[i + 2] === 1 ? "P1" : "P2",
    ]);
  }
  return {
    stones,
    attacker: position.toMove,
    placements_remaining: position.movesRemaining,
    config: {
      win_length: position.winLength,
      placement_radius: position.placementRadius,
      max_moves: position.maxMoves,
    },
  };
}

function forcingWorkerUrl() {
  const version = encodeURIComponent((window.__HEXO_CFG__ || {}).assetVersion || "");
  return `${URL_PREFIX}/static/solver-worker.js?v=${version}`;
}

function ensureForcingWorker() {
  if (forcingWorker) return forcingWorker;
  forcingWorker = new Worker(forcingWorkerUrl(), {type: "module", name: "strix-forced-win"});
  forcingWorker.onmessage = (event) => {
    const message = event.data || {};
    if (!forcingRun || message.requestId !== forcingRun.id) return;
    if (message.type === "result") finishForcingSolve(message.result);
    else if (message.type === "error") failForcingSolve(message.error);
  };
  forcingWorker.onerror = (event) => {
    if (forcingRun) failForcingSolve(event.message || "The solver worker crashed.");
  };
  return forcingWorker;
}

function formatSolverElapsed(ms) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(ms >= 10_000 ? 0 : 1)}s` : `${ms.toFixed(0)}ms`;
}

function isPdsEngine(engine) {
  return engine === "pdspn" || engine === "pdspn-shortest";
}

function formatSolverNodes(nodes, engine = "") {
  if (!nodes || nodes === "0") return "search-step count unavailable";
  try { return `${BigInt(nodes).toLocaleString()} search steps`; }
  catch (_error) { return `${nodes} search steps`; }
}

function formatSolverBudget(engine, budget) {
  return `${budget.toLocaleString()} search steps`;
}

function formatForcingLine(engine, attackerTurns, placements) {
  const turns = Number(attackerTurns);
  const placementText = `${placements} placement${placements === 1 ? "" : "s"}`;
  if (engine === "pdspn-shortest") {
    return `shortest win in ${turns} turn${turns === 1 ? "" : "s"} by the winning side; example uses ${placementText}`;
  }
  if (engine && engine !== "idtt") {
    return `example win in ${turns} turn${turns === 1 ? "" : "s"} by the winning side; ${placementText}`;
  }
  return `${turns} turn${turns === 1 ? "" : "s"} by the winning side; ${placementText}`;
}

function settleForcingUi() {
  if (forcingRun && forcingRun.ticker) clearInterval(forcingRun.ticker);
  setForcingControlsRunning(false);
}

function releaseForcingWorker() {
  if (forcingWorker) forcingWorker.terminate();
  forcingWorker = null;
}

function finishForcingSolve(result) {
  if (!forcingRun) return;
  const run = forcingRun;
  settleForcingUi();
  forcingRun = null;
  // Rust allocations are freed by the worker, but WebAssembly linear memory
  // cannot shrink. End the one-shot worker so a deep proof's arena/TT is
  // actually returned to the browser after its serialized result arrives.
  releaseForcingWorker();
  const engineInfo = FORCING_ENGINE_INFO[run.engine];
  const widthLabel = run.width === "wide" ? "Broad search" : "Direct-only search";
  const elapsed = formatSolverElapsed(result.elapsedMs);
  const work = formatSolverNodes(result.nodes, run.engine);
  run.node.result.forcing_search = {
    kind: result.kind,
    engine: run.engine,
    width: run.width,
    depth_cap: run.depth,
    node_budget: run.budget,
    leaf_node_budget: isPdsEngine(run.engine) ? run.leafBudget : null,
    nodes: result.nodes,
    elapsed_ms: result.elapsedMs,
    proof_depth: result.depth,
    line_placements: result.pv.length,
    certificate_summary: result.certificateSummary,
    best_upper_depth: result.bestUpperDepth,
    excluded_through_depth: result.excludedThroughDepth,
    shortest_certified: result.shortestCertified,
    threshold_probes: result.thresholdProbes,
  };

  if (result.certificate && result.certificateSummary) {
    const optimization = run.engine === "pdspn-shortest" ? {
      method: "pdspn-shortest-v1",
      shortestCertified: Boolean(result.shortestCertified),
      bestUpperDepth: result.bestUpperDepth,
      excludedThroughDepth: result.excludedThroughDepth,
      thresholdProbes: result.thresholdProbes,
      ...(result.turns && result.turns.length ? {sampleLine: result.turns} : {}),
    } : null;
    run.node.result.forcing_certificate = {
      format: "hexo-pdspn-proof-bundle-v1",
      position: portableProofPosition(run.position),
      // The retained DAG is always a PDS-PN proof, even when a bounded PDS-PN
      // optimizer subsequently tightened its depth.
      engine: "pdspn",
      width: run.width,
      verification: result.certificateSummary,
      certificate: result.certificate,
      ...(optimization ? {optimization} : {}),
    };
  } else {
    delete run.node.result.forcing_certificate;
  }
  for (const id of ["analysis-explore-certificate-btn", "analysis-share-certificate-btn",
                    "analysis-download-certificate-btn"]) {
    const button = document.getElementById(id);
    if (button) button.hidden = !run.node.result.forcing_certificate;
  }

  if (result.kind === "win") {
    const hasLine = result.pv.length > 0;
    run.node.result.forcing = {
      winner: run.position.toMove,
      attacker_is_mover: true,
      first_move: hasLine ? result.pv[0] : null,
      depth: result.depth,
      pv: result.pv,
      line_placements: result.pv.length,
      pv_len: result.pv.length,
      pv_owners: hasLine ? result.pvOwners : null,
      wide: run.width === "wide",
      engine: run.engine,
      certificate_summary: result.certificateSummary,
      verdict_only: !hasLine,
      defense: null,
    };
    const proof = hasLine
      ? formatForcingLine(run.engine, result.depth, result.pv.length)
      : "result only; no line shown";
    const certificate = result.certificateSummary;
    const certified = certificate
      ? ` · every saved reply checked; win within ${certificate.maxAttackerTurns} turns by the winning side`
      : "";
    if (run.engine === "pdspn-shortest" && result.shortestCertified) {
      const upper = result.bestUpperDepth;
      const lower = result.excludedThroughDepth;
      setForcingStatus(
        `Shortest forced win for ${run.position.toMove}: ${upper} turns by that player. The search ruled out every win in ${lower} turns or fewer. ${widthLabel} · ${work} · ${elapsed}.`,
        "win",
      );
    } else {
      setForcingStatus(
        `Forced win proved for ${run.position.toMove} (${proof}). ${engineInfo.label} · ${widthLabel} · ${work}${certified} · ${elapsed}.`,
        "win",
      );
    }
  } else if (result.kind === "no") {
    const scope = run.engine === "idtt"
      ? `within ${run.depth} turns by that player`
      : `within the selected settings`;
    setForcingStatus(
      `No forced win exists for ${run.position.toMove} ${scope}. ${engineInfo.label} · ${work} · ${elapsed}.`,
      "no",
    );
  } else {
    if (run.engine === "pdspn-shortest" && result.bestUpperDepth) {
      const lower = Number(result.excludedThroughDepth || 0);
      const range = lower > 0
        ? `The shortest win takes between ${lower + 1} and ${result.bestUpperDepth} turns by the winning side.`
        : `A win within ${result.bestUpperDepth} turns is proved, but it may be possible sooner.`;
      setForcingStatus(
        `The search stopped before it could prove the exact shortest win. ${range} ${widthLabel} · ${work} · ${elapsed}. Choose more search effort to continue.`,
        "budget",
      );
    } else {
      setForcingStatus(
        `The search used all ${formatSolverBudget(run.engine, run.budget)} without reaching a conclusion. Choose more effort or another method. ${engineInfo.label} · ${widthLabel} · ${elapsed}.`,
        "budget",
      );
    }
  }
  if (analysisCurrent === run.node) {
    renderNode(run.node);
    renderMoveTree();
  }
}

function downloadForcingCertificate() {
  const bundle = typeof currentProofBundle === "function" ? currentProofBundle()
    : (analysisCurrent && analysisCurrent.result
      ? analysisCurrent.result.forcing_certificate : null);
  if (!bundle) return;
  const blob = new Blob([JSON.stringify(bundle, null, 2) + "\n"], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `hexo-pdspn-proof-${Date.now()}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function failForcingSolve(error) {
  settleForcingUi();
  forcingRun = null;
  releaseForcingWorker();
  setForcingStatus(`The search stopped because of an error: ${error}`, "error");
}

function cancelForcingSolve(message = "Search cancelled. No result was recorded.") {
  if (!forcingRun) return;
  settleForcingUi();
  forcingRun = null;
  forcingRequestSerial += 1;
  releaseForcingWorker();
  setForcingStatus(message, "budget");
}

function solveCurrentForcing() {
  if (!analysisCurrent || !analysisCurrent.result) {
    setForcingStatus("Load a position before starting the search.", "error");
    return;
  }
  if (forcingRun) return;
  const engine = document.getElementById("analysis-forcing-engine").value;
  const width = document.getElementById("analysis-forcing-width").value;
  const depth = engine === "pns" ? MAX_ANALYSIS_FORCING_DEPTH : forcingDepthFromUi();
  const budget = Number(document.getElementById("analysis-forcing-budget").value);
  const leafBudget = Number(document.getElementById("analysis-forcing-leaf-budget").value);
  const node = analysisCurrent;
  let position;
  try {
    position = forcingPosition(node);
    const existingProof = node.result.forcing_certificate;
    const reusableCertificate = engine === "pdspn-shortest"
      && existingProof && existingProof.width === width
      ? existingProof.certificate : null;
    const worker = ensureForcingWorker();
    const id = ++forcingRequestSerial;
    forcingRun = {
      id, node, position, engine, width, depth, budget, leafBudget,
      reusedCertificate: Boolean(reusableCertificate), started: performance.now(),
    };
    forcingRun.ticker = setInterval(() => {
      if (!forcingRun || forcingRun.id !== id) return;
      const seconds = Math.floor((performance.now() - forcingRun.started) / 1000);
      const depthText = engine === "idtt" ? ` for wins within ${depth} turns`
        : engine === "pdspn-shortest" ? ` for wins within ${depth} turns` : "";
      setForcingStatus(
        `Searching${depthText} with ${FORCING_ENGINE_INFO[engine].label}… ${seconds}s elapsed. This uses one processor core on your device.`,
      );
    }, 1000);
    setForcingControlsRunning(true);
    const depthText = engine === "idtt" || engine === "pdspn-shortest"
      ? `, checking up to ${depth} turns by the winning side` : "";
    const proofText = reusableCertificate ? " The search will reuse the winning replies already saved for this position." : "";
    setForcingStatus(
      `Starting ${FORCING_ENGINE_INFO[engine].label}: ${width === "wide" ? "broad" : "direct-only"} search${depthText}, with ${formatSolverBudget(engine, budget)}.${proofText}`,
    );
    worker.postMessage({
      type: "solve", requestId: id, position, engine, width,
      depthCap: depth, nodeBudget: String(budget), leafNodeBudget: String(leafBudget),
      certificate: reusableCertificate,
    });
  } catch (error) {
    failForcingSolve(error.message || error);
  }
}

// Read an SSE stream, invoking onEvent(event, data) per frame. Resolves when
// the stream ends. Honors an AbortSignal (aborting cancels server-side).
async function streamSSE(url, payload, onEvent, signal) {
  const resp = await fetch(url, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload), signal,
  });
  if (!resp.ok) {
    let msg = resp.statusText;
    try { msg = (await resp.json()).error || msg; } catch (_e) {}
    throw new Error(msg);
  }
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const {done, value} = await reader.read();
    if (done) break;
    buf += dec.decode(value, {stream: true});
    let sep;
    while ((sep = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, sep);
      buf = buf.slice(sep + 2);
      let ev = "message", data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) ev = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) { try { onEvent(ev, JSON.parse(data)); } catch (_e) {} }
    }
  }
}

function replayPlayerAt(index) {
  return index === 0 ? 1 : (Math.floor((index - 1) / 2) % 2 === 0 ? 2 : 1);
}

function replayHasWin(stones, player, winLength) {
  const occupied = new Set(stones.filter(s => s.player === player).map(s => `${s.q},${s.r}`));
  for (const stone of stones) {
    if (stone.player !== player) continue;
    for (const [dq, dr] of [[1, 0], [0, 1], [1, -1]]) {
      if (occupied.has(`${stone.q - dq},${stone.r - dr}`)) continue;
      let count = 1;
      while (occupied.has(`${stone.q + dq * count},${stone.r + dr * count}`)) count++;
      if (count >= winLength) return true;
    }
  }
  return false;
}

function replayLegalMoves(stones, radius) {
  const occupied = new Set(stones.map(s => `${s.q},${s.r}`));
  const legal = new Set();
  for (const stone of stones) {
    for (let dq = -radius; dq <= radius; dq++) {
      for (let dr = -radius; dr <= radius; dr++) {
        if ((Math.abs(dq) + Math.abs(dr) + Math.abs(dq + dr)) / 2 > radius) continue;
        const key = `${stone.q + dq},${stone.r + dr}`;
        if (!occupied.has(key)) legal.add(key);
      }
    }
  }
  return [...legal].map(key => key.split(",").map(Number));
}

function replayEntryAt(moves, end, config) {
  const stones = moves.slice(0, end + 1).map((move, i) => ({
    q: move[0], r: move[1], player: replayPlayerAt(i),
  }));
  const winnerNum = replayHasWin(stones, 1, config.win_length) ? 1
    : replayHasWin(stones, 2, config.win_length) ? 2 : null;
  const draw = end >= config.max_moves;
  const terminal = Boolean(winnerNum || draw);
  const placements = Math.max(0, end);
  const toMove = Math.floor(placements / 2) % 2 === 0 ? 2 : 1;
  const lastMover = stones[stones.length - 1].player;
  const current = terminal ? (lastMover === 1 ? 2 : 1) : toMove;
  return {
    analyzed: false, value: null,
    current_player: current === 1 ? "P1" : "P2",
    terminal, winner: winnerNum ? `P${winnerNum}` : null,
    // Legal expansion is generated lazily when this prefix is viewed. Long
    // games can contain hundreds of prefixes; eagerly materialising every
    // radius-expanded move list would make a nominally instant load expensive.
    legal: terminal ? [] : null,
    stones: stones.map(s => [[s.q, s.r], s.player === 1 ? "P1" : "P2"]),
  };
}

function replayLoadedGame(moves) {
  const config = {
    win_length: analysisCfg.winLength,
    placement_radius: analysisCfg.placementRadius,
    max_moves: analysisCfg.maxMoves,
  };
  const seen = new Set();
  const trajectory = [];
  for (let i = 0; i < moves.length; i++) {
    const key = `${moves[i][0]},${moves[i][1]}`;
    if (seen.has(key)) throw new Error(`Move ${i + 1} repeats the occupied hex ${key}.`);
    seen.add(key);
    if (trajectory.length && trajectory[trajectory.length - 1].terminal)
      throw new Error(`The record contains moves after the game ended at move ${i}.`);
    trajectory.push(replayEntryAt(moves, i, config));
  }
  return trajectory;
}

let analysisRunActive = false;
let positionAnalysisSequence = 0;
function setAnalysisLoadedUi(loaded) {
  const navigation = document.getElementById("analysis-navigation");
  const emptyState = document.getElementById("analysis-empty-state");
  const proofLabLaunch = document.getElementById("proof-lab-launch");
  const setup = document.getElementById("analysis-setup");
  const sourceSummary = document.getElementById("analysis-source-summary");
  const sourceMeta = document.getElementById("analysis-source-meta");
  const sourceCancel = document.getElementById("analysis-source-cancel");
  if (navigation) navigation.hidden = !loaded;
  if (emptyState) emptyState.hidden = loaded;
  if (proofLabLaunch) proofLabLaunch.disabled = !loaded;
  if (setup) setup.hidden = loaded;
  if (sourceSummary) sourceSummary.hidden = !loaded;
  if (sourceMeta && loaded) sourceMeta.textContent = `${analysisMain.length} positions`;
  if (sourceCancel) sourceCancel.hidden = !loaded;
  if (!loaded) setProofLabOpen(false);
}

function editAnalysisSource() {
  document.getElementById("analysis-source-summary")?.setAttribute("hidden", "");
  document.getElementById("analysis-setup")?.removeAttribute("hidden");
  const cancel = document.getElementById("analysis-source-cancel");
  if (cancel) cancel.hidden = !analysisTree;
  document.getElementById("analysis-htttx")?.focus();
}

function cancelAnalysisSourceEdit() {
  if (!analysisTree) return;
  document.getElementById("analysis-setup")?.setAttribute("hidden", "");
  document.getElementById("analysis-source-summary")?.removeAttribute("hidden");
  setAnalysisActionButtons(true, analysisRunActive);
}

function setAnalysisActionButtons(enabled, running = false) {
  analysisRunActive = running;
  const position = document.getElementById("analysis-position-btn");
  const game = document.getElementById("analysis-game-btn");
  const load = document.getElementById("analysis-load-btn");
  if (position) position.disabled = !enabled || running || Boolean(analysisCurrent?.result?.terminal);
  if (game) game.disabled = !enabled || running;
  if (load) load.disabled = running;
}

function analysisInputChanged() {
  if (!analysisRunActive) setAnalysisActionButtons(false);
}

function loadGame() {
  const text = document.getElementById("analysis-htttx").value;
  const moves = parseHtttx(text);
  if (moves.length === 0) {
    document.getElementById("analysis-info").textContent = "No moves found. Check that the game record is in HTTTX format.";
    return;
  }
  analysisMoves = [[0, 0], ...moves];
  if (analysisCancel) analysisCancel.abort();
  cancelLocalInference();
  showProgress(false);
  try {
    const trajectory = replayLoadedGame(analysisMoves);
    buildTreeFromTrajectory({trajectory, boundary_indices: []}, false);
    setAnalysisActionButtons(true);
  } catch (error) {
    analysisTree = null;
    analysisMain = [];
    analysisCurrent = null;
    analysisTrajectory = null;
    setAnalysisLoadedUi(false);
    setAnalysisActionButtons(false);
    document.getElementById("analysis-info").textContent = `Could not load game: ${error.message || error}`;
  }
}

async function analyzeWholeGame() {
  const text = document.getElementById("analysis-htttx").value;
  const moves = parseHtttx(text);
  if (moves.length === 0) {
    document.getElementById("analysis-info").textContent = "No moves found. Check that the game record is in HTTTX format.";
    return;
  }
  analysisMoves = [[0, 0], ...moves];
  try {
    replayLoadedGame(analysisMoves);
  } catch (error) {
    document.getElementById("analysis-info").textContent = `Could not load game: ${error.message || error}`;
    return;
  }
  if (analysisCancel) analysisCancel.abort();
  // A full-game run owns the shared progress surface from this point onward.
  // Any automatic position request already in flight may still populate its
  // node, but must not hide or overwrite this run's progress UI.
  positionAnalysisSequence++;
  cancelLocalInference();
  const myCancel = new AbortController();
  analysisCancel = myCancel;
  setAnalysisActionButtons(true, true);
  let localProgress = false;
  showProgress(true, "Preparing analysis on your device…", 0, analysisMoves.length, null);
  try {
    const strength = analysisStrength();
    const result = await localInference("analyzeGame", {
      moves: analysisMoves,
      config: {win_length: analysisCfg.winLength, placement_radius: analysisCfg.placementRadius, max_moves: analysisCfg.maxMoves},
      sims: strength.sims, mActions: 16, strength, autoForcing: automaticForcingEnabled(),
    }, (done, total, progress) => {
      localProgress = true;
      const saved = Number(progress?.cacheHits || 0);
      const searched = Number(progress?.cacheMisses || 0);
      const message = searched === 0
        ? "Loading saved analysis…"
        : saved > 0
          ? `Analyzing and rating on your device (${strength.name}) · ${saved} saved…`
          : `Analyzing and rating on your device (${strength.name})…`;
      showProgress(true, message, done, total, null);
    });
    // A newer analysis may have started while we awaited; if so, this
    // result is stale — discard it so it can't overwrite the newer tree.
    if (analysisCancel !== myCancel) return;
    if (!result) {
      showProgress(false);
      setAnalysisActionButtons(true);
      document.getElementById("analysis-info").textContent = "Analysis finished without a result. Try again or choose a lower effort setting.";
      return;
    }
    buildTreeFromTrajectory(result);
    await rateMainlineTurns(myCancel);
    if (analysisCancel !== myCancel) return;
    showProgress(false);
    setAnalysisActionButtons(true);
  } catch (e) {
    if (e.name === "AbortError") return;
    // Compatibility path for browsers that cannot instantiate this WASM build.
    // It is intentionally reached only after local initialization/search fails.
    if (analysisCancel !== myCancel) return;
    if (localProgress) {
      showProgress(false);
      setAnalysisActionButtons(true);
      document.getElementById("analysis-info").textContent =
        `Analysis stopped on this device: ${e.message || e}. Try again or choose a lower effort setting.`;
      return;
    }
    showProgress(true, "This browser could not run the analysis. Trying the server…", 0, 0, null);
    try {
      let result = null;
      await streamSSE(URL_PREFIX + "/analyze_game", {moves: analysisMoves}, (ev, data) => {
        if (ev === "queued") showProgress(true, "Waiting for server analysis…", 0, 0, null);
        else if (ev === "progress") showProgress(true, "Analyzing on the server…", data.done, data.total, data.eta_seconds);
        else if (ev === "result") result = data;
      }, myCancel.signal);
      if (analysisCancel !== myCancel) return;
      if (result) {
        buildTreeFromTrajectory(result);
        await rateMainlineTurns(myCancel);
        if (analysisCancel !== myCancel) return;
        showProgress(false);
        setAnalysisActionButtons(true);
      }
      else throw e;
    } catch (fallbackError) {
      showProgress(false);
      setAnalysisActionButtons(true);
      if (fallbackError.name !== "AbortError")
        document.getElementById("analysis-info").textContent = `Analysis could not finish: ${fallbackError.message || fallbackError}`;
    }
  }
}

async function analyzeNode(node, automatic = false) {
  if (!node?.result || node.result.terminal) return;
  const sequence = ++positionAnalysisSequence;
  if (!automatic) setAnalysisActionButtons(true, true);
  const strength = analysisStrength();
  showPositionThinking(true, automatic ? "Thinking about your move…" : "Checking this position…");
  showProgress(true,
    automatic
      ? `Analyzing the new move on your device (${strength.name})…`
      : `Analyzing this position on your device (${strength.name})…`,
    0, 0, null);
  try {
    const result = await analyzePosition(lineOf(node), estimate => {
      if (sequence !== positionAnalysisSequence || !estimate) return;
      const hasValue = Number.isFinite(estimate.value);
      const hasForcing = Object.prototype.hasOwnProperty.call(estimate, "forcing");
      if (!hasValue && !hasForcing) return;
      const provisional = {
        ...node.result,
        ...(hasValue ? {value: estimate.value} : {}),
        ...(hasForcing ? {forcing: estimate.forcing} : {}),
        current_player: estimate.current_player || node.result.current_player,
        estimate_pending: true,
      };
      node.result = provisional;
      if (analysisTrajectory && analysisMain[node.depth] === node)
        analysisTrajectory.trajectory[node.depth] = provisional;
      if (analysisCurrent === node) renderNode(node);
      renderEvalBar();
      const nextStep = hasForcing && strength.sims > 0
        ? "Forced-win check ready · searching moves…"
        : strength.sims > 0
          ? "Estimate ready · checking forced wins…"
        : automaticForcingEnabled()
          ? "Estimate ready · checking forced wins…"
          : "Estimate ready";
      showPositionThinking(true, nextStep);
    });
    if (!result) throw new Error("analysis returned no result");
    result.analyzed = true;
    node.result = result;
    if (analysisTrajectory && analysisMain[node.depth] === node)
      analysisTrajectory.trajectory[node.depth] = result;
    // An explicit position check is one operation from the user's point of
    // view. Finish its turn verdict before dismissing the thinking state;
    // otherwise the lazy navigation pass can make the rating appear only
    // after another click. Automatic analysis follows the same rule whenever
    // the newly placed hex completes a turn.
    if (isTurnEnd(node) && !node.result.quality) {
      showPositionThinking(true, "Rating this turn…");
      await attachSideLineVerdict(node);
    }
    if (analysisCurrent === node) setCurrent(node);
    else renderMoveTree();
  } catch (error) {
    if (analysisCurrent === node && sequence === positionAnalysisSequence)
      document.getElementById("analysis-info").textContent = `${automatic ? "Automatic analysis" : "Analysis"} could not finish: ${error.message || error}`;
  } finally {
    if (sequence === positionAnalysisSequence) {
      showProgress(false);
      showPositionThinking(false);
    }
    if (!automatic) setAnalysisActionButtons(Boolean(analysisTree));
  }
}

async function analyzeCurrentPosition() {
  return analyzeNode(analysisCurrent);
}

function buildTreeFromTrajectory(result, analyzed = true) {
  // Every loaded game drives the position timeline. Entries without inference
  // stay on the neutral midline; analysis can fill them in later.
  analysisTrajectory = result;
  const tr = result.trajectory || [];
  // Build the mainline spine. trajectory[i] is the position AFTER applying
  // analysisMoves[0..i]; node i's move is analysisMoves[i] (null for the seed).
  analysisTree = _newNode(null, null, null, tr[0] || null);
  analysisMain = [analysisTree];
  let parent = analysisTree;
  for (let i = 1; i < tr.length; i++) {
    const mv = analysisMoves[i] || null;
    const node = _newNode(mv, playerAtDepth(i), parent, tr[i]);
    parent.children.push(node);
    analysisMain.push(node);
    parent = node;
  }
  setAnalysisLoadedUi(true);
  setCurrent(analysisMain[analysisMain.length - 1]);
  renderEvalBar();
  renderMoveTree();
}

function setCurrent(node) {
  if (forcingRun && node !== forcingRun.node) {
    cancelForcingSolve("Search cancelled because the selected position changed.");
  }
  analysisCurrent = node;
  const previousButton = document.getElementById("analysis-previous-position");
  const latestButton = document.getElementById("analysis-latest-mainline");
  if (previousButton) previousButton.disabled = !node?.parent;
  if (latestButton) latestButton.disabled = analysisMain[analysisMain.length - 1] === node;
  updateProofLabPosition();
  if (node?.result?.analyzed === false && node.result.legal === null) {
    const stones = node.result.stones.map(s => ({
      q: s[0][0], r: s[0][1], player: s[1] === "P1" ? 1 : 2,
    }));
    node.result.legal = replayLegalMoves(stones, analysisCfg.placementRadius);
  }
  setAnalysisActionButtons(Boolean(analysisTree), analysisRunActive);
  // Any ordinary navigation (timeline, undo, board click, a plain move link)
  // dismisses a selected missed-win callout; showMissedWin re-selects it
  // AFTER calling this, once its own navigation has landed.
  missedWinSelected = null;
  renderNode(node);
  renderEvalBar();
  renderMoveTree();
  renderMissedWinCallout();
  if (!forcingRun) restoreForcingStatusForNode(node);
  updateAnalysisHash();
  scheduleThreatDefense(node);
  scheduleTurnQuality(node);
}

function positionForMoves(moves) {
  return {
    config: {win_length: analysisCfg.winLength, placement_radius: analysisCfg.placementRadius, max_moves: analysisCfg.maxMoves},
    stones: moves.map((move, i) => ({
      q: move[0], r: move[1],
      player: i === 0 ? 1 : (Math.floor((i - 1) / 2) % 2 === 0 ? 2 : 1),
    })),
    to_move: Math.floor(Math.max(0, moves.length - 1) / 2) % 2 === 0 ? 2 : 1,
    moves_remaining: Math.max(0, moves.length - 1) % 2 === 0 ? 2 : 1,
  };
}

// Whole-game analysis deliberately uses a small, threat-only forcing pass so
// loading a long game stays fast. Complete the selected threat after scrubbing
// stops, then cache that richer result. This gives the board real defences (or
// the longest-delay move) without multiplying the cost across every prefix.
async function completeThreatDefense(node) {
  const forcing = node?.result?.forcing;
  if (!forcing || forcing.attacker_is_mover || forcing.defense || forcing.defense_status)
    return forcing;
  if (node._defensePromise) return node._defensePromise;
  node._defenseHydrating = true;
  node._defensePromise = (async () => {
    const strength = analysisStrength();
    const result = await localInference("analyzeDefense", {
      position: positionForMoves(lineOf(node)), strength,
      sims: strength.sims, mActions: 16, autoForcing: true, forcing,
    });
    if (result.status === "budget" && result.forcing)
      result.forcing.defense_status = "budget";
    node.result.forcing = result.forcing;
    if (analysisTrajectory && analysisMain[node.depth] === node)
      analysisTrajectory.trajectory[node.depth].forcing = result.forcing;
    return result.forcing;
  })();
  try {
    return await node._defensePromise;
  } finally {
    node._defenseHydrating = false;
    node._defensePromise = null;
  }
}

function scheduleThreatDefense(node) {
  clearTimeout(defenseHydrationTimer);
  defenseHydrationTimer = null;
  const forcing = node?.result?.forcing;
  if (!automaticForcingEnabled() || analysisRunActive || !forcing ||
      forcing.attacker_is_mover || forcing.defense || forcing.defense_status ||
      node._defenseHydrating) return;
  const serial = ++defenseHydrationSerial;
  defenseHydrationTimer = setTimeout(async () => {
    if (serial !== defenseHydrationSerial || analysisCurrent !== node) return;
    showPositionThinking(true, "Checking possible defences…");
    try {
      await completeThreatDefense(node);
      if (analysisCurrent === node) {
        renderNode(node);
        renderEvalBar();
      }
    } catch (_error) {
      // The already-proven threat remains useful if this optional pass fails.
    } finally {
      if (serial === defenseHydrationSerial) showPositionThinking(false);
    }
  }, 260);
}

// A loaded game or single-position analysis rates the selected turn lazily.
// Explicit whole-game analysis calls rateMainlineTurns instead, so every
// completed turn has a verdict before the operation reports that it is done.
function scheduleTurnQuality(node) {
  clearTimeout(qualityHydrationTimer);
  qualityHydrationTimer = null;
  if (analysisRunActive || !node?.result || node.result.quality || node.result.analyzed === false ||
      !isTurnEnd(node) || node._qualityHydrating) return;
  const serial = ++qualityHydrationSerial;
  qualityHydrationTimer = setTimeout(async () => {
    if (serial !== qualityHydrationSerial || analysisCurrent !== node) return;
    node._qualityHydrating = true;
    showPositionThinking(true, "Rating this turn…");
    try {
      await attachSideLineVerdict(node);
      if (analysisCurrent === node) {
        renderNode(node);
        renderMoveTree();
      }
    } finally {
      node._qualityHydrating = false;
      if (serial === qualityHydrationSerial && !node._defenseHydrating)
        showPositionThinking(false);
    }
  }, 180);
}

function showProgress(on, msg, done, total, eta) {
  const wrap = document.getElementById("analysis-progress");
  const bar = document.getElementById("analysis-progress-bar");
  const lbl = document.getElementById("analysis-progress-label");
  if (!wrap) return;
  // Use "block" (not "") for the on case: the element has a CSS rule
  // `display:none`, so clearing the inline style to "" would fall back to that
  // rule and keep the bar hidden during loading.
  wrap.style.display = on ? "block" : "none";
  if (!on) return;
  const pct = total > 0 ? Math.round(100 * done / total) : 0;
  if (bar) bar.style.transform = `scaleX(${pct / 100})`;
  let etaStr = "";
  if (eta !== null && eta !== undefined && total > 0) {
    etaStr = eta >= 60 ? ` · ETA ${Math.round(eta / 60)}m${String(Math.round(eta % 60)).padStart(2,"0")}s`
                       : ` · ETA ${Math.round(eta)}s`;
  }
  if (lbl) lbl.textContent = total > 0 ? `${msg} ${done}/${total}${etaStr}` : msg;
}

function showPositionThinking(on, message = "Checking position…") {
  const indicator = document.getElementById("analysis-thinking");
  const label = document.getElementById("analysis-thinking-label");
  if (!indicator) return;
  indicator.hidden = !on;
  if (on && label) label.textContent = message;
}

function returnToMainline() {
  if (analysisMain.length) setCurrent(analysisMain[analysisMain.length - 1]);
}

function analysisUndo() {
  // Step back to the parent of the current node.
  if (analysisCurrent && analysisCurrent.parent) setCurrent(analysisCurrent.parent);
}

function forcingIsUnstoppable(forcing) {
  const d = forcing?.defense;
  return Boolean(d && !(d.killers?.length || d.pair_anchors?.length) && d.best_delay);
}

function forcingIsCertain(forcing) {
  return Boolean(forcing?.attacker_is_mover || forcingIsUnstoppable(forcing));
}

// Effective P1-perspective eval for a trajectory entry's `result`: a PROVEN
// forced win (for the mover, or an opponent threat whose defense check found
// no refutation) or
// a terminal result pins the eval to ±1.0, overriding the value head, which
// routinely under-rates a forced win it hasn't been trained to recognise. A
// mere threat (attacker_is_mover false) is NOT certain — the mover may defend
// — so it falls through to the value head. Used by the eval bar, the gauge,
// and the "eval (P1)" readout so they reflect a proven win, not just q̂.
function effectiveP1Eval(entry) {
  if (!entry) return 0;
  if (entry.terminal && entry.winner) return entry.winner === "P1" ? 1.0 : -1.0;
  const f = entry.forcing;
  if (f && forcingIsCertain(f) && f.winner) return f.winner === "P1" ? 1.0 : -1.0;
  return Number.isFinite(entry.value) ? p1Perspective(entry.value, entry.current_player) : 0;
}

function entryHasEvaluation(entry) {
  return Boolean(entry && (Number.isFinite(entry.value) || entry.terminal ||
    (forcingIsCertain(entry.forcing) && entry.forcing?.winner)));
}

function renderEvalBar() {
  const wrap = document.getElementById("analysis-eval-wrap");
  const canvas = document.getElementById("analysis-eval-bar");
  if (!canvas || !wrap) return;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#0a0c0b";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const hasTrajectory = Boolean(analysisTrajectory?.trajectory?.length);
  wrap.hidden = !hasTrajectory;
  if (!hasTrajectory) return;
  const tr = analysisTrajectory.trajectory;
  const boundaries = new Set(analysisTrajectory.boundary_indices || []);
  const W = canvas.width, H = canvas.height;
  const pad = 4;
  const n = tr.length;
  const hasAnyEvaluation = tr.some(entryHasEvaluation);
  if (n < 1) return;
  const midY = H / 2;
  // Midline (P1/P2 even).
  ctx.strokeStyle = "#1b201f";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, midY); ctx.lineTo(W, midY); ctx.stroke();
  const xAt = (i) => n === 1 ? W / 2 : pad + (W - 2 * pad) * i / (n - 1);
  const yAt = (v) => midY - v * (H / 2 - pad);
  // Filled area: P1-orange above midline, P2-blue below — clipped per side so
  // the colours don't both paint the whole bar.
  ctx.save();
  ctx.beginPath(); ctx.rect(0, 0, W, midY); ctx.clip();
  ctx.fillStyle = "#f08a3c44";
  ctx.beginPath(); ctx.moveTo(xAt(0), midY);
  for (let i = 0; i < n; i++) { const v = effectiveP1Eval(tr[i]); ctx.lineTo(xAt(i), yAt(v)); }
  ctx.lineTo(xAt(n - 1), midY); ctx.closePath(); ctx.fill();
  ctx.restore();
  ctx.save();
  ctx.beginPath(); ctx.rect(0, midY, W, midY); ctx.clip();
  ctx.fillStyle = "#3fb6d944";
  ctx.beginPath(); ctx.moveTo(xAt(0), midY);
  for (let i = 0; i < n; i++) { const v = effectiveP1Eval(tr[i]); ctx.lineTo(xAt(i), yAt(v)); }
  ctx.lineTo(xAt(n - 1), midY); ctx.closePath(); ctx.fill();
  ctx.restore();
  // The eval line itself (P1 perspective — no per-ply oscillation).
  ctx.strokeStyle = hasAnyEvaluation ? "#ece6da" : "#646a64";
  ctx.lineWidth = 1.5;
  if (!hasAnyEvaluation) ctx.setLineDash([4, 4]);
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = xAt(i);
    const v = effectiveP1Eval(tr[i]);
    const y = yAt(v);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);
  // Turn-boundary dots, coloured by P1 advantage.
  for (let i = 0; i < n; i++) {
    if (!boundaries.has(i)) continue;
    const x = xAt(i);
    const v = effectiveP1Eval(tr[i]);
    const y = yAt(v);
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, 2 * Math.PI);
    ctx.fillStyle = !entryHasEvaluation(tr[i]) ? "#646a64"
      : v > 0.05 ? "#79cf9a" : v < -0.05 ? "#e25c5c" : "#f2b65a";
    ctx.fill();
  }
  const si = analysisCurrent && analysisMain[analysisCurrent.depth] === analysisCurrent
    ? analysisCurrent.depth : -1;
  if (si >= 0 && si < n) {
    const v = effectiveP1Eval(tr[si]);
    ctx.strokeStyle = "#c9a35e";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(xAt(si), pad); ctx.lineTo(xAt(si), H - pad); ctx.stroke();
  }
  if (analysisEvalHoverIdx !== null && analysisEvalHoverIdx >= 0 && analysisEvalHoverIdx < n) {
    const hi = analysisEvalHoverIdx;
    const v = effectiveP1Eval(tr[hi]);
    ctx.strokeStyle = "#ece6da99";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(xAt(hi), pad); ctx.lineTo(xAt(hi), H - pad); ctx.stroke();
    ctx.beginPath(); ctx.arc(xAt(hi), yAt(v), 3, 0, 2 * Math.PI);
    ctx.fillStyle = "#ece6da"; ctx.fill();
  }
  canvas.setAttribute("aria-valuemax", String(n));
  canvas.setAttribute("aria-valuenow", String(Math.max(0, si) + 1));
  if (si >= 0) {
    const entry = tr[si];
    const value = effectiveP1Eval(entry);
    canvas.setAttribute("aria-valuetext", entryHasEvaluation(entry)
      ? `Position ${si + 1}, P1 ${value >= 0 ? "+" : ""}${value.toFixed(2)}`
      : `Position ${si + 1}, not analyzed`);
  }
}

function analysisEvalIndexAt(clientX) {
  const canvas = document.getElementById("analysis-eval-bar");
  const count = analysisTrajectory?.trajectory?.length || 0;
  if (!canvas || !count) return -1;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width) return -1;
  const internalX = (clientX - rect.left) * canvas.width / rect.width;
  const fraction = Math.max(0, Math.min(1, (internalX - 4) / (canvas.width - 8)));
  return Math.round(fraction * (count - 1));
}

function updateAnalysisEvalPreview(index) {
  const preview = document.getElementById("analysis-eval-preview");
  const entry = analysisTrajectory?.trajectory?.[index];
  if (!preview || !entry) {
    if (preview) preview.hidden = true;
    return;
  }
  const value = effectiveP1Eval(entry);
  preview.textContent = entryHasEvaluation(entry)
    ? `Position ${index + 1} · P1 ${value >= 0 ? "+" : ""}${value.toFixed(2)}`
    : `Position ${index + 1} · not analyzed`;
  preview.hidden = false;
}

function onAnalysisEvalPointerMove(event) {
  analysisEvalHoverIdx = analysisEvalIndexAt(event.clientX);
  updateAnalysisEvalPreview(analysisEvalHoverIdx);
  if (analysisEvalScrubbing && analysisMain[analysisEvalHoverIdx])
    setCurrent(analysisMain[analysisEvalHoverIdx]);
  renderEvalBar();
}

function onAnalysisEvalPointerDown(event) {
  if (event.button !== 0) return;
  analysisEvalScrubbing = true;
  try { event.currentTarget?.setPointerCapture?.(event.pointerId); } catch (_error) {}
  const index = analysisEvalIndexAt(event.clientX);
  analysisEvalHoverIdx = index;
  updateAnalysisEvalPreview(index);
  if (analysisMain[index]) setCurrent(analysisMain[index]);
}

function onAnalysisEvalPointerUp(event) {
  analysisEvalScrubbing = false;
  try {
    if (event.currentTarget?.hasPointerCapture?.(event.pointerId))
      event.currentTarget.releasePointerCapture(event.pointerId);
  } catch (_error) {}
}

function onAnalysisEvalPointerLeave() {
  if (analysisEvalScrubbing) return;
  analysisEvalHoverIdx = null;
  updateAnalysisEvalPreview(-1);
  renderEvalBar();
}

function onAnalysisEvalClick(event) {
  const index = analysisEvalIndexAt(event.clientX);
  if (analysisMain[index]) setCurrent(analysisMain[index]);
}

function onAnalysisEvalKeydown(event) {
  if (!analysisMain.length) return;
  let index = analysisCurrent && analysisMain[analysisCurrent.depth] === analysisCurrent
    ? analysisCurrent.depth : analysisMain.length - 1;
  if (event.key === "ArrowLeft") index--;
  else if (event.key === "ArrowRight") index++;
  else if (event.key === "Home") index = 0;
  else if (event.key === "End") index = analysisMain.length - 1;
  else return;
  event.preventDefault();
  setCurrent(analysisMain[Math.max(0, Math.min(analysisMain.length - 1, index))]);
}

function currentAnalysisIdx() {
  // The index into the CURRENT line (lineOf(analysisCurrent)) — used for the
  // last-placed-hex / quality lookups.
  return analysisCurrent ? analysisCurrent.depth : 0;
}

// Re-draw the current node (used by the "Suggested moves" / "Forced wins" /
// "Threats" toggles so flipping them on/off repaints the board immediately).
function rerenderCurrentAnalysis() {
  renderNode(analysisCurrent);
}

// Move-quality is computed server-side (classify_turn_quality) and shipped on
// the trajectory entry at turn boundaries as {label, icon, color}. The frontend
// just reads node.result.quality.
function qualityOf(node) {
  return node && node.result ? (node.result.quality || null) : null;
}

// A turn normally = 2 placements, ending at even depth (seed=P1@d0; P2@d1,d2;
// P1@d3,d4; ...). But a WINNING/terminal placement ends the turn early (at odd
// depth), so a terminal node is also a turn end regardless of parity.
function isTurnEnd(node) {
  if (!node) return false;
  if (node.result && node.result.terminal) return true;
  return node.depth >= 2 && node.depth % 2 === 0;
}
// Depth where the current turn STARTED (the pre-turn position). For an even
// (normal) turn end that's depth-2; for a terminal turn that ended after a
// single placement, it's depth-1.
function turnStartDepth(node) {
  const single = node.result && node.result.terminal && node.depth % 2 === 1;
  return node.depth - (single ? 1 : 2);
}

const _QUAL_ICON = {best: "★", winning: "◆", good: "✓", mistake: "?", blunder: "✗", forced: "◇"};
// Match server QUALITY_COLORS: gold best, green good, amber mistake, crimson blunder.
const _QUAL_COLOR = {best: "#f2c14e", winning: "#79cf9a", good: "#79cf9a", mistake: "#e0a23a", blunder: "#e25c5c", forced: "#aeb8b1"};

// Client-side port of classify_turn_quality (analysis.py) so that turns played
// out in a SIDE LINE get the same best/good/mistake/blunder verdict the server
// computes for the mainline. preResult = the pre-turn position's /analyze
// (mover to move, has q_hat); postResult = the turn-end position's /analyze;
// turnMoves = the (up to 2) placements played this turn.
// The engine's CHOICE at a position is argmax improved_policy (the MCTS visit
// distribution = what it actually plays), NOT argmax q_hat: the value head can
// rate an under-visited move higher while the search correctly avoids it (e.g. a
// move that fails to block an open four). q_hat only SIZES the loss between two
// moves at the SAME position, where the estimates are directly comparable. This
// mirrors the server's classify_turn_quality so mainline and side lines agree.
function _bestMoveBy(res, name) {
  const arr = res && res[name];
  if (!arr || !res.legal) return null;
  let bi = -1, bp = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (res.candidate_set && !res.candidate_set[i]) continue;
    if (arr[i] > bp) { bp = arr[i]; bi = i; }
  }
  return bi >= 0 ? res.legal[bi] : null;
}
function _qOfMove(res, mv) {
  if (!res || !res.q_hat || !res.legal) return null;
  for (let i = 0; i < res.legal.length; i++)
    if (res.legal[i][0] === mv[0] && res.legal[i][1] === mv[1]) return res.q_hat[i];
  return null;
}
function labelFromLoss(matched, loss) {
  if (matched) return "best";
  if (loss >= 0.40) return "blunder";
  if (loss >= 0.15) return "mistake";
  return "good";
}

// POST a line to /analyze; returns the position result (JSON) or null.
async function analyzePosition(moves, onEstimate = null) {
  try {
    const strength = analysisStrength();
    const position = positionForMoves(moves);
    return await localInference("analyzePosition", {
      position, sims: strength.sims, mActions: 16, strength,
      autoForcing: automaticForcingEnabled(), previewValue: true,
    }, null, onEstimate);
  } catch (_e) {
    try {
      const resp = await fetch(URL_PREFIX + "/analyze", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({moves}),
      });
      return resp.ok ? await resp.json() : null;
    } catch (_fallbackError) { return null; }
  }
}

// Client mirror of classify_turn_quality (analysis.py) — ORDER-INDEPENDENT: a
// turn is judged by its placed-SET vs the engine's best placed-set, so [A,B] and
// [B,A] reach the same board and get the same verdict. `node` is the turn-end
// node. Async because the engine's 2nd pick can lie off the played line (when the
// engine's top pick wasn't played first) and then needs an on-demand /analyze.
// Returns the same quality shape the server ships, or null if unclassifiable.
async function computeTurnQuality(node) {
  const keyOf = (m) => `${m[0]},${m[1]}`;
  const a = turnStartDepth(node);
  let start = node; while (start && start.depth > a) start = start.parent;
  if (!start || !start.result) return null;
  const seq = []; { let n = node; while (n && n.depth > a) { seq.unshift(n); n = n.parent; } }
  if (!seq.length) return null;
  const played = seq.map(s => s.move);

  const e0 = _bestMoveBy(start.result, "improved_policy");
  if (!e0) return null;
  const engineLine = [e0];
  let engineEndQ = _qOfMove(start.result, e0);
  if (played.length >= 2) {
    let e1 = null;
    if (keyOf(e0) === keyOf(played[0])) {
      // Engine's first pick == player's first move → after-position is on-line.
      const after = seq[0].result;
      e1 = after ? _bestMoveBy(after, "improved_policy") : null;
      if (e1) engineEndQ = _qOfMove(after, e1);
    } else {
      // Off-line: evaluate the position after the engine's OWN first pick.
      const res = await analyzePosition([...lineOf(start), e0]);
      if (res) { e1 = _bestMoveBy(res, "improved_policy"); if (e1) engineEndQ = _qOfMove(res, e1); }
    }
    if (e1) engineLine.push(e1);
  }
  const engineSet = new Set(engineLine.map(keyOf));
  const playedSet = new Set(played.map(keyOf));
  const matched = engineLine.length === played.length &&
                  [...playedSet].every(k => engineSet.has(k));
  // Player's achieved end-of-turn value: q of the last placement at the node
  // just before it (the same board the engine's line is scored against).
  const beforeLast = seq.length >= 2 ? seq[seq.length - 2] : start;
  let playerEndQ = beforeLast && beforeLast.result
    ? _qOfMove(beforeLast.result, played[played.length - 1]) : null;
  let loss = 0;
  if (!matched && engineEndQ != null && playerEndQ != null) loss = Math.max(0, engineEndQ - playerEndQ);
  let label = labelFromLoss(matched, loss);

  // Forced-loss override (mirrors the server's _opponent_forced_loss): if the
  // end-of-turn position is a PROVEN loss for the mover — the opponent has a
  // forced win at this position (forcing.attacker_is_mover with winner == opp)
  // or it's terminal with the opponent winning — the turn is a blunder
  // regardless of the q-hat loss, and the effective eval is -1.0 so the swing
  // shows the full disaster. The value head often misses a forced loss the VCF
  // solver catches. `node` is the turn-end position (opponent to move).
  const mover = start.result && start.result.current_player;
  const opp = mover === "P1" ? "P2" : (mover === "P2" ? "P1" : null);
  let forcedBeforeTurn = false;
  if (opp) {
    let startForcing = start.result.forcing;
    if (automaticForcingEnabled() && startForcing && !startForcing.attacker_is_mover &&
        !startForcing.defense && !startForcing.defense_status) {
      try { startForcing = await completeThreatDefense(start); } catch (_error) { /* keep the known threat */ }
    }
    forcedBeforeTurn = Boolean(
      startForcing && forcingIsCertain(startForcing) && startForcing.winner === opp);
  }
  let forcedLoss = false;
  if (opp && node.result) {
    const r = node.result;
    if (r.terminal) {
      forcedLoss = r.winner === opp;
    } else if (r.forcing && forcingIsCertain(r.forcing) && r.forcing.winner === opp) {
      forcedLoss = true;
    }
  }
  if (forcedLoss) {
    if (forcedBeforeTurn) {
      label = "forced";
      loss = 0;
    } else {
      playerEndQ = -1.0;
      loss = engineEndQ != null ? Math.max(0, engineEndQ - playerEndQ) : 1.0;
      label = "blunder";
    }
  }
  // Proven results outrank the network's approximate score. If the mover ends
  // the turn with a verified forced win, a lower q estimate or a slower line
  // must never turn that winning move into a mistake/blunder.
  let endForcing = node.result && node.result.forcing;
  if (endForcing && !endForcing.attacker_is_mover &&
      !endForcing.defense && !endForcing.defense_status) {
    try { endForcing = await completeThreatDefense(node); } catch (_error) { /* an unverified threat is not proof */ }
  }
  const provenWin = Boolean(opp && node.result && (
    (node.result.terminal && node.result.winner === mover) ||
    (endForcing && forcingIsCertain(endForcing) && endForcing.winner === mover)));
  const startForcing = start.result && start.result.forcing;
  const winBeforeTurn = Boolean(startForcing && forcingIsCertain(startForcing) &&
    startForcing.winner === mover);
  if (provenWin && !matched) {
    label = "winning";
    loss = 0;
  }
  return {
    label, icon: _QUAL_ICON[label], color: _QUAL_COLOR[label],
    matched, engine_pair: engineLine, played_pair: played,
    loss, player_end_q: playerEndQ, engine_end_q: engineEndQ,
    forced_loss: forcedLoss, forced_before_turn: forcedBeforeTurn,
    proven_win: provenWin, win_before_turn: winBeforeTurn,
    turn_start_depth: a,
  };
}

// After a side-line node is analyzed, if it ends a turn, compute + attach its
// (order-independent) verdict — mirrors the server's classify_turn_quality.
async function attachSideLineVerdict(node) {
  if (!node || !node.result || node.result.quality) return;   // skip if server already set it
  if (!isTurnEnd(node)) return;
  const q = await computeTurnQuality(node);
  if (q) node.result.quality = q;
}

// Rate the mainline as a second phase of explicit whole-game analysis. A Hexo
// turn can contain two placements, so the verdict belongs to the completed
// turn rather than either placement in isolation. Existing server/cache
// verdicts are retained and only missing ones are computed.
async function rateMainlineTurns(owner = null) {
  const pending = analysisMain.filter(node =>
    node?.result && !node.result.quality && isTurnEnd(node));
  if (!pending.length) return;

  for (let i = 0; i < pending.length; i++) {
    if ((owner && analysisCancel !== owner) || owner?.signal.aborted)
      throw new DOMException("Cancelled", "AbortError");
    const node = pending[i];
    showProgress(true, "Rating completed turns on your device…", i, pending.length, null);
    node._qualityHydrating = true;
    try {
      await attachSideLineVerdict(node);
    } finally {
      node._qualityHydrating = false;
    }
    showProgress(true, "Rating completed turns on your device…", i + 1, pending.length, null);
  }

  if (analysisCurrent) renderNode(analysisCurrent);
  renderMoveTree();
}

// Render the position for a tree node: info line, board, heatmap, quality marks.
// The verdict card is ORDER-INDEPENDENT: a turn is judged by its resulting board
// (its placed-set) vs the engine's best placed-set, so a reversed in-turn order
// that reaches the same position still "matches". The whole turn's placements are
// marked, not just the last stone.
function renderNode(node) {
  if (!node || !node.result) { return; }
  const result = node.result;
  const info = document.getElementById("analysis-info");
  const cp = result.current_player || "?";
  const hasValue = Number.isFinite(result.value);
  const v1 = hasValue || (result.terminal && result.winner) ? effectiveP1Eval(result) : null;
  const evalStr = v1 == null ? null : (v1 >= 0 ? `+${v1.toFixed(2)}` : v1.toFixed(2));
  const q = qualityOf(node);

  updateGauge(v1, cp);
  const fmt = (arr) => arr.map(m => `[${m[0]},${m[1]}]`).join("");

  // Turn verdict: read the (order-free) fields the server/attachSideLineVerdict
  // shipped on quality — played-set vs engine's best set, sized by end-of-turn q.
  let playedMoves = null, turnPlayer = null, startNode = null;
  if (q && isTurnEnd(node)) {
    const a = (q.turn_start_depth != null) ? q.turn_start_depth : turnStartDepth(node);
    startNode = node; while (startNode && startNode.depth > a) startNode = startNode.parent;
    turnPlayer = (startNode && startNode.result && startNode.result.current_player)
               || playerAtDepth(node.depth);
    playedMoves = q.played_pair || null;
  }

  let html;
  if (q && playedMoves) {
    info.classList.add("vc");
    info.style.setProperty("--c", q.color);
    const lost = q.label === "mistake" || q.label === "blunder";
    const pcls = turnPlayer === "P1" ? "p1c" : "p2c";
    const who = `<span class="${pcls}">${turnPlayer}</span>`;
    const sgn = (x) => (x >= 0 ? "+" : "") + Number(x).toFixed(2);
    const enginePair = q.engine_pair || [];

    let board, pickTier = "", metric = "";
    const opp = turnPlayer === "P1" ? "P2" : "P1";
    const forcedLoss = q.forced_loss === true;
    const forcedBeforeTurn = q.forced_before_turn === true;
    const provenWin = q.proven_win === true;
    if (forcedBeforeTurn) {
      board = `${opp} had a <b>forced win</b> before this turn. No ${who} move could stop it. `;
      if (q.matched) {
        board += `${who} played the computer's preferred line <b class="mvc">${fmt(playedMoves)}</b>.`;
      } else {
        board += `${who} played <b class="mvc">${fmt(playedMoves)}</b>.`;
        if (enginePair.length)
          board += ` The computer preferred <b class="mvc">${fmt(enginePair)}</b>, but that line also loses.`;
      }
    } else if (forcedLoss) {
      // The turn handed the opponent a PROVEN forced win — that overrides the
      // q-hat loss (the value head often hasn't noticed a forced loss), so it's
      // always a blunder and the swing reflects the full -1.0 effective eval.
      if (q.matched) {
        board = `${who} played <b class="mvc">${fmt(playedMoves)}</b> and gave ${opp} a <b>forced win</b>. The computer's suggested line reaches the same lost position.`;
      } else if (enginePair.length) {
        board = `${who} played <b class="mvc">${fmt(playedMoves)}</b> and gave ${opp} a <b>forced win</b>. The computer preferred <b class="mvc">${fmt(enginePair)}</b>, which keeps the game alive.`;
        if (startNode && startNode.result) {
          pickTier = `<div class="vc-sec"><div class="h">Suggested line from here</div>`
                   + `${renderTopMovesHtml(startNode.result, startNode.depth)}</div>`;
        }
      } else {
        board = `${who} played <b class="mvc">${fmt(playedMoves)}</b> and handed ${opp} a <b>forced win</b>.`;
      }
    } else if (provenWin) {
      board = `${who} played <b class="mvc">${fmt(playedMoves)}</b> and `
            + `${q.win_before_turn ? "kept" : "created"} a <b>verified forced win</b>.`;
      if (enginePair.length && !q.matched)
        board += ` The computer preferred <b class="mvc">${fmt(enginePair)}</b>, but both outcomes force a win.`;
    } else if (q.matched) {
      // Same resulting position as the engine's line. Flag a reversed order so the
      // "why isn't this a mistake?" question answers itself.
      const reordered = enginePair.length === playedMoves.length && fmt(enginePair) !== fmt(playedMoves);
      board = `${who} matched the computer's suggested line <b class="mvc">${fmt(playedMoves)}</b>`
            + (reordered ? ` <span class="ro-note">(same position, reversed order)</span>` : "") + ".";
    } else if (enginePair.length) {
      board = `${who} played <b class="mvc">${fmt(playedMoves)}</b>. The computer `
            + `${lost ? "preferred" : "also liked"} <b class="mvc">${fmt(enginePair)}</b>`
            + (lost ? `, which reaches a stronger position` : "") + ".";
      // Explore the engine's line from the turn-start position (chips branch from
      // that node; ranked by policy — the engine's choice).
      if (startNode && startNode.result) {
        pickTier = `<div class="vc-sec"><div class="h">Suggested line from here</div>`
                 + `${renderTopMovesHtml(startNode.result, startNode.depth)}</div>`;
      }
    } else {
      board = `${who} played <b class="mvc">${fmt(playedMoves)}</b>.`;
    }
    // Metric compares the two full TURNS at their end positions (comparable q̂).
    // Shown whenever there's a real comparison — including a forced-loss turn
    // that "matched" the engine's own losing line (the -1.0 swing is the point).
    if (!forcedBeforeTurn && !provenWin && q.player_end_q != null && q.engine_end_q != null && (forcedLoss || !q.matched))
      metric = `<div class="vc-metric"><span>played line <b>${sgn(q.player_end_q)}</b></span><span class="sep">·</span>`
             + `<span>suggested line <b>${sgn(q.engine_end_q)}</b></span><span class="sep">·</span>`
             + `<span>difference <b>${sgn(q.player_end_q - q.engine_end_q)}</b></span></div>`;

    html =
      `<div class="vc-head"><div class="vc-glyph">${q.icon}</div>`
      + `<div class="vc-label">${q.label === "forced" ? "No saving move" : q.label === "winning" ? "Winning move" : q.label[0].toUpperCase() + q.label.slice(1)}<small>Turn ${Math.ceil(node.depth / 2)} · ${turnPlayer}</small></div></div>`
      + metric
      + `<div class="vc-sec"><div class="h">What happened</div><p>${board}</p></div>`
      + pickTier
      + (provenWin || forcedBeforeTurn ? "" : `<div class="vc-prov">The rating is based on the position left at the end of the turn. Playing the same two hexes in the opposite order receives the same rating.</div>`);
  } else {
    info.classList.remove("vc");
    info.style.removeProperty("--c");
    const positionTotal = analysisMain[node.depth] === node ? analysisMain.length : lineOf(node).length;
    html = `<div class="position-readout"><span class="position-turn"><b>${cp}</b> to move</span>`
         + (evalStr == null
           ? `<span class="ro-eval ro-unanalysed">Not analyzed</span>`
           : `<span class="ro-eval"><small>P1 ${result.estimate_pending ? "estimate" : "score"}</small><b>${evalStr}</b></span>`)
         + `<span class="ro-pos" title="Position ${node.depth + 1} of ${positionTotal}">${node.depth + 1}/${positionTotal}</span></div>`
         + renderTopMovesHtml(result);
  }
  info.innerHTML = html;
  // The BOARD heatmap always shows THIS position's suggested next moves.
  drawAnalysisBoard(result, result, q, playedMoves);
}

// Update the advantage gauge (needle + pole readouts) from a P1-perspective
// eval. Reveals the gauge + legend once analysis data exists.
function updateGauge(v1, cp) {
  const wrap = document.getElementById("gauge-wrap");
  const legend = document.getElementById("board-legend");
  if (!wrap) return;
  if (v1 === null || v1 === undefined || isNaN(v1)) { wrap.hidden = true; if (legend) legend.hidden = true; return; }
  wrap.hidden = false; if (legend) legend.hidden = false;
  const clamped = Math.max(-1, Math.min(1, v1));
  // P1 is the upper pole, so a P1 advantage pulls the marker upward.
  const needle = document.getElementById("gauge-needle");
  needle.style.left = "";
  needle.style.top = `${((1 - clamped) / 2 * 100).toFixed(1)}%`;
  const s = (x) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(2);
  const p1El = document.getElementById("gauge-v1"), p2El = document.getElementById("gauge-v2");
  p1El.textContent = s(v1);        // each pole reads positive when THAT player is ahead
  p2El.textContent = s(-v1);
  // Emphasise the leader (dim the trailing pole) so who's winning is obvious.
  p1El.style.opacity = clamped < -0.02 ? "0.4" : "1";
  p2El.style.opacity = clamped > 0.02 ? "0.4" : "1";
}

// A compact "engine likes:" row of the top candidate moves, ranked by the
// improved policy (MCTS visit-improved), with q_hat eval. When `branchDepth` is
// given, clicking a chip branches from the node at THAT depth on the current
// line (used by a verdict card to explore the engine's line from the exact
// decision point); otherwise it plays from the current node.
function renderTopMovesHtml(result, branchDepth) {
  if (!result || !result.improved_policy || !result.legal) return "";
  const ip = result.improved_policy, qh = result.q_hat, cs = result.candidate_set;
  const idx = ip.map((p, i) => i).filter(i => !cs || cs[i]);
  idx.sort((a, b) => ip[b] - ip[a]);   // rank by policy — the engine's choice
  const top = idx.slice(0, 5);
  if (!top.length) return "";
  const useDepth = branchDepth !== undefined && branchDepth !== null;
  let s = `<div class="analysis-suggestions"><span class="analysis-suggestions-label">Suggested next moves</span><div class="analysis-suggestion-list">`;
  for (const i of top) {
    const mv = result.legal[i];
    const qv = qh ? (qh[i] >= 0 ? "+" : "") + qh[i].toFixed(2) : "";
    const onclick = useDepth ? `analysisBranchAtDepth(${branchDepth},${mv[0]},${mv[1]})`
                             : `analysisCellClick(${mv[0]},${mv[1]})`;
    s += `<span class="topmove" onclick="${onclick}" `
       + `title="Try this move"><span>[${mv[0]},${mv[1]}]</span><small>${qv}</small></span>`;
  }
  return s + `</div></div>`;
}

// boardResult drives the stones + cell layout (the position you're looking at).
// heatResult drives the q_hat / probs heatmap + best-move markers (== boardResult
// for normal positions; the PRE-TURN position for a turn verdict). playedMoves,
// if given, are this turn's placements to mark with the quality icon.
function drawAnalysisBoard(boardResult, heatResult, quality, playedMoves) {
  const svg = document.getElementById("analysis-board");
  if (!boardResult || !boardResult.legal) { svg.innerHTML = ""; return; }
  heatResult = heatResult || boardResult;
  const stones = boardResult.stones || [];
  const stoneMap = new Map(stones.map(s => [`${s[0][0]},${s[0][1]}`, s[1]]));
  const margin = 8;
  const seeds = stones.length ? stones.map(s => ({q:s[0][0], r:s[0][1]})) : [{q:0,r:0}];
  const cellSet = new Set();
  for (const s of seeds) {
    for (let dq = -margin; dq <= margin; dq++) {
      for (let dr = -margin; dr <= margin; dr++) {
        if ((Math.abs(dq) + Math.abs(dr) + Math.abs(dq + dr)) / 2 <= margin) {
          cellSet.add(`${s.q+dq},${s.r+dr}`);
        }
      }
    }
  }
  for (const [q, r] of boardResult.legal) cellSet.add(`${q},${r}`);
  if (heatResult.legal) for (const [q, r] of heatResult.legal) cellSet.add(`${q},${r}`);

  const showHeat = document.getElementById("analysis-heatmap").checked;
  // Suggested-next-move heatmap: rank the side-to-move's MCTS-visited candidates
  // by the improved policy (how much the engine wants to play each) and shade
  // them in THAT player's colour, intensity ∝ strength; q_hat labels the top few.
  const mover = heatResult.current_player;
  const BASE = mover === "P2" ? [63, 182, 217] : [240, 138, 60];   // cyan / ember
  const RING = mover === "P2" ? "#9fe6f8" : "#ffcf9a";             // brighter tint
  const ip = heatResult.improved_policy, qh = heatResult.q_hat, cs = heatResult.candidate_set;
  const sugg = [];
  if (showHeat && ip) for (let i = 0; i < heatResult.legal.length; i++) {
    if (cs && !cs[i]) continue;                                    // MCTS-visited only
    sugg.push({ key: `${heatResult.legal[i][0]},${heatResult.legal[i][1]}`, ip: ip[i], q: qh ? qh[i] : null });
  }
  sugg.sort((a, b) => b.ip - a.ip);
  const maxIp = sugg.length ? sugg[0].ip : 0;
  const suggMap = new Map(sugg.map((s, rank) => [s.key, { ...s, rank }]));
  const topKey = sugg.length ? sugg[0].key : null;

  let body = `<g id="analysis-board-group">`;
  const labels = [];  // numeric q_hat labels for in-contention moves
  for (const key of cellSet) {
    const [q, r] = key.split(",").map(Number);
    const { x, y } = axialToPixel(q, r);
    const occ = stoneMap.get(key);
    let cls = "hex hex-empty";
    let fill = null, stroke = "";
    if (occ === "P1") { cls = "hex hex-p1"; fill = "#f08a3c"; }
    else if (occ === "P2") { cls = "hex hex-p2"; fill = "#3fb6d9"; }
    else if (showHeat && suggMap.has(key)) {
      // Suggested move for the side to move: player colour, opacity ∝ strength.
      const s = suggMap.get(key);
      const t = maxIp > 0 ? s.ip / maxIp : 0;
      fill = `rgba(${BASE[0]},${BASE[1]},${BASE[2]},${(0.14 + 0.60 * t).toFixed(3)})`;
      if (s.rank < 3 && s.q != null) labels.push({ x, y, t: (s.q >= 0 ? "+" : "") + s.q.toFixed(2) });
    }
    body += `<polygon class="${cls}" points="${hexCorners(x,y)}" data-q="${q}" data-r="${r}"`
           + (fill ? ` style="fill:${fill}${stroke}"` : "")
           + ` onclick="analysisCellClick(${q},${r})"/>`;
  }
  // Last-turn highlight (like policy_viewer): outline each side's most recent
  // turn — P1 light orange, P2 light blue. Walk the ordered move list back from
  // the current node, collecting the two latest contiguous same-player runs.
  if (analysisCurrent) {
    const line = lineOf(analysisCurrent);          // [[0,0](seed,P1), m1(P2), ...]
    const playerAt = (i) => i === 0 ? "P1" : playerAtDepth(i);
    const lastTurn = {P1: [], P2: []};
    let i = line.length - 1;
    let run = playerAt(i);
    while (i >= 0 && playerAt(i) === run) { lastTurn[run].push(line[i]); i--; }
    if (i >= 0) { run = playerAt(i); while (i >= 0 && playerAt(i) === run) { lastTurn[run].push(line[i]); i--; } }
    // Last-turn highlight: a LIGHT ring (lighter than the stone) drawn fully
    // INSIDE the cell (inset via the hexCorners scale) so it reads clearly.
    const TURN_RING = {P1: "#ffd7a8", P2: "#b3ecfb"};
    for (const player of ["P1", "P2"]) for (const mv of lastTurn[player]) {
      const p = axialToPixel(mv[0], mv[1]);
      body += `<polygon points="${hexCorners(p.x, p.y, 0.74)}" fill="none" stroke="${TURN_RING[player]}" stroke-width="2.5" stroke-opacity="0.95" pointer-events="none"/>`;
    }
  }
  // Terminal position: highlight the winning 6-in-a-row (parity with the play
  // board), so the final move shows the win rather than a bare position.
  if (boardResult.terminal && boardResult.winner) {
    body += winLineSvg(stones, boardResult.winner, (analysisTrajectory && analysisTrajectory.win_length) || 6);
  }
  // Ring the single strongest suggestion in the mover's brighter tint.
  if (showHeat && topKey) {
    const [tq, tr] = topKey.split(",").map(Number);
    const p = axialToPixel(tq, tr);
    body += `<circle cx="${p.x}" cy="${p.y}" r="${S * 0.6}" fill="none" stroke="${RING}" stroke-width="2.5" pointer-events="none"/>`;
  }
  // q_hat numeric labels on contention moves (drawn above fills).
  if (showHeat) for (const l of labels) {
    body += `<text x="${l.x}" y="${l.y + 1}" font-size="${Math.round(S * 0.34)}" fill="#f4efe5" text-anchor="middle" dominant-baseline="middle" pointer-events="none">${l.t}</text>`;
  }
  // Forcing-line (VCF) overlay: number every cell of the solved PV, attacker
  // placements filled in the winner's stone colour, defender replies muted/
  // outlined. Ownership comes from the server's per-cell replay
  // (forcing.pv_owners) — the PV's chunk lengths are NOT a fixed pairs-of-2
  // cadence (the first chunk is whatever moves_remaining_this_turn() was on
  // the solved side, and the final chunk can be a single cell that ends the
  // game), so it can't be inferred from position alone. If the server's
  // replay failed (pv_owners null), fall back to styling every cell as
  // attacker rather than guessing a cadence that's often wrong.
  // Two toggles gate the display: "Forced wins" (attacker_is_mover true — a
  // PROVEN win for the side to move, default on) and "Threats"
  // (attacker_is_mover false — a perspective-flipped solve that pretends the
  // mover skips their turn, so it's often defensible; default OFF because
  // routinely showing them muddies the analysis).
  const missedForcing = selectedMissedWinForcing();
  const boardForcing = missedForcing || boardResult.forcing;
  const isProven = forcingIsCertain(boardForcing);
  const showThis = boardForcing && (missedForcing || (isProven
    ? document.getElementById("analysis-forcing").checked
    : document.getElementById("analysis-threats").checked));
  const forcing = showThis ? boardForcing : null;
  if (forcing && forcing.pv && forcing.pv.length) {
    const f = forcing;
    const attackerColor = f.winner === "P2" ? "#3fb6d9" : "#f08a3c";
    // Defender's forced blocks render in the DEFENDER's own stone colour
    // (still dotted via .forcing-pv-defender), so the two sides of the line
    // are immediately distinguishable instead of colour-matching the winner.
    const defenderColor = f.winner === "P2" ? "#f08a3c" : "#3fb6d9";
    f.pv.forEach((mv, i) => {
      const p = axialToPixel(mv[0], mv[1]);
      const isAttacker = f.pv_owners ? f.pv_owners[i] === f.winner : true;
      const cls = `forcing-pv ${isAttacker ? "forcing-pv-attacker" : "forcing-pv-defender"}`;
      const labelCls = `forcing-pv-label ${isAttacker ? "forcing-pv-label-attacker" : "forcing-pv-label-defender"}`;
      const fontSize = Math.round(S * 0.42);
      if (isAttacker) {
        body += `<circle class="${cls}" cx="${p.x}" cy="${p.y}" r="${S * 0.42}" fill="${attackerColor}"/>`;
        body += `<text class="${labelCls}" x="${p.x}" y="${p.y + 1}" font-size="${fontSize}">${i + 1}</text>`;
      } else {
        body += `<circle class="${cls}" cx="${p.x}" cy="${p.y}" r="${S * 0.42}" stroke="${defenderColor}"/>`;
        body += `<text class="${labelCls}" x="${p.x}" y="${p.y + 1}" font-size="${fontSize}" fill="${defenderColor}">${i + 1}</text>`;
      }
    });
  }
  // Defense read-out overlay (threat banners from single-position analysis):
  // killers as check-marked rings in the DEFENDER's stone colour, pair
  // defenses as dashed anchor rings linked to a faded follow-up cell, the
  // max-delay fallback as a lone dotted ring. Board markers instead of raw
  // coordinates — the board has no coordinate labels.
  if (forcing && forcing.defense) {
    const d = forcing.defense;
    // The defender is the side to move (opponent of forcing.winner).
    const dColor = forcing.winner === "P2" ? "#f08a3c" : "#3fb6d9";
    // Corner BADGE, not a cell-sized ring: a small dark disc with a
    // defender-colour rim tucked toward the hex's top-right vertex (the -30°
    // corner of the pointy-top cell), so the cell's face stays readable and —
    // critically — clickable: hit-testing is confined to the badge itself.
    const bd = S * 0.66; // badge center: 2/3 of the way to the -30° vertex
    const badgeAt = (cell) => {
      const p = axialToPixel(cell[0], cell[1]);
      return { x: p.x + bd * Math.cos(-Math.PI / 6), y: p.y + bd * Math.sin(-Math.PI / 6) };
    };
    const badge = (cell, dash, glyph, opacity, tip) => {
      const b = badgeAt(cell);
      let g = `<g class="defense-badge" opacity="${opacity}" pointer-events="all">`;
      g += `<title>${tip}</title>`;
      g += `<circle cx="${b.x}" cy="${b.y}" r="${S * 0.26}" fill="#0d0f0e" stroke="${dColor}" stroke-width="${S * 0.055}"${dash ? ` stroke-dasharray="${S * 0.11} ${S * 0.08}"` : ""}/>`;
      g += `<text x="${b.x}" y="${b.y + 0.5}" font-size="${Math.round(S * 0.34)}" fill="${dColor}" text-anchor="middle" dominant-baseline="middle" font-weight="bold" pointer-events="none">${glyph}</text>`;
      return g + "</g>";
    };
    if (d.killers.length) {
      for (const c of d.killers) {
        body += badge(c, false, "✓", 0.95, "Defends: the threat is no longer provable after this placement");
      }
    } else if (d.pair_anchors.length) {
      // Multiple verified pairs would clutter the board; show only the best
      // one. The solver emits anchors in threat-PV order (most direct
      // refutation first), so the first pair is the natural pick.
      const [a, b] = d.pair_anchors[0];
      const ba = badgeAt(a), bb = badgeAt(b);
      body += `<line class="defense-badge" x1="${ba.x}" y1="${ba.y}" x2="${bb.x}" y2="${bb.y}" stroke="${dColor}" stroke-width="${S * 0.045}" stroke-dasharray="${S * 0.11} ${S * 0.11}" opacity="0.35" pointer-events="none"/>`;
      body += badge(a, true, "1", 0.95, "Pair defense: play this first — it only refutes together with the linked follow-up");
      body += badge(b, true, "2", 0.6, "Pair defense follow-up (re-checked after the first placement)");
    } else if (d.best_delay) {
      body += badge(d.best_delay, true, "…", 0.75, "The search did not prove a move that stops the threat. This move delays it longest.");
    }
  }
  // Move-quality icon: mark EVERY placement of the verdict's turn (not just the
  // last stone), so a 2-placement blunder shows the whole turn. Falls back to
  // the single current move for side-line nodes with no turn context.
  if (quality) {
    const marks = (playedMoves && playedMoves.length) ? playedMoves
                : (analysisCurrent && analysisCurrent.move ? [analysisCurrent.move] : []);
    for (const mv of marks) {
      const p = axialToPixel(mv[0], mv[1]);
      body += `<circle cx="${p.x}" cy="${p.y}" r="${S * 0.42}" fill="${quality.color}" opacity="0.9" pointer-events="none"/>`;
      body += `<text x="${p.x}" y="${p.y + 1}" font-size="${Math.round(S * 0.5)}" font-weight="bold" fill="#0d0f0e" text-anchor="middle" dominant-baseline="middle" pointer-events="none">${quality.icon}</text>`;
    }
  }
  body += "</g>";
  svg.innerHTML = body;
  updateAnalysisTransform();
  updateForcingBanner(forcing);
}

// One-line banner above the board reporting the both-side VCF solve, if any
// (result.forcing, pre-filtered by drawAnalysisBoard's per-kind toggles; see
// its PV overlay for the coordinate rendering). Created on first use so no
// static markup needs to change.
function updateForcingBanner(forcing) {
  const container = document.getElementById("analysis-board-container");
  if (!container) return;
  let banner = document.getElementById("forcing-banner");
  if (!banner) {
    banner = document.createElement("div");
    banner.id = "forcing-banner";
    banner.className = "forcing-pv-banner";
    container.insertBefore(banner, container.firstChild);
  }
  if (!forcing) { banner.hidden = true; banner.textContent = ""; return; }
  banner.hidden = false;
  // attacker_is_mover=true is a PROVEN win (the side to move executes it).
  // attacker_is_mover=false comes from a perspective-FLIP solve ("winner
  // would win if it were their turn") — the actual side to move places
  // first and may break it, so it is a threat to answer, NOT a lost
  // position. Don't overstate it.
  const defender = forcing.winner === "P1" ? "P2" : "P1";
  const placements = forcing.line_placements != null
    ? forcing.line_placements : forcing.pv_len;
  const hasLine = forcing.pv && forcing.pv.length > 0;
  const depthText = forcing.depth != null
    ? `example takes ${forcing.depth} turn${forcing.depth === 1 ? "" : "s"} by the winning side`
    : "number of turns unavailable";
  const proofText = forcing.verdict_only || !hasLine
    ? "win proved; moves not available"
    : `${depthText} · ${placements} placement${placements === 1 ? "" : "s"}`;
  const certified = forcing.certificate_summary
    ? ` · every saved reply checked; win within ${forcing.certificate_summary.maxAttackerTurns} turns by the winning side`
    : "";
  const d = forcing.defense;
  const hasCheckedDefense = Boolean(d && (d.killers.length || d.pair_anchors.length));
  const isUnstoppable = forcingIsUnstoppable(forcing);
  banner.textContent = forcing.attacker_is_mover
    ? `${forcing.winner} has a forced win (${proofText}${certified})`
    : isUnstoppable
      ? `${forcing.winner} has an unstoppable forced win (${proofText}) — ${defender} can only delay it`
      : `${forcing.winner} threatens a forced win (${proofText}) — ${defender} must answer`;
  if (forcing.engine) {
    const tag = document.createElement("span");
    tag.className = "forcing-engine-tag";
    tag.textContent = FORCING_ENGINE_INFO[forcing.engine]
      ? FORCING_ENGINE_INFO[forcing.engine].label : forcing.engine;
    banner.appendChild(tag);
  }
  if (forcing.wide) {
    // Verdict came from the wide generator (threat + quiet-builder turns) —
    // lines the tight live-play solver cannot see. Absent on results from an
    // extension predating the wide kwarg, or on old cached analyses.
    const tag = document.createElement("span");
    tag.className = "forcing-wide-tag";
    tag.textContent = "broad search";
    tag.title = "The search considered every legal move, including moves that build towards a later threat.";
    banner.appendChild(tag);
  }
  // Threat banners from single-position analysis may carry the defense
  // read-out: which placements refute the threat. Advisory (verified at
  // analysis budgets, time-boxed) — absent on wins, old extensions, or when
  // the defense sub-analysis had trouble.
  if (d && (d.killers.length || d.pair_anchors.length || d.best_delay)) {
    // The cells themselves are marked on the board (rings in the defender's
    // colour — see drawAnalysisBoard's defense overlay); the banner only
    // summarizes what kind of defense exists.
    let text;
    if (d.killers.length) {
      text = d.killers.length === 1
        ? "1 defending placement marked ✓ on the board"
        : `${d.killers.length} defending placements marked ✓ on the board`;
    } else if (d.pair_anchors.length) {
      text = "defense needs a pair — play 1, then 2 (best pair marked on the board)"
        + (d.pair_anchors.length > 1 ? ` — ${d.pair_anchors.length - 1} alternative${d.pair_anchors.length > 2 ? "s" : ""} exist` : "");
    } else {
      text = "No defense stops the win. The move that delays it longest is marked … on the board.";
    }
    const line = document.createElement("div");
    line.className = "forcing-defense-line";
    line.textContent = text;
    line.title = "These defensive moves were checked with the selected effort. A longer search may find more.";
    banner.appendChild(line);
  } else if (forcing.defense_status === "budget") {
    const line = document.createElement("div");
    line.className = "forcing-defense-line";
    line.textContent = "The threat is proved, but this effort level did not identify the best reply.";
    banner.appendChild(line);
  }
  // "win"/"threat" classes carry the colour (observatory.css).
  banner.classList.toggle("win", forcingIsCertain(forcing));
  banner.classList.toggle("threat", !forcingIsCertain(forcing));
}

function updateAnalysisTransform() {
  const g = document.getElementById("analysis-board-group");
  if (!g) return;
  const c = document.getElementById("analysis-board-container");
  const cx = c.clientWidth / 2, cy = c.clientHeight / 2;
  g.setAttribute("transform",
    `translate(${cx + analysisView.x},${cy + analysisView.y}) scale(${analysisView.scale})`);
}

// Click a board hex / "engine likes" chip -> branch from the CURRENT node.
function analysisCellClick(q, r) { return branchFrom(analysisCurrent, q, r); }

// Branch from the node at `depth` on the current line — used by verdict-card
// chips to explore the engine's line from the exact decision point (walks up
// from the current node, so it needs no node ids).
function analysisBranchAtDepth(depth, q, r) {
  let n = analysisCurrent;
  while (n && n.depth > depth) n = n.parent;
  return branchFrom(n || analysisCurrent, q, r);
}

function branchFrom(node, q, r) {
  // A touch drag ends with a browser-generated click on the last cell under
  // the finger. Do not turn that pan gesture into an accidental move.
  if (Date.now() < _analysisSuppressTapUntil) return;
  if (analysisPanning) return;
  if (!node || !node.result || !node.result.legal) return;
  if (!node.result.legal.some(c => c[0] === q && c[1] === r)) return;
  // If this child already exists (mainline or a prior side line), just go there.
  let child = node.children.find(c => c.move && c.move[0] === q && c.move[1] === r);
  if (child) { setCurrent(child); return; }
  // New side line: replay it immediately without inference. The user can then
  // request analysis for exactly this position with the explicit button.
  const moves = [...lineOf(node), [q, r]];
  const config = {
    win_length: analysisCfg.winLength,
    placement_radius: analysisCfg.placementRadius,
    max_moves: analysisCfg.maxMoves,
  };
  const result = replayEntryAt(moves, moves.length - 1, config);
  child = _newNode([q, r], playerAtDepth(node.depth + 1), node, result);
  node.children.push(child);
  setCurrent(child);
  if (automaticAnalysisEnabled() && !analysisRunActive && !result.terminal)
    void analyzeNode(child, true);
}

// A failed /analyze must not leave `child` in the tree: branchFrom's
// existing-child early-exit would then forever navigate to a node that renders
// nothing (result=null) and never re-fetch — the "stuck side line". Drop the
// placeholder and step back so re-clicking the cell retries from scratch.
function discardSideLine(parent, child, message) {
  const i = parent.children.indexOf(child);
  if (i >= 0) parent.children.splice(i, 1);
  if (analysisCurrent === child) setCurrent(parent);
  else renderMoveTree();
  // After setCurrent repaints the parent's info line, surface why the side
  // line vanished.
  const info = document.getElementById("analysis-info");
  if (info) info.textContent = message;
}

// --- PGN-style move tree rendering -----------------------------------------
function _moveLink(node) {
  const active = node === analysisCurrent ? " move-active" : "";
  // missed_win is per placement (computed after a whole-game analysis), so
  // side-line nodes from a single-position analysis do not carry it. Check it
  // here
  // rather than once per turn: whichever of the turn's (up to 2) move links
  // squandered the win gets the badge, not necessarily the turn-end one.
  const mw = node.result && node.result.missed_win;
  const badge = mw
    ? ` <span class="missed-win-badge" onclick="showMissedWin(${node._id})" `
    + `title="This move gave up a forced win. Show the winning line.">&#9889; Win missed</span>`
    : "";
  return `<span class="move-link${active}" onclick="scrubToNodeId(${node._id})">`
       + `[${node.move[0]},${node.move[1]}]</span>${badge}`;
}

// Selected missed-win callout: the {by, at_prefix, first_move, pv, pv_len,
// pv_owners} dict from the badge that's currently pinned open, or null.
let missedWinSelected = null;

function selectedMissedWinForcing() {
  const mw = missedWinSelected;
  if (!mw || analysisCurrent?.depth !== mw.at_prefix) return null;
  return {
    winner: mw.by, attacker_is_mover: true, first_move: mw.first_move,
    depth: mw.depth, pv: mw.pv || [], pv_owners: mw.pv_owners || null,
    line_placements: mw.line_placements, pv_len: mw.pv_len,
    wide: true, defense: null,
  };
}

// Click a "Missed win!" badge: navigate the board to `at_prefix` (the
// position where the win still existed — Task 2's forcing-pv overlay
// renders the line there automatically via that node's own `forcing` field,
// which is exactly what `missed_win` was copied from) and pin the callout
// panel open describing it.
function showMissedWin(nodeId) {
  const node = _nodeById[nodeId];
  const mw = node && node.result && node.result.missed_win;
  if (!mw) return;
  const target = analysisMain[mw.at_prefix];
  if (target) setCurrent(target);   // clears missedWinSelected + navigates
  missedWinSelected = mw;
  if (target) renderNode(target);
  renderMissedWinCallout();
}

// Compact explanation for a selected missed win. The detailed line belongs on
// the board; repeating every coordinate here makes the sidebar harder to scan.
function renderMissedWinCallout() {
  const movetree = document.getElementById("analysis-movetree");
  if (!movetree) return;
  let panel = document.getElementById("missed-win-callout");
  if (!panel) {
    panel = document.createElement("div");
    panel.id = "missed-win-callout";
    panel.className = "missed-win-callout";
    movetree.insertAdjacentElement("afterend", panel);
  }
  const mw = missedWinSelected;
  if (!mw) { panel.hidden = true; panel.innerHTML = ""; return; }
  panel.hidden = false;
  const placements = mw.line_placements != null ? mw.line_placements : mw.pv_len;
  const depth = mw.depth != null
    ? `${mw.depth} turn${mw.depth === 1 ? "" : "s"} by the winning side`
    : "number of turns unavailable";
  const first = mw.first_move ? `[${mw.first_move[0]},${mw.first_move[1]}]` : "shown on the board";
  panel.innerHTML = `<div class="mwc-head">Forced win missed</div>`
                   + `<p><strong>${mw.by}</strong> had a forced win here. The selected move gave it up; the winning line is now shown on the board.</p>`
                   + `<div class="mwc-meta">Start with <span>${first}</span> · ${depth} · ${placements} placement${placements === 1 ? "" : "s"}</div>`;
}

// Assign ids so onclick can find nodes. Re-runs on every render: clears the
// registry and re-numbers the whole tree (a node's id must always be present
// in _nodeById, or move-list clicks become no-ops).
let _nodeId = 0;
const _nodeById = {};
function _idify(node) {
  node._id = _nodeId++;
  _nodeById[node._id] = node;
  for (const c of node.children) _idify(c);
}

function scrubToNodeId(id) {
  const node = _nodeById[id];
  if (node) setCurrent(node);
}

// Split a line of nodes into turns: contiguous runs of the same player (up to 2
// placements each). nodes[] excludes the root/seed. Returns [{player, nodes,
// endNode}]. The seed (P1 @ depth 0) is its own degenerate turn.
function _turnsOf(nodes) {
  const turns = [];
  let cur = null;
  for (const n of nodes) {
    const p = playerAtDepth(n.depth);
    if (!cur || cur.player !== p) { cur = {player: p, nodes: [n], endNode: n}; turns.push(cur); }
    else { cur.nodes.push(n); cur.endNode = n; }
  }
  return turns;
}

// A turn cell: its (up to 2) move links + the end-of-turn verdict icon.
function _turnCellHtml(turn) {
  if (!turn) return "";
  const q = qualityOf(turn.endNode);
  const icon = q ? ` <span class="mv-q" style="color:${q.color}" title="${q.label}">${q.icon}</span>` : "";
  return turn.nodes.map(_moveLink).join(" ") + icon;
}

// Render the move list as a chess.com-style grid: one row per ROUND (a P1 turn
// + the following P2 turn), with the round number, then each side's turn cell.
// Side lines branch as indented sub-rows under the round they diverge from.
function renderMoveTree() {
  const el = document.getElementById("analysis-movetree");
  if (!el || !analysisTree) return;
  _nodeId = 0; for (const k in _nodeById) delete _nodeById[k];
  _idify(analysisTree);

  const mainNodes = analysisMain.slice(1);  // drop the seed (depth 0)
  const turns = _turnsOf(mainNodes);
  // HeXO opens with P1's seed then P2 moves first, so the first real turn is P2.
  // Pair turns into rounds [P1?, P2?]. We lead each round with P1's turn; the
  // very first round has no P1 turn (just the seed), so it shows P2 only.
  // Simplest robust pairing: walk turns, start a new round whenever we hit a P1
  // turn (or at the start), and slot P1/P2 by the turn's player.
  const rounds = [];
  let round = null;
  for (const t of turns) {
    if (t.player === "P1") { round = {P1: t, P2: null}; rounds.push(round); }
    else { // P2
      if (!round || round.P2) { round = {P1: null, P2: t}; rounds.push(round); }
      else round.P2 = t;
    }
  }

  // Variations: a side line starts at a node whose parent's mainline child is a
  // different node. Render each as an indented block of its own mini move list.
  function variationRows(startNode, nesting) {
    // Build the side line's own spine (following first-child).
    const nodes = [];
    let n = startNode;
    while (n && n.move) { nodes.push(n); n = n.children[0]; }
    const vturns = _turnsOf(nodes);
    const turnsHtml = vturns.map(t =>
      `<div class="variation-turn"><span class="var-side">${t.player}</span><span>${_turnCellHtml(t)}</span></div>`
    ).join("");
    const branchPosition = (startNode.parent?.depth || 0) + 1;
    let html = `<div class="variation" style="--variation-indent:${Math.min(nesting, 3) * 8}px">` +
      `<div class="variation-heading">Alternative from position ${branchPosition}</div>${turnsHtml}</div>`;
    // nested variations off any node in this side line
    for (const nd of nodes) for (const c of nd.children.slice(1)) html += variationRows(c, nesting + 1);
    return html;
  }
  // Collect variation HTML keyed by the mainline depth they branch from, so we
  // can drop them right after the relevant round.
  const varByDepth = {};
  const addVar = (parentNode) => {
    const mainChild = analysisMain[parentNode.depth + 1];
    for (const c of parentNode.children) {
      if (c !== mainChild && c.move) {
        (varByDepth[parentNode.depth] = varByDepth[parentNode.depth] || []).push(variationRows(c, 0));
      }
    }
  };
  addVar(analysisTree);
  for (const node of mainNodes) addVar(node);

  // Emit the grid.
  let html = `<div class="mv-grid">`;
  html += `<div class="mv-h">#</div><div class="mv-h">P1</div><div class="mv-h">P2</div>`;
  rounds.forEach((rd, i) => {
    const alt = i % 2 === 1 ? " mv-row-alt" : "";
    html += `<div class="mv-num${alt}">${i + 1}.</div>`;
    html += `<div class="mv-cell${alt}">${rd.P1 ? _turnCellHtml(rd.P1) : "<span class='mv-dim'>…</span>"}</div>`;
    html += `<div class="mv-cell${alt}">${rd.P2 ? _turnCellHtml(rd.P2) : "<span class='mv-dim'>…</span>"}</div>`;
    // Variations branching from any node within this round's turns.
    let vhtml = "";
    for (const t of [rd.P1, rd.P2]) if (t) for (const n of t.nodes)
      if (varByDepth[n.depth]) vhtml += varByDepth[n.depth].join("");
    if (vhtml) html += `<div class="mv-varspan">${vhtml}</div>`;
  });
  // Variations off the seed (root) appear before round 1.
  html += `</div>`;
  let seedVar = varByDepth[0] ? `<div class="mv-grid"><div class="mv-varspan">${varByDepth[0].join("")}</div></div>` : "";
  el.innerHTML = (seedVar + html) || "<span style='color:#5f635d'>No moves.</span>";
}

function serializeHtttx(moves) {
  let body = "version[1];\n";
  if (moves.length <= 1) return body;
  const rest = moves.slice(1).map(m => mirrorAxial(m[0], m[1]));
  let turn = 1, i = 0;
  while (i < rest.length) {
    const a = rest[i];
    const b = rest[i + 1];
    if (b === undefined) {
      body += `${turn}. [${a[0]},${a[1]}];\n`;
    } else {
      body += `${turn}. [${a[0]},${a[1]}][${b[0]},${b[1]}];\n`;
    }
    turn++; i += 2;
  }
  return body;
}

// Pan/zoom for analysis board
const analysisSvg = document.getElementById("analysis-board");
let _analysisTouchDragged = false;
let _analysisSuppressTapUntil = 0;
analysisSvg.addEventListener("mousedown", e => {
  analysisPanning = true;
  analysisPanStart = { x: e.clientX, y: e.clientY, vx: analysisView.x, vy: analysisView.y };
  analysisSvg.classList.add("panning");
});
window.addEventListener("mousemove", e => {
  if (!analysisPanning) return;
  analysisView.x = analysisPanStart.vx + (e.clientX - analysisPanStart.x);
  analysisView.y = analysisPanStart.vy + (e.clientY - analysisPanStart.y);
  updateAnalysisTransform();
});
window.addEventListener("mouseup", () => {
  analysisPanning = false;
  analysisSvg.classList.remove("panning");
});
// Single-finger pan for phones. The desktop mouse path above is not promoted
// consistently by mobile browsers, and without this explicit path a finger
// drag either scrolls the page or generates an accidental cell click.
analysisSvg.addEventListener("touchstart", e => {
  if (e.touches.length !== 1 || _analysisPinchStartDist) return;
  const t = e.touches[0];
  analysisPanning = true;
  _analysisTouchDragged = false;
  analysisPanStart = { x: t.clientX, y: t.clientY, vx: analysisView.x, vy: analysisView.y };
  analysisSvg.classList.add("panning");
}, {passive: true});
analysisSvg.addEventListener("touchmove", e => {
  if (e.touches.length !== 1 || _analysisPinchStartDist || !analysisPanning) return;
  const t = e.touches[0];
  const dx = t.clientX - analysisPanStart.x, dy = t.clientY - analysisPanStart.y;
  if (Math.abs(dx) + Math.abs(dy) > 4) _analysisTouchDragged = true;
  if (!_analysisTouchDragged) return;
  e.preventDefault();
  analysisView.x = analysisPanStart.vx + dx;
  analysisView.y = analysisPanStart.vy + dy;
  updateAnalysisTransform();
}, {passive: false});
const _endAnalysisTouch = () => {
  if (_analysisTouchDragged) _analysisSuppressTapUntil = Date.now() + 300;
  analysisPanning = false;
  analysisSvg.classList.remove("panning");
  _analysisTouchDragged = false;
};
analysisSvg.addEventListener("touchend", _endAnalysisTouch);
analysisSvg.addEventListener("touchcancel", _endAnalysisTouch);
analysisSvg.addEventListener("wheel", e => {
  e.preventDefault();
  const factor = Math.exp(-e.deltaY * 0.0004);
  const c = document.getElementById("analysis-board-container");
  const cx = c.clientWidth / 2, cy = c.clientHeight / 2;
  const rect = c.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  analysisView.x = mx - factor * (mx - analysisView.x - cx) - cx;
  analysisView.y = my - factor * (my - analysisView.y - cy) - cy;
  analysisView.scale = Math.max(0.3, Math.min(4, analysisView.scale * factor));
  updateAnalysisTransform();
}, {passive:false});
window.addEventListener("resize", updateAnalysisTransform);

// --- Responsive helpers (phone UI) ------------------------------------------
// On phones (≤768px) the analysis sidebar is a bottom sheet; clicking the
// handle or swiping the body toggles it. On desktop (>768px) the handle is
// display:none and these calls are harmless no-ops.
function analysisSheetIsOpen() {
  const c = document.getElementById("analysis-controls");
  return c && c.classList.contains("sheet-open");
}
function setAnalysisSheetOpen(open) {
  const c = document.getElementById("analysis-controls");
  if (!c) return;
  c.classList.toggle("sheet-open", open);
  const handle = document.getElementById("analysis-sheet-handle");
  if (handle) handle.setAttribute("data-label", open ? "Hide" : "Controls");
  // The board's available height changes when the sheet opens/closes on
  // phone layouts. The transform-origin / center recompute on the next frame.
  requestAnimationFrame(updateAnalysisTransform);
}
function toggleAnalysisSheet(event) {
  if (event) event.stopPropagation();
  setAnalysisSheetOpen(!analysisSheetIsOpen());
}
// Close the sheet with Escape — common mobile pattern when a keyboard's
// open or a deep link scrolls.
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && analysisSheetIsOpen()) setAnalysisSheetOpen(false);
});
// On resize across the 768px breakpoint, snap the sheet to its closed state
// (transform no longer applies on desktop, but the .sheet-open class would
// otherwise linger and look weird if the user resizes back down).
window.addEventListener("resize", () => {
  if (window.innerWidth > 768) setAnalysisSheetOpen(false);
});

// --- Auto-open the sheet on first visit (no position loaded) ---------------
// On phones, if the user opens the analysis screen with no position loaded,
// there's nothing on the board to look at — the controls ARE the interesting
// content. Auto-expand the bottom sheet so they're immediately useful, and
// collapse it once a position loads (the board becomes the focus).
window.addEventListener("hexo:view-changed", (e) => {
  if (e.detail && e.detail.view === "analysis" && window.innerWidth <= 768) {
    if (!analysisTree) setAnalysisSheetOpen(true);
  }
});
// When a position finishes loading, the board becomes the point — collapse
// the sheet so the user sees the board. Only on phones. We hook the
// `#analysis-info` element (populated by renderNode once data arrives) via
// MutationObserver rather than wrapping renderNode itself.
const _infoEl = document.getElementById("analysis-info");
let _lastAutoCollapsedAnalysisTree = null;
if (_infoEl) {
  new MutationObserver(() => {
    const info = document.getElementById("analysis-info");
    if (info && info.textContent.trim() && analysisTree &&
        analysisTree !== _lastAutoCollapsedAnalysisTree && window.innerWidth <= 768) {
      _lastAutoCollapsedAnalysisTree = analysisTree;
      setAnalysisSheetOpen(false);
    }
  }).observe(_infoEl, { childList: true, characterData: true, subtree: true });
}

// --- Responsive hex size ---------------------------------------------------
// shared.js dispatches "hexo:hex-size-changed" when the viewport crosses a
// breakpoint. Redraw the board with the new hex scale so stones sit at the
// right cell pitch.
window.addEventListener("hexo:hex-size-changed", () => {
  if (analysisCurrent && typeof renderNode === "function") {
    renderNode(analysisCurrent);
    // Re-fit: the previous pan/zoom was calibrated to the old hex size, so
    // recenter the view to the new container.
    analysisView = { x: 0, y: 0, scale: 1 };
    updateAnalysisTransform();
  }
});

// --- Pinch-to-zoom (touch) -------------------------------------------------
// Two-finger pinch on touch devices. Matches the wheel-zoom math: factor is
// the ratio of current-to-previous finger distance, and the zoom is anchored
// at the midpoint of the two fingers (so the gesture zooms toward where the
// user's fingers are, not the board center). The single-finger pan handler
// is gated on `e.touches.length === 1`, so adding the second finger is enough
// to "promote" the gesture from pan to pinch.
let _analysisPinchStartDist = 0;
let _analysisPinchStartScale = 1;
let _analysisPinchStartView = null;
let _analysisPinchMid = null;
analysisSvg.addEventListener("touchstart", e => {
  if (e.touches.length !== 2) return;
  e.preventDefault();
  const [t1, t2] = e.touches;
  const dx = t1.clientX - t2.clientX, dy = t1.clientY - t2.clientY;
  _analysisPinchStartDist = Math.hypot(dx, dy);
  _analysisPinchStartScale = analysisView.scale;
  _analysisPinchStartView = { x: analysisView.x, y: analysisView.y };
  const c = document.getElementById("analysis-board-container");
  const rect = c.getBoundingClientRect();
  _analysisPinchMid = { x: ((t1.clientX + t2.clientX) / 2) - rect.left,
                        y: ((t1.clientY + t2.clientY) / 2) - rect.top };
  analysisSvg.classList.add("panning");
}, {passive: false});
analysisSvg.addEventListener("touchmove", e => {
  if (e.touches.length !== 2 || !_analysisPinchStartDist) return;
  e.preventDefault();
  const [t1, t2] = e.touches;
  const dx = t1.clientX - t2.clientX, dy = t1.clientY - t2.clientY;
  const dist = Math.hypot(dx, dy);
  const factor = dist / _analysisPinchStartDist;
  const newScale = Math.max(0.3, Math.min(4, _analysisPinchStartScale * factor));
  // Anchor the zoom at the gesture midpoint: keep the midpoint's world coord
  // fixed under the fingers, recompute view.x/y accordingly. Mirrors the
  // wheel-zoom math in observatory.css's CSS file (see updateAnalysisTransform).
  const c = document.getElementById("analysis-board-container");
  const cx = c.clientWidth / 2, cy = c.clientHeight / 2;
  const mx = _analysisPinchMid.x, my = _analysisPinchMid.y;
  const scaleRatio = newScale / _analysisPinchStartScale;
  analysisView.scale = newScale;
  analysisView.x = mx - scaleRatio * (mx - _analysisPinchStartView.x - cx) - cx;
  analysisView.y = my - scaleRatio * (my - _analysisPinchStartView.y - cy) - cy;
  updateAnalysisTransform();
}, {passive: false});
const _endAnalysisPinch = () => {
  _analysisPinchStartDist = 0;
  _analysisPinchStartView = null;
  _analysisPinchMid = null;
  analysisSvg.classList.remove("panning");
};
analysisSvg.addEventListener("touchend", _endAnalysisPinch);
analysisSvg.addEventListener("touchcancel", _endAnalysisPinch);
