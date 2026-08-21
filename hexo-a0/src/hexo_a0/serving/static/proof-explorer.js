// Interactive, board-first reader for verified PDS-PN proof certificates.
// The solver/verifier owns correctness; this file presents that immutable DAG
// as ranked, already-certified attacker choices followed by every exact
// defender response.

const PROOF_CHOICE_COLORS = ["#3fb6d9", "#a58ae8", "#e0a23a", "#79cf9a"];
// Mirror of analysis.js positionNumber(); stand-alone so the proof explorer
// works even when analysis.js is not loaded (e.g. /proof/{id} shared link).
function proofPositionNumber(depth) {
  // Mirrors analysis.js positionNumber(); the proof explorer's "depth" is the
  // cell index within the SAMPLE LINE (a flat list of placements across the
  // captured PV turns), not an analysisMain placement index. Under standard
  // HeXO rules every turn places 2 stones, so round 1 = 4 placements and the
  // formula (depth+3)/4 round-numbers each placement correctly. The previous
  // (depth+1)/4 + 1 made round 1 only 3 placements wide, which contradicts the
  // standard 2-stone opener; the actual game starts with the side to move
  // having moves_left=2 unless `bundle.position.placements_remaining === 1`,
  // and the sample line cell counts already encode that for the sample-line
  // overlay (see proofSampleLineOverlaySvg).
  const select = document.getElementById("analysis-numbering");
  const mode = (select && select.value) || (typeof localStorage !== "undefined"
    && localStorage.getItem("hexo_analysis_numbering")) || "ply";
  if (mode !== "round") return depth + 1;
  if (depth <= 0) return 0;
  return Math.floor((depth + 3) / 4);
}
let proofExplorerState = null;
let proofPreviousFocus = null;
let proofBoardBounds = null;
let proofHighlightCenter = null;
let proofPreviewHighlight = null;
let proofView = {x: 0, y: 0, scale: 1};
let proofPan = null;
const proofSavedUrls = new WeakMap();
let proofVerificationSerial = 0;
// User toggle: draw the sample winning line on the proof board (mirrors the
// "Winning lines" overlay on the analysis board). Defaults to true when a
// sample line exists; the user can flip it off via the board-tools checkbox.
let proofShowLine = true;

function proofAttackerChoices(node) {
  if (node.kind !== "attacker_move") return [];
  const alternatives = node.alternatives == null ? [] : node.alternatives;
  if (!Array.isArray(alternatives)) {
    throw new Error("The saved result contains an invalid list of winning moves.");
  }
  return [{action: node.action, child: node.child}, ...alternatives];
}

function proofNodeEdges(node) {
  if (node.kind === "attacker_move") {
    return proofAttackerChoices(node).map((choice, index) => ({
      ...choice, role: "attacker", index,
    }));
  }
  if (node.kind === "defender_replies") {
    return node.responses.map((response, index) => ({
      ...response, role: "defender", index,
    }));
  }
  return [];
}

function proofNodeChildren(node) {
  return proofNodeEdges(node).map(edge => edge.child);
}

function buildProofModel(bundle) {
  const certificate = bundle && bundle.certificate;
  if (!certificate || !Array.isArray(certificate.nodes) || !certificate.nodes.length) {
    throw new Error("This result does not include the replies needed for exploration.");
  }
  const nodes = certificate.nodes;
  const root = Number(certificate.root);
  if (!Number.isInteger(root) || root < 0 || root >= nodes.length) {
    throw new Error("The saved result has an invalid starting position.");
  }
  const depths = new Array(nodes.length);
  const visiting = new Set();
  const parents = new Uint32Array(nodes.length);
  let edges = 0;

  for (let id = 0; id < nodes.length; id++) {
    for (const child of proofNodeChildren(nodes[id])) {
      if (!Number.isInteger(child) || child < 0 || child >= nodes.length) {
        throw new Error(`The saved result links position ${id} to a missing reply.`);
      }
      parents[child]++;
      edges++;
    }
  }

  function remaining(id) {
    if (depths[id] !== undefined) return depths[id];
    if (visiting.has(id)) throw new Error("The saved result contains a loop and cannot be shown.");
    visiting.add(id);
    const node = nodes[id];
    let depth;
    if (node.kind === "immediate_win" || node.kind === "unstoppable") {
      depth = 1;
    } else if (node.kind === "attacker_move") {
      const bounds = proofAttackerChoices(node).map(choice => 1 + remaining(choice.child));
      for (let index = 1; index < bounds.length; index++) {
        if (bounds[index] < bounds[index - 1]) {
          throw new Error(`The winning moves saved for position ${id} are not in the expected order.`);
        }
      }
      depth = bounds[0];
    } else if (node.kind === "defender_replies" && node.responses.length) {
      depth = Math.max(...node.responses.map(response => remaining(response.child)));
    } else {
      throw new Error(`Position ${id} in the saved result is incomplete.`);
    }
    visiting.delete(id);
    depths[id] = depth;
    return depth;
  }

  const maxAttackerTurns = remaining(root);
  if (bundle.verification && bundle.verification.maxAttackerTurns != null
      && Number(bundle.verification.maxAttackerTurns) !== maxAttackerTurns) {
    throw new Error("The downloaded summary does not match its saved replies.");
  }

  function worstResponseIndex(node) {
    let best = 0;
    for (let i = 1; i < node.responses.length; i++) {
      if (remaining(node.responses[i].child) > remaining(node.responses[best].child)) best = i;
    }
    return best;
  }

  return {
    certificate, nodes, root, parents, edges, remaining, worstResponseIndex,
    attackerChoices: proofAttackerChoices, maxAttackerTurns,
  };
}

function proofOptimizationDescription(bundle) {
  const optimization = bundle && bundle.optimization;
  if (!optimization || optimization.method !== "pdspn-shortest-v1"
      || !optimization.bestUpperDepth) return null;
  const upper = Number(optimization.bestUpperDepth);
  const lower = Number(optimization.excludedThroughDepth || 0);
  const dagDepth = Number(bundle.verification.maxAttackerTurns);
  const certMatches = dagDepth === upper;
  if (optimization.shortestCertified && lower + 1 === upper && certMatches) {
    // Tightest case: the saved DAG itself proves the exact shortest bound.
    return {
      short: `shortest win: ${upper} turns`,
      full: `The search proved that the winning side can force a win in ${upper} turns and cannot force one sooner. Every reply shown here was checked on this device.`,
    };
  }
  if (optimization.shortestCertified && lower + 1 === upper && !certMatches) {
    // Defensive branch: the server only flips shortestCertified when the
    // saved cert re-verifies at the new bound, but an older bundle saved
    // before that fix could still ship the inconsistent state. Don't claim
    // a proven shortest win in that case.
    return {
      short: lower > 0 ? `shortest win: ${lower + 1}–${upper} turns` : `win within ${upper} turns`,
      full: `The search narrowed the shortest win to between ${lower + 1} and ${upper} turns by the winning side, but the saved replies still prove a win within ${dagDepth} turns. Re-run with more effort to save a tighter certificate.`,
    };
  }
  // General case: the optimizer proved only a SEARCH upper, not a PROVEN
  // shortest bound. Be honest about that — the saved cert proves dagDepth
  // turns, and there may be a shorter win within `upper` turns.
  return {
    short: lower > 0 ? `shortest win: ${lower + 1}–${upper} turns` : `win within ${upper} turns`,
    full: lower > 0
      ? `The search proved that the shortest win takes between ${lower + 1} and ${upper} turns by the winning side. The saved replies shown here prove a win within ${dagDepth} turns.`
      : `The search proved a win within ${upper} turns by the winning side, but did not prove whether a faster win exists. The saved replies prove a win within ${dagDepth} turns.`,
  };
}

function buildProofSampleLine(bundle, rootStones, attacker) {
  const turns = bundle && bundle.optimization && bundle.optimization.sampleLine;
  if (!Array.isArray(turns) || !turns.length) return null;
  const entries = [{
    nodeId: Number(bundle.certificate.root), stones: new Map(rootStones),
    attackerTurnsPlayed: 0, label: "Start", lastAction: null, shownWin: false,
  }];
  let stones = new Map(rootStones);
  let attackerTurnsPlayed = 0;
  for (let index = 0; index < turns.length; index++) {
    const turn = turns[index];
    const expected = index % 2 === 0 ? attacker : (attacker === "P1" ? "P2" : "P1");
    if (!turn || turn.player !== expected || !Array.isArray(turn.cells)) {
      throw new Error(`Shortest sample turn ${index + 1} is malformed.`);
    }
    stones = applyProofAction(stones, turn.cells, turn.player);
    if (turn.player === attacker) attackerTurnsPlayed++;
    entries.push({
      nodeId: Number(bundle.certificate.root), stones,
      attackerTurnsPlayed,
      label: turn.player === attacker ? `Winning turn ${attackerTurnsPlayed}` : `Reply ${Math.ceil(index / 2)}`,
      lastAction: {action: turn.cells, player: turn.player},
      shownWin: index === turns.length - 1,
    });
  }
  return {turns, entries};
}

function applyProofAction(stones, action, player) {
  if (!Array.isArray(action) || action.length < 1 || action.length > 2) {
    throw new Error("Each saved move must place one or two hexes.");
  }
  const next = new Map(stones);
  for (const cell of action) {
    if (!Array.isArray(cell) || cell.length !== 2
        || !Number.isInteger(cell[0]) || !Number.isInteger(cell[1])) {
      throw new Error("A saved move has invalid coordinates.");
    }
    const key = `${cell[0]},${cell[1]}`;
    if (next.has(key)) throw new Error(`A saved move tries to use the occupied hex ${key}.`);
    next.set(key, player);
  }
  return next;
}

function normalizeProofPosition(position) {
  if (position && Array.isArray(position.stones)) return position;
  // Compatibility with proof bundles produced briefly during development.
  if (position && Array.isArray(position.stonesFlat)) {
    const stones = [];
    for (let i = 0; i < position.stonesFlat.length; i += 3) {
      stones.push([
        position.stonesFlat[i], position.stonesFlat[i + 1],
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
  throw new Error("The saved result does not include its starting position.");
}

function currentProofBundle() {
  if (proofExplorerState && proofExplorerState.bundle) return proofExplorerState.bundle;
  return analysisCurrent && analysisCurrent.result
    ? analysisCurrent.result.forcing_certificate : null;
}

function openProofExplorer() {
  const bundle = currentProofBundle();
  if (!bundle) return;
  openProofExplorerBundle(bundle);
}

function openProofExplorerBundle(bundle, {savedUrl = null} = {}) {
  try {
    const model = buildProofModel(bundle);
    const position = normalizeProofPosition(bundle.position);
    const stones = new Map();
    for (const stone of position.stones) {
      stones.set(`${stone[0]},${stone[1]}`, stone[2]);
    }
    const attacker = position.attacker;
    const rootEntry = {
      nodeId: model.root,
      stones,
      attackerTurnsPlayed: 0,
      label: "Start",
      lastAction: null,
      shownWin: false,
    };
    const sampleLine = buildProofSampleLine(bundle, stones, attacker);
    proofExplorerState = {
      bundle, model, position, attacker,
      defender: attacker === "P1" ? "P2" : "P1",
      history: [rootEntry], sampleLine, lineIndex: 0, mode: "proof",
    };
    if (savedUrl) proofSavedUrls.set(bundle, savedUrl);
    proofPreviousFocus = document.activeElement;
    document.getElementById("proof-explorer").hidden = false;
    document.body.classList.add("proof-explorer-open");
    // The "Show winning line" toggle only makes sense when the cert was saved
    // with a sample line (plain PDS-PN runs produce a cert but no sample line,
    // so the toggle is disabled rather than silently showing nothing).
    const toggle = document.getElementById("proof-show-line");
    const hasSampleLine = Boolean(sampleLine && sampleLine.turns.length);
    proofShowLine = hasSampleLine;
    if (toggle) {
      toggle.checked = hasSampleLine;
      toggle.disabled = !hasSampleLine;
      toggle.parentElement.style.opacity = hasSampleLine ? "" : "0.5";
      toggle.parentElement.title = hasSampleLine
        ? "Show every cell of the saved winning line on the board"
        : "No sample line was saved with this result";
    }
    renderProofExplorer();
    requestAnimationFrame(() => {
      proofFitBoard();
      document.getElementById("proof-close-btn").focus();
    });
  } catch (error) {
    setForcingStatus(`This result could not be opened: ${error.message || error}`, "error");
  }
}

function proofWorkerPosition(bundle) {
  const position = normalizeProofPosition(bundle.position);
  const stonesFlat = position.stones.flatMap(stone => [
    Number(stone[0]), Number(stone[1]), stone[2] === "P1" ? 1 : 2,
  ]);
  return {
    winLength: Number(position.config.win_length),
    placementRadius: Number(position.config.placement_radius),
    maxMoves: Number(position.config.max_moves),
    toMove: position.attacker,
    movesRemaining: Number(position.placements_remaining),
    stonesFlat,
  };
}

function verificationSummariesMatch(expected, actual) {
  return expected && actual
    && Number(expected.dagNodes) === Number(actual.dagNodes)
    && Number(expected.proofEdges) === Number(actual.proofEdges)
    && Number(expected.maxAttackerTurns) === Number(actual.maxAttackerTurns);
}

function verifySavedProof(bundle) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(forcingWorkerUrl(), {type: "module", name: "strix-proof-verifier"});
    const requestId = `verify-${++proofVerificationSerial}`;
    const finish = callback => value => {
      worker.terminate();
      callback(value);
    };
    worker.onmessage = event => {
      const message = event.data || {};
      if (message.requestId !== requestId) return;
      if (message.type === "error") {
        finish(reject)(new Error(message.error || "The saved replies could not be checked."));
        return;
      }
      if (message.type !== "verified") return;
      if (!verificationSummariesMatch(bundle.verification, message.summary)) {
        finish(reject)(new Error("The saved summary does not match the replies checked in this browser."));
        return;
      }
      finish(resolve)(message.summary);
    };
    worker.onerror = event => finish(reject)(
      new Error(event.message || "This browser stopped while checking the saved replies."),
    );
    try {
      worker.postMessage({
        type: "verify",
        requestId,
        position: proofWorkerPosition(bundle),
        certificate: bundle.certificate,
      });
    } catch (error) {
      finish(reject)(error);
    }
  });
}

async function loadSavedProof(proofId) {
  setForcingStatus("Loading the saved result…");
  try {
    const response = await fetch(`${URL_PREFIX}/api/proofs/${encodeURIComponent(proofId)}`);
    if (!response.ok) {
      throw new Error(response.status === 404
        ? "That saved result is no longer available on this server."
        : `The server returned HTTP ${response.status}.`);
    }
    const bundle = await response.json();
    setForcingStatus("Checking every saved reply in your browser…");
    const summary = await verifySavedProof(bundle);
    const optimization = proofOptimizationDescription(bundle);
    setForcingStatus(
      `Saved result checked: ${summary.dagNodes.toLocaleString()} positions, `
      + `with a win within ${summary.maxAttackerTurns} turns by the winning side.`
      + (optimization ? ` ${optimization.short}.` : ""),
      "win",
    );
    openProofExplorerBundle(bundle, {savedUrl: location.href.split("#")[0]});
  } catch (error) {
    setForcingStatus(`The saved result could not be opened: ${error.message || error}`, "error");
  }
}

async function copyProofText(text) {
  // Try the async Clipboard API first. It is the only path that works in
  // cross-origin iframes and in most secure contexts, but it rejects with
  // NotAllowedError when the user has blocked clipboard access, the document
  // is not focused, or the call loses its user-activation window — in which
  // case the legacy textarea + execCommand fallback below still succeeds on
  // every major browser.
  if (navigator.clipboard && navigator.clipboard.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch (_error) {
      // Fall through to the legacy path.
    }
  }
  const field = document.createElement("textarea");
  field.value = text;
  field.setAttribute("readonly", "");
  // Keep the field off-screen and out of the viewport's overflow scroll so
  // the user cannot accidentally land focus on it; `position: fixed` plus
  // negative offsets is the widely-recommended recipe.
  field.style.position = "fixed";
  field.style.top = "0";
  field.style.left = "0";
  field.style.width = "1px";
  field.style.height = "1px";
  field.style.opacity = "0";
  field.style.pointerEvents = "none";
  document.body.appendChild(field);
  const previousSelection = document.getSelection();
  const previousRange = previousSelection && previousSelection.rangeCount > 0
    ? previousSelection.getRangeAt(0) : null;
  field.focus({preventScroll: true});
  field.select();
  field.setSelectionRange(0, field.value.length);
  let copied = false;
  try {
    copied = document.execCommand("copy");
  } finally {
    field.remove();
    if (previousRange && previousSelection) {
      previousSelection.removeAllRanges();
      previousSelection.addRange(previousRange);
    }
  }
  if (!copied) throw new Error("Your browser blocked clipboard access.");
}

function setProofShareUi(message, state = "") {
  const status = document.getElementById("proof-share-status");
  if (status) {
    status.textContent = message;
    if (state) status.dataset.state = state;
    else delete status.dataset.state;
  }
  for (const id of ["analysis-share-certificate-btn", "proof-share-btn"]) {
    const button = document.getElementById(id);
    if (button) button.textContent = message === "Link copied" ? "Link copied"
      : message === "Saving…" ? "Saving…" : "Copy result link";
  }
}

async function shareForcingCertificate() {
  const bundle = currentProofBundle();
  if (!bundle) return;
  const buttons = ["analysis-share-certificate-btn", "proof-share-btn"]
    .map(id => document.getElementById(id)).filter(Boolean);
  buttons.forEach(button => { button.disabled = true; });
  setProofShareUi("Saving…");
  try {
    let savedUrl = proofSavedUrls.get(bundle);
    if (!savedUrl) {
      const response = await fetch(`${URL_PREFIX}/api/proofs`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(bundle),
      });
      let result = {};
      try { result = await response.json(); } catch (_error) {}
      if (!response.ok) throw new Error(result.error || `The server returned HTTP ${response.status}.`);
      savedUrl = new URL(result.url, location.origin).href;
      proofSavedUrls.set(bundle, savedUrl);
    }
    await copyProofText(savedUrl);
    setProofShareUi("Link copied", "ok");
  } catch (error) {
    setProofShareUi(`Share failed: ${error.message || error}`, "error");
  } finally {
    buttons.forEach(button => { button.disabled = false; });
  }
}

function closeProofExplorer() {
  const explorer = document.getElementById("proof-explorer");
  if (explorer) explorer.hidden = true;
  document.body.classList.remove("proof-explorer-open");
  proofExplorerState = null;
  proofPreviewHighlight = null;
  proofPan = null;
  if (proofPreviousFocus && typeof proofPreviousFocus.focus === "function") proofPreviousFocus.focus();
  proofPreviousFocus = null;
}

function currentProofEntry() {
  if (proofExplorerState && proofExplorerState.mode === "shortest") {
    return proofExplorerState.sampleLine.entries[proofExplorerState.lineIndex];
  }
  return proofExplorerState.history[proofExplorerState.history.length - 1];
}

function formatProofAction(action) {
  return action.map(cell => `[${cell[0]},${cell[1]}]`).join(" + ");
}

function proofPlayerClass(player) {
  return player === "P1" ? "P1 · orange" : "P2 · blue";
}

function proofActionsMatch(left, right) {
  return Array.isArray(left) && Array.isArray(right)
    && left.length === right.length
    && left.every((cell, index) => cell[0] === right[index][0] && cell[1] === right[index][1]);
}

function proofTreeBranchHtml({letter, action, copy, bound, color, onclick, badge = "", badgeClass = ""}) {
  const preview = `proofPreviewAction(${JSON.stringify(action)},'${color}')`;
  return `<button class="proof-tree-node future" role="treeitem" style="--choice-color:${color}" onclick="${onclick}"`
    + ` onpointerenter="${preview}" onpointerleave="proofClearPreview()"`
    + ` onfocus="${preview}" onblur="proofClearPreview()">`
    + `<span class="proof-tree-marker"><span>${letter}</span></span>`
    + `<span class="proof-tree-copy"><b>${formatProofAction(action)}</b><small>${copy}</small></span>`
    + `<span class="proof-tree-bound">within ${bound} turn${bound === 1 ? "" : "s"}`
    + (badge ? `<span class="proof-tree-badge ${badgeClass}">${badge}</span>` : "")
    + `</span></button>`;
}

function proofTreePathNodeHtml(entry, index, currentIndex, allowFuture) {
  const current = index === currentIndex;
  const future = index > currentIndex;
  const label = index === 0 ? "Starting position" : entry.label;
  const action = entry.lastAction ? formatProofAction(entry.lastAction.action) : "Proof begins here";
  const onclick = current ? "" : ` onclick="proofExplorerJump(${index})"`;
  return `<button class="proof-tree-node ${current ? "current" : future ? "future" : "past"}" role="treeitem"`
    + ` aria-level="${index + 1}"${current ? ` aria-current="step" data-proof-selected disabled` : ""}`
    + ((!future || allowFuture) ? onclick : " disabled") + `>`
    + `<span class="proof-tree-marker"><span>${current ? "●" : index + 1}</span></span>`
    + `<span class="proof-tree-copy"><b>${label}</b><small>${action}</small></span>`
    + `<span class="proof-tree-bound">${current ? "current" : future ? "later" : "return"}</span>`
    + `</button>`;
}

function proofTreeSiblingHtml(parentIndex, selectedEntry) {
  const state = proofExplorerState;
  const parent = state.history[parentIndex];
  const node = state.model.nodes[parent.nodeId];
  const edges = proofNodeEdges(node);
  const alternatives = edges.filter(edge => !(
    edge.child === selectedEntry.nodeId
    && selectedEntry.lastAction
    && proofActionsMatch(edge.action, selectedEntry.lastAction.action)
  ));
  if (!alternatives.length) return "";
  const noun = alternatives.length === 1 ? "branch" : "branches";
  const buttons = alternatives.map(edge => `<button class="proof-tree-sibling" role="treeitem"`
    + ` onclick="proofExplorerChooseSibling(${parentIndex},${edge.index})">`
    + `${formatProofAction(edge.action)}</button>`).join("");
  return `<details class="proof-tree-siblings" ontoggle="proofTreeDetailsToggled()">`
    + `<summary>${alternatives.length} other ${noun} here</summary>`
    + `<div class="proof-tree-sibling-list">${buttons}</div></details>`;
}

function centerSelectedProofNode(behavior = "auto") {
  const tree = document.getElementById("proof-tree");
  const selected = tree && tree.querySelector("[data-proof-selected]");
  if (!tree || !selected) return;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const treeRect = tree.getBoundingClientRect();
  const selectedRect = selected.getBoundingClientRect();
  const top = tree.scrollTop + selectedRect.top - treeRect.top
    - (tree.clientHeight - selectedRect.height) / 2;
  if (typeof tree.scrollTo === "function") {
    tree.scrollTo({top: Math.max(0, top), behavior: reduceMotion ? "auto" : behavior});
  } else {
    tree.scrollTop = Math.max(0, top);
  }
}

function proofTreeDetailsToggled() {
  requestAnimationFrame(() => centerSelectedProofNode("auto"));
}

function renderProofTree(branches = []) {
  const state = proofExplorerState;
  const tree = document.getElementById("proof-tree");
  proofPreviewHighlight = null;
  if (state.mode === "shortest") {
    const entries = state.sampleLine.entries;
    tree.innerHTML = `<div class="proof-tree-inner"><div class="proof-tree-trunk">`
      + entries.map((entry, index) => proofTreePathNodeHtml(entry, index, state.lineIndex, true)).join("")
      + `</div></div>`;
  } else {
    const history = state.history;
    const path = history.map((entry, index) => {
      const siblings = index + 1 < history.length
        ? proofTreeSiblingHtml(index, history[index + 1]) : "";
      return proofTreePathNodeHtml(entry, index, history.length - 1, false) + siblings;
    }).join("");
    tree.innerHTML = `<div class="proof-tree-inner"><div class="proof-tree-trunk">${path}</div>`
      + (branches.length ? `<div class="proof-tree-children">${branches.map(proofTreeBranchHtml).join("")}</div>` : "")
      + `</div>`;
  }
  requestAnimationFrame(() => centerSelectedProofNode());
}

function proofStepTags(entry, node, remaining) {
  const model = proofExplorerState.model;
  const tags = [
    `<span class="proof-step-tag strong">win within ${remaining} turn${remaining === 1 ? "" : "s"} by the winning side</span>`,
  ];
  if (node.kind === "defender_replies") {
    tags.push(`<span class="proof-step-tag">all ${node.responses.length} checked replies shown</span>`);
  } else if (node.kind === "attacker_move") {
    const count = model.attackerChoices(node).length;
    tags.push(`<span class="proof-step-tag">${count} winning move${count === 1 ? "" : "s"} found</span>`);
  }
  return `<div class="proof-step-tags">${tags.join("")}</div>`;
}

function renderProofExplorer() {
  if (!proofExplorerState) return;
  if (proofExplorerState.mode === "shortest") {
    renderShortestSampleLine();
    return;
  }
  const {model, attacker, defender, history} = proofExplorerState;
  const entry = currentProofEntry();
  const node = model.nodes[entry.nodeId];
  const remaining = entry.shownWin ? 0 : model.remaining(entry.nodeId);
  const summary = document.getElementById("proof-explorer-summary");
  const optimization = proofOptimizationDescription(proofExplorerState.bundle);
  summary.textContent = (optimization ? `${optimization.short} · ` : "")
    + `${model.nodes.length.toLocaleString()} positions checked · `
    + `win within ${model.maxAttackerTurns} turns by the winning side`;
  const optimizationNote = document.getElementById("proof-optimization-note");
  optimizationNote.hidden = !optimization;
  optimizationNote.textContent = optimization ? optimization.full : "";
  document.getElementById("proof-attacker-swatch").style.background = attacker === "P1" ? "#f08a3c" : "#3fb6d9";
  document.getElementById("proof-defender-swatch").style.background = defender === "P1" ? "#f08a3c" : "#3fb6d9";
  document.getElementById("proof-attacker-legend").textContent = `${attacker} · winning side`;
  document.getElementById("proof-defender-legend").textContent = `${defender} · defending side`;
  document.getElementById("proof-back-btn").disabled = history.length <= 1;
  document.getElementById("proof-node-label").textContent = entry.shownWin
    ? "win complete" : `step ${history.length}`;
  document.getElementById("proof-progress-label").textContent = entry.shownWin
    ? `${entry.attackerTurnsPlayed} winning turn${entry.attackerTurnsPlayed === 1 ? "" : "s"} · complete`
    : `${entry.attackerTurnsPlayed} winning turn${entry.attackerTurnsPlayed === 1 ? "" : "s"} played · ${remaining} remain`;
  const progress = entry.shownWin ? 100
    : Math.min(98, entry.attackerTurnsPlayed / model.maxAttackerTurns * 100);
  document.getElementById("proof-progress-bar").style.transform = `scaleX(${progress / 100})`;

  const card = document.getElementById("proof-step-card");
  const worstButton = document.getElementById("proof-worst-btn");
  const shortestButton = document.getElementById("proof-shortest-line-btn");
  shortestButton.hidden = !proofExplorerState.sampleLine;
  const sampleAttainsBound = proofExplorerState.sampleLine
    && proofExplorerState.sampleLine.entries.at(-1).attackerTurnsPlayed
      === Number(proofExplorerState.bundle.optimization.bestUpperDepth);
  shortestButton.textContent = sampleAttainsBound ? "Best defence line" : "Example winning line";
  let cardHtml = "";
  let branches = [];
  let accent = "var(--brass)";

  if (entry.shownWin) {
    accent = "var(--good)";
    cardHtml = `<div class="proof-step-kicker">Win complete</div>`
      + `<div class="proof-step-title">The winning line is complete</div>`
      + `<div class="proof-step-copy">The highlighted move completes six in a row. Go back to try another reply, or start again from the first position.</div>`;
    worstButton.disabled = true;
    worstButton.textContent = "Win complete";
  } else if (node.kind === "attacker_move") {
    accent = attacker === "P1" ? "var(--p1)" : "var(--p2)";
    const attacks = model.attackerChoices(node);
    const hasAlternatives = attacks.length > 1;
    cardHtml = `<div class="proof-step-kicker">Winning side · ${proofPlayerClass(attacker)}</div>`
      + `<div class="proof-step-title">${hasAlternatives ? `${attacks.length} winning moves found` : "Winning move found"}</div>`
      + (hasAlternatives
        ? `<div class="proof-step-copy">Each move shown can force a win. Moves that guarantee a quicker win appear first. The search may not have saved every winning move.</div>`
        : `<div class="proof-step-copy">The search saved one move that can force a win here. Other winning moves may exist.</div>`)
      + proofStepTags(entry, node, remaining);
    const attackBounds = attacks.map(choice => 1 + model.remaining(choice.child));
    branches = attacks.map((choice, index) => {
      const bound = attackBounds[index];
      const tiedShortest = index > 0 && bound === attackBounds[0];
      return {
        letter: String(index + 1), action: choice.action,
        color: index === 0 ? (attacker === "P1" ? "#f08a3c" : "#3fb6d9")
          : PROOF_CHOICE_COLORS[index % PROOF_CHOICE_COLORS.length],
        copy: index === 0
          ? "Recommended · guarantees the quickest win found"
          : tiedShortest ? "Also guarantees the quickest win found"
            : "Also wins · may take longer",
        bound,
        badge: index === 0 ? "recommended" : tiedShortest ? "equally quick" : "",
        badgeClass: "recommended",
        onclick: `proofExplorerPlayAttacker(${index})`,
      };
    });
    worstButton.disabled = false;
    worstButton.textContent = "Play recommended move →";
  } else if (node.kind === "defender_replies") {
    accent = defender === "P1" ? "var(--p1)" : "var(--p2)";
    const plural = node.responses.length === 1 ? "response" : "responses";
    cardHtml = `<div class="proof-step-kicker">Defending side · ${proofPlayerClass(defender)}</div>`
      + `<div class="proof-step-title">Choose a defence</div>`
      + `<div class="proof-step-copy">The search checked all ${node.responses.length} ${plural}. A best defence is the strongest refutation here: it forces the winning side to take the longest proved continuation.</div>`
      + proofStepTags(entry, node, remaining);
    const worstIndex = model.worstResponseIndex(node);
    const worstBound = model.remaining(node.responses[worstIndex].child);
    const worstTies = node.responses.filter(response => model.remaining(response.child) === worstBound).length;
    branches = node.responses.map((response, index) => {
      const bound = model.remaining(response.child);
      const isWorst = bound === worstBound;
      return {
        letter: String.fromCharCode(65 + index), action: response.action,
        color: PROOF_CHOICE_COLORS[index % PROOF_CHOICE_COLORS.length],
        copy: isWorst
          ? (worstTies > 1 ? "Also a best defence · delays the win longest" : "Best defence · delays the win longest")
          : "The winning side can answer this too",
        bound, badge: isWorst ? "best defence" : "", badgeClass: "longest",
        onclick: `proofExplorerChooseDefense(${index})`,
      };
    });
    worstButton.disabled = false;
    worstButton.textContent = "Choose best defence →";
  } else if (node.kind === "immediate_win") {
    accent = "var(--good)";
    cardHtml = `<div class="proof-step-kicker">Winning side · ${proofPlayerClass(attacker)}</div>`
      + `<div class="proof-step-title">Complete the six</div>`
      + `<div class="proof-step-copy">The winning side can complete six in a row during this turn. Show the move on the board to finish this line.</div>`
      + proofStepTags(entry, node, remaining);
    branches = [{
      letter: "✓", action: node.action, color: "#79cf9a", copy: "Show the move that completes the win",
      bound: 1, onclick: "proofExplorerPlayAttacker()",
    }];
    worstButton.disabled = false;
    worstButton.textContent = "Show winning move →";
  } else if (node.kind === "unstoppable") {
    accent = "var(--good)";
    const threats = Array.isArray(node.threats) ? node.threats : [];
    const title = threats.length
      ? `${threats.length} winning threat${threats.length === 1 ? "" : "s"}`
      : "Win cannot be blocked";
    const copy = threats.length
      ? `The marked shapes show checked ways for ${attacker} to complete six. Stopping all of them needs at least three placements, but ${defender} has two.`
      : `${defender} cannot block every way for ${attacker} to make six on its next turn.`;
    cardHtml = `<div class="proof-step-kicker">No reply can stop the win</div>`
      + `<div class="proof-step-title">${title}</div>`
      + `<div class="proof-step-copy">${copy}</div>`
      + proofStepTags(entry, node, remaining);
    worstButton.disabled = true;
    worstButton.textContent = "No defense remains";
  }
  card.style.setProperty("--proof-accent", accent);
  card.innerHTML = cardHtml;
  renderProofTree(branches);
  drawProofBoard();
}

function renderShortestSampleLine() {
  const state = proofExplorerState;
  const {model, attacker, defender, sampleLine, lineIndex} = state;
  const entry = currentProofEntry();
  const total = sampleLine.turns.length;
  const next = lineIndex < total ? sampleLine.turns[lineIndex] : null;
  const optimization = state.bundle.optimization;
  const sampleAttackerTurns = sampleLine.entries[total].attackerTurnsPlayed;
  const attainsBound = sampleAttackerTurns === Number(optimization.bestUpperDepth);
  document.getElementById("proof-explorer-summary").textContent =
    `shortest win ${optimization.bestUpperDepth} turns · ${attainsBound ? "best defence" : "example"} uses ${sampleAttackerTurns} turns by the winning side · ${total} total turns`;
  const note = document.getElementById("proof-optimization-note");
  note.hidden = false;
  note.textContent = attainsBound
    ? `This is the best-defence line: the winning side chooses its quickest proved win while the defender always chooses a reply that delays it longest. It shows why no listed defence can extend the win beyond ${optimization.bestUpperDepth} turns.`
    : `This is one example winning line. It finishes after ${sampleAttackerTurns} turns by the winning side because these replies do not delay the win as long as possible. Against any reply, the win takes no more than ${optimization.bestUpperDepth} turns.`;
  document.getElementById("proof-attacker-swatch").style.background = attacker === "P1" ? "#f08a3c" : "#3fb6d9";
  document.getElementById("proof-defender-swatch").style.background = defender === "P1" ? "#f08a3c" : "#3fb6d9";
  document.getElementById("proof-attacker-legend").textContent = `${attacker} · winning side`;
  document.getElementById("proof-defender-legend").textContent = `${defender} · defending side`;
  document.getElementById("proof-back-btn").disabled = lineIndex === 0;
  document.getElementById("proof-node-label").textContent = `turn ${lineIndex} of ${total}`;
  document.getElementById("proof-progress-label").textContent = next
    ? `${entry.attackerTurnsPlayed} winning turn${entry.attackerTurnsPlayed === 1 ? "" : "s"} played`
    : `${sampleAttackerTurns} winning turn${sampleAttackerTurns === 1 ? "" : "s"} · line complete`;
  document.getElementById("proof-progress-bar").style.transform = `scaleX(${total ? lineIndex / total : 0})`;

  const shortestButton = document.getElementById("proof-shortest-line-btn");
  shortestButton.hidden = false;
  shortestButton.textContent = "Return to all replies";
  const worstButton = document.getElementById("proof-worst-btn");
  worstButton.disabled = !next;
  worstButton.textContent = next ? "Next turn →" : "Line complete";
  const card = document.getElementById("proof-step-card");
  if (next) {
    const role = next.player === attacker ? "Attacker" : "Defender";
    card.style.setProperty("--proof-accent", next.player === "P1" ? "var(--p1)" : "var(--p2)");
    card.innerHTML = `<div class="proof-step-kicker">${attainsBound ? "Best defence line" : "Example winning line"} · ${role === "Attacker" ? "winning side" : "defending side"}</div>`
      + `<div class="proof-step-title">Play turn ${lineIndex + 1}</div>`
      + `<div class="proof-step-copy">${attainsBound ? "The winning side chooses its quickest proved win. The other side chooses the reply that delays it longest." : "This is one example. The defending side may have another reply that delays the win longer."}</div>`;
  } else {
    card.style.setProperty("--proof-accent", "var(--good)");
    card.innerHTML = `<div class="proof-step-kicker">${attainsBound ? "Best defence line complete" : "Example line complete"}</div>`
      + `<div class="proof-step-title">Winning line complete</div>`
      + `<div class="proof-step-copy">The final highlighted move completes the win. Return to all replies to try other defensive moves.</div>`;
  }
  renderProofTree();
  drawProofBoard();
}

function proofExplorerToggleShortestLine() {
  if (!proofExplorerState || !proofExplorerState.sampleLine) return;
  proofExplorerState.mode = proofExplorerState.mode === "shortest" ? "proof" : "shortest";
  renderProofExplorer();
  requestAnimationFrame(proofFitBoard);
}

function proofExplorerShortestNext() {
  const state = proofExplorerState;
  if (!state || state.mode !== "shortest" || state.lineIndex >= state.sampleLine.turns.length) return;
  state.lineIndex++;
  renderProofExplorer();
  requestAnimationFrame(proofKeepHighlightVisible);
}

function proofAdvance(child, action, player, label, shownWin = false) {
  const state = proofExplorerState;
  const entry = currentProofEntry();
  try {
    const stones = applyProofAction(entry.stones, action, player);
    state.history.push({
      nodeId: child == null ? entry.nodeId : child,
      stones,
      attackerTurnsPlayed: entry.attackerTurnsPlayed + (player === state.attacker ? 1 : 0),
      label,
      lastAction: {action, player},
      shownWin,
    });
    renderProofExplorer();
    requestAnimationFrame(proofKeepHighlightVisible);
  } catch (error) {
    closeProofExplorer();
    setForcingStatus(`The saved moves could not be shown: ${error.message || error}`, "error");
  }
}

function proofExplorerPlayAttacker(index = 0) {
  if (!proofExplorerState) return;
  const entry = currentProofEntry();
  const node = proofExplorerState.model.nodes[entry.nodeId];
  if (node.kind === "attacker_move") {
    const choice = proofExplorerState.model.attackerChoices(node)[index];
    if (!choice) return;
    const turn = entry.attackerTurnsPlayed + 1;
    proofAdvance(choice.child, choice.action, proofExplorerState.attacker,
      `Winning turn ${turn} · move ${index + 1}`);
  } else if (node.kind === "immediate_win") {
    const turn = entry.attackerTurnsPlayed + 1;
    proofAdvance(null, node.action, proofExplorerState.attacker, `Win ${turn}`, true);
  }
}

function proofExplorerChooseDefense(index) {
  if (!proofExplorerState) return;
  const entry = currentProofEntry();
  const node = proofExplorerState.model.nodes[entry.nodeId];
  if (node.kind !== "defender_replies" || !node.responses[index]) return;
  const response = node.responses[index];
  proofAdvance(response.child, response.action, proofExplorerState.defender,
    `Reply ${String.fromCharCode(65 + index)}`);
}

function proofExplorerWorstCase() {
  if (!proofExplorerState) return;
  if (proofExplorerState.mode === "shortest") {
    proofExplorerShortestNext();
    return;
  }
  const entry = currentProofEntry();
  if (entry.shownWin) return;
  const node = proofExplorerState.model.nodes[entry.nodeId];
  if (node.kind === "attacker_move" || node.kind === "immediate_win") {
    proofExplorerPlayAttacker();
  } else if (node.kind === "defender_replies") {
    proofExplorerChooseDefense(proofExplorerState.model.worstResponseIndex(node));
  }
}

function proofExplorerBack() {
  if (!proofExplorerState) return;
  if (proofExplorerState.mode === "shortest") {
    if (proofExplorerState.lineIndex > 0) proofExplorerState.lineIndex--;
    renderProofExplorer();
    return;
  }
  if (proofExplorerState.history.length <= 1) return;
  proofExplorerState.history.pop();
  renderProofExplorer();
}

function proofExplorerReset() {
  if (!proofExplorerState) return;
  if (proofExplorerState.mode === "shortest") {
    proofExplorerState.lineIndex = 0;
    renderProofExplorer();
    requestAnimationFrame(proofFitBoard);
    return;
  }
  proofExplorerState.history.splice(1);
  renderProofExplorer();
  requestAnimationFrame(proofFitBoard);
}

function proofExplorerJump(index) {
  if (proofExplorerState && proofExplorerState.mode === "shortest") {
    if (index < 0 || index >= proofExplorerState.sampleLine.entries.length
        || index === proofExplorerState.lineIndex) return;
    proofExplorerState.lineIndex = index;
    renderProofExplorer();
    requestAnimationFrame(proofKeepHighlightVisible);
    return;
  }
  if (!proofExplorerState || index < 0 || index >= proofExplorerState.history.length - 1) return;
  proofExplorerState.history.splice(index + 1);
  renderProofExplorer();
}

function proofExplorerChooseSibling(parentIndex, edgeIndex) {
  const state = proofExplorerState;
  if (!state || state.mode !== "proof" || parentIndex < 0 || parentIndex >= state.history.length) return;
  state.history.splice(parentIndex + 1);
  const node = state.model.nodes[currentProofEntry().nodeId];
  if (node.kind === "attacker_move") proofExplorerPlayAttacker(edgeIndex);
  else if (node.kind === "defender_replies") proofExplorerChooseDefense(edgeIndex);
}

function proofPreviewAction(action, color) {
  if (!proofExplorerState || proofExplorerState.mode !== "proof") return;
  proofPreviewHighlight = {action, color};
  drawProofBoard();
}

function proofClearPreview() {
  if (!proofExplorerState || !proofPreviewHighlight) return;
  proofPreviewHighlight = null;
  drawProofBoard();
}

function proofAvailableHighlights(node, entry) {
  if (entry.shownWin) return [];
  if (node.kind === "attacker_move") {
    return proofExplorerState.model.attackerChoices(node).map((choice, index) => ({
      action: choice.action,
      color: index === 0
        ? (proofExplorerState.attacker === "P1" ? "#f08a3c" : "#3fb6d9")
        : PROOF_CHOICE_COLORS[index % PROOF_CHOICE_COLORS.length],
    }));
  }
  if (node.kind === "immediate_win") {
    return [{action: node.action, color: proofExplorerState.attacker === "P1" ? "#f08a3c" : "#3fb6d9"}];
  }
  if (node.kind === "defender_replies") {
    return node.responses.map((response, index) => ({
      action: response.action,
      color: PROOF_CHOICE_COLORS[index % PROOF_CHOICE_COLORS.length],
    }));
  }
  if (node.kind === "unstoppable" && Array.isArray(node.threats)) {
    return node.threats.map(action => ({action, color: "#79cf9a", terminal: true}));
  }
  return [];
}

function proofWinningLineSvg(stones, winner, winLength) {
  const axes = [[1, 0], [0, 1], [1, -1]];
  const won = new Set();
  for (const [key, player] of stones) if (player === winner) won.add(key);
  let line = null;
  for (const [dq, dr] of axes) {
    for (const key of won) {
      const [sq, sr] = key.split(",").map(Number);
      if (won.has(`${sq - dq},${sr - dr}`)) continue;
      const run = [];
      let q = sq, r = sr;
      while (won.has(`${q},${r}`)) { run.push([q, r]); q += dq; r += dr; }
      if (run.length >= winLength) { line = run.slice(0, winLength); break; }
    }
    if (line) break;
  }
  if (!line) return "";
  const ring = winner === "P1" ? "#ffe1b0" : "#c8f1fd";
  return line.map(([q, r]) => {
    const point = axialToPixel(q, r);
    return `<polygon points="${hexCorners(point.x, point.y, .82)}" fill="none" stroke="${ring}" stroke-width="3" filter="url(#proof-win-glow)" pointer-events="none"/>`;
  }).join("");
}

function drawProofBoard() {
  if (!proofExplorerState) return;
  const entry = currentProofEntry();
  const node = proofExplorerState.model.nodes[entry.nodeId];
  const lineTurn = proofExplorerState.mode === "shortest"
    ? proofExplorerState.sampleLine.turns[proofExplorerState.lineIndex] : null;
  const terminalHighlights = proofExplorerState.mode === "proof" && node.kind === "unstoppable"
    ? proofAvailableHighlights(node, entry) : [];
  const highlights = lineTurn
    ? [{action: lineTurn.cells, color: lineTurn.player === "P1" ? "#f08a3c" : "#3fb6d9"}]
    : proofExplorerState.mode === "shortest" ? []
      : proofPreviewHighlight ? [proofPreviewHighlight] : terminalHighlights;
  const availableHighlights = proofExplorerState.mode === "proof"
    ? proofAvailableHighlights(node, entry) : highlights;
  const cellSet = new Set();
  const seeds = [];
  for (const key of entry.stones.keys()) {
    const [q, r] = key.split(",").map(Number);
    seeds.push([q, r]);
  }
  for (const highlight of availableHighlights) seeds.push(...highlight.action);
  if (!seeds.length) seeds.push([0, 0]);
  const contextRadius = 3;
  for (const [q, r] of seeds) {
    for (let dq = -contextRadius; dq <= contextRadius; dq++) {
      for (let dr = -contextRadius; dr <= contextRadius; dr++) {
        if ((Math.abs(dq) + Math.abs(dr) + Math.abs(dq + dr)) / 2 <= contextRadius) {
          cellSet.add(`${q + dq},${r + dr}`);
        }
      }
    }
  }

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  let body = `<defs><filter id="proof-win-glow" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="2.6" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>`;
  body += `<g id="proof-board-group">`;
  for (const key of cellSet) {
    const [q, r] = key.split(",").map(Number);
    const point = axialToPixel(q, r);
    minX = Math.min(minX, point.x - S); maxX = Math.max(maxX, point.x + S);
    minY = Math.min(minY, point.y - S); maxY = Math.max(maxY, point.y + S);
    const player = entry.stones.get(key);
    const cls = player === "P1" ? "proof-hex proof-hex-p1"
      : player === "P2" ? "proof-hex proof-hex-p2" : "proof-hex proof-hex-empty";
    body += `<polygon class="${cls}" points="${hexCorners(point.x, point.y)}"><title>[${q},${r}]${player ? ` · ${player}` : ""}</title></polygon>`;
  }

  if (entry.lastAction) {
    const color = entry.lastAction.player === "P1" ? "#ffd7a8" : "#b3ecfb";
    for (const [q, r] of entry.lastAction.action) {
      const point = axialToPixel(q, r);
      body += `<polygon points="${hexCorners(point.x, point.y, .76)}" fill="none" stroke="${color}" stroke-width="2.7" pointer-events="none"/>`;
    }
  }

  const focusPoints = availableHighlights.flatMap(highlight => highlight.action.map(([q, r]) => axialToPixel(q, r)));
  const drawnMarkers = new Set();
  for (const highlight of highlights) {
    if (highlight.action.length === 2) {
      const a = axialToPixel(highlight.action[0][0], highlight.action[0][1]);
      const b = axialToPixel(highlight.action[1][0], highlight.action[1][1]);
      body += `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="${highlight.color}" stroke-width="${S * .09}" stroke-dasharray="${S * .18} ${S * .13}" opacity=".7" pointer-events="none"/>`;
    }
    highlight.action.forEach(([q, r]) => {
      const markerKey = `${q},${r}`;
      if (drawnMarkers.has(markerKey)) return;
      drawnMarkers.add(markerKey);
      const point = axialToPixel(q, r);
      const title = highlight.terminal ? `<title>Winning threat at [${q},${r}]</title>` : "";
      body += `<polygon class="proof-preview-hex${highlight.terminal ? " proof-terminal-threat" : ""}" points="${hexCorners(point.x, point.y, .56)}" fill="#0d0f0e" fill-opacity=".82" stroke="${highlight.color}" stroke-width="2.8" pointer-events="none">${title}</polygon>`;
      body += `<circle class="proof-preview-dot" cx="${point.x}" cy="${point.y}" r="${S * .085}" fill="${highlight.color}" pointer-events="none"/>`;
    });
  }
  if (entry.shownWin) {
    const winLength = Number((proofExplorerState.position.config || {}).win_length || 6);
    body += proofWinningLineSvg(entry.stones, proofExplorerState.attacker, winLength);
  }
  body += proofSampleLineOverlaySvg();
  body += `</g>`;
  document.getElementById("proof-board").innerHTML = body;
  proofBoardBounds = {minX, minY, maxX, maxY};
  proofHighlightCenter = focusPoints.length ? {
    x: focusPoints.reduce((sum, point) => sum + point.x, 0) / focusPoints.length,
    y: focusPoints.reduce((sum, point) => sum + point.y, 0) / focusPoints.length,
  } : null;
  updateProofTransform();
}

function updateProofTransform() {
  const group = document.getElementById("proof-board-group");
  const container = document.getElementById("proof-board-container");
  if (!group || !container) return;
  group.setAttribute("transform",
    `translate(${container.clientWidth / 2 + proofView.x},${container.clientHeight / 2 + proofView.y}) scale(${proofView.scale})`);
}

function proofFitBoard() {
  const container = document.getElementById("proof-board-container");
  if (!proofExplorerState || !container || !proofBoardBounds || !container.clientWidth) return;
  const width = Math.max(1, proofBoardBounds.maxX - proofBoardBounds.minX);
  const height = Math.max(1, proofBoardBounds.maxY - proofBoardBounds.minY);
  const scale = Math.max(.08, Math.min(1.35,
    (container.clientWidth - 50) / width,
    (container.clientHeight - 60) / height));
  const centerX = (proofBoardBounds.minX + proofBoardBounds.maxX) / 2;
  const centerY = (proofBoardBounds.minY + proofBoardBounds.maxY) / 2;
  proofView = {x: -centerX * scale, y: -centerY * scale, scale};
  updateProofTransform();
}

function proofKeepHighlightVisible() {
  if (!proofExplorerState || !proofHighlightCenter) return;
  const container = document.getElementById("proof-board-container");
  const x = container.clientWidth / 2 + proofView.x + proofView.scale * proofHighlightCenter.x;
  const y = container.clientHeight / 2 + proofView.y + proofView.scale * proofHighlightCenter.y;
  const marginX = Math.min(120, container.clientWidth * .2);
  const marginY = Math.min(100, container.clientHeight * .22);
  let dx = 0, dy = 0;
  if (x < marginX) dx = marginX - x;
  else if (x > container.clientWidth - marginX) dx = container.clientWidth - marginX - x;
  if (y < marginY) dy = marginY - y;
  else if (y > container.clientHeight - marginY) dy = container.clientHeight - marginY - y;
  if (dx || dy) {
    proofView.x += dx; proofView.y += dy;
    updateProofTransform();
  }
}

function proofZoom(factor, anchorX = null, anchorY = null) {
  if (!proofExplorerState) return;
  const container = document.getElementById("proof-board-container");
  const oldScale = proofView.scale;
  const newScale = Math.max(.08, Math.min(5, oldScale * factor));
  const x = anchorX == null ? container.clientWidth / 2 : anchorX;
  const y = anchorY == null ? container.clientHeight / 2 : anchorY;
  const cx = container.clientWidth / 2, cy = container.clientHeight / 2;
  const worldX = (x - cx - proofView.x) / oldScale;
  const worldY = (y - cy - proofView.y) / oldScale;
  proofView.x = x - cx - worldX * newScale;
  proofView.y = y - cy - worldY * newScale;
  proofView.scale = newScale;
  updateProofTransform();
}

function proofSetShowLine(visible) {
  proofShowLine = Boolean(visible);
  const toggle = document.getElementById("proof-show-line");
  if (toggle && toggle.checked !== proofShowLine) toggle.checked = proofShowLine;
  drawProofBoard();
}

function syncProofExplorerButtons(node) {
  const available = Boolean(node && node.result && node.result.forcing_certificate);
  for (const id of ["analysis-explore-certificate-btn", "analysis-share-certificate-btn",
                    "analysis-download-certificate-btn"]) {
    const button = document.getElementById(id);
    if (button) button.hidden = !available;
  }
}

if (typeof document !== "undefined") {
  const board = document.getElementById("proof-board");
  board.addEventListener("pointerdown", event => {
    if (!proofExplorerState || event.button !== 0) return;
    board.setPointerCapture(event.pointerId);
    proofPan = {id: event.pointerId, x: event.clientX, y: event.clientY, vx: proofView.x, vy: proofView.y};
    board.classList.add("panning");
  });
  board.addEventListener("pointermove", event => {
    if (!proofPan || proofPan.id !== event.pointerId) return;
    proofView.x = proofPan.vx + event.clientX - proofPan.x;
    proofView.y = proofPan.vy + event.clientY - proofPan.y;
    updateProofTransform();
  });
  const endPan = event => {
    if (!proofPan || proofPan.id !== event.pointerId) return;
    proofPan = null;
    board.classList.remove("panning");
  };
  board.addEventListener("pointerup", endPan);
  board.addEventListener("pointercancel", endPan);
  board.addEventListener("wheel", event => {
    if (!proofExplorerState) return;
    event.preventDefault();
    const rect = document.getElementById("proof-board-container").getBoundingClientRect();
    proofZoom(Math.exp(-event.deltaY * .0006), event.clientX - rect.left, event.clientY - rect.top);
  }, {passive: false});
  document.addEventListener("keydown", event => {
    const explorer = document.getElementById("proof-explorer");
    if (!explorer || explorer.hidden) return;
    if (event.key === "Escape") closeProofExplorer();
    else if (event.key === "ArrowLeft") proofExplorerBack();
    else if (event.key === "ArrowRight") proofExplorerWorstCase();
    else return;
    event.preventDefault();
  });
  window.addEventListener("resize", updateProofTransform);
  window.addEventListener("hexo:hex-size-changed", () => {
    if (!proofExplorerState) return;
    drawProofBoard();
    requestAnimationFrame(proofFitBoard);
  });
}

// Render the sample winning line as numbered stones on the proof board. Mirrors
// the analysis board's `forcing-pv*` styling: solid fills in the winner's stone
// colour for the attacker's cells, outlined in the defender's stone colour for
// the defender's forced replies. Skipped when the toggle is off or when no
// sample line was saved with the certificate (e.g. plain PDS-PN runs that
// produced a cert without a sample line).
function proofSampleLineOverlaySvg() {
  if (!proofShowLine) return "";
  const state = proofExplorerState;
  if (!state || !state.sampleLine || !state.sampleLine.turns.length) return "";
  const attacker = state.attacker;
  const attackerColor = attacker === "P2" ? "#3fb6d9" : "#f08a3c";
  const defenderColor = attacker === "P2" ? "#f08a3c" : "#3fb6d9";
  // The numbering mode is read live so flipping "Ply/Round" in the analysis
  // settings takes effect immediately on the proof board. Under "round" we group
  // placements by their owning turn (round k = turns 2k-2 + 2k-1) instead of
  // using a flat placement counter — that respects the actual moves_left per turn
  // (a turn may legitimately have 1 or 2 cells depending on the starting
  // moves_remaining) rather than assuming the standard 2-stone opener.
  const mode = (typeof positionNumbering === "function"
    && positionNumbering()) || ((typeof localStorage !== "undefined"
    && localStorage.getItem("hexo_analysis_numbering")) || "ply");
  const placed = new Set();
  let cells = "";
  let depth = 0;
  let turnIndex = -1;
  for (const turn of state.sampleLine.turns) {
    turnIndex++;
    for (const [q, r] of turn.cells) {
      const key = `${q},${r}`;
      if (placed.has(key)) { depth++; continue; }
      placed.add(key);
      const point = axialToPixel(q, r);
      const isAttacker = turn.player === attacker;
      const cls = isAttacker
        ? "forcing-pv forcing-pv-attacker"
        : "forcing-pv forcing-pv-defender";
      const labelCls = isAttacker
        ? "forcing-pv-label forcing-pv-label-attacker"
        : "forcing-pv-label forcing-pv-label-defender";
      // Round grouping: round 1 = turns 0+1, round 2 = turns 2+3, etc.
      // Ply grouping: each placement is a unique number across the whole line.
      const label = mode === "round"
        ? Math.floor(turnIndex / 2) + 1
        : depth + 1;
      const fontSize = Math.round(S * 0.42);
      if (isAttacker) {
        cells += `<circle class="${cls}" cx="${point.x}" cy="${point.y}" r="${S * 0.42}" fill="${attackerColor}"/>`;
      } else {
        cells += `<circle class="${cls}" cx="${point.x}" cy="${point.y}" r="${S * 0.42}" stroke="${defenderColor}"/>`;
      }
      cells += `<text class="${labelCls}" x="${point.x}" y="${point.y + 1}" font-size="${fontSize}" fill="${isAttacker ? "#0d0f0e" : defenderColor}">${label}</text>`;
      depth++;
    }
  }
  return `<g class="proof-sample-line">${cells}</g>`;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    buildProofModel, applyProofAction, normalizeProofPosition, proofAttackerChoices,
    copyProofText, proofOptimizationDescription, proofSampleLineOverlaySvg, proofSetShowLine,
  };
}
