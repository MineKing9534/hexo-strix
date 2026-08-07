// Interactive, board-first reader for verified PDS-PN proof certificates.
// The solver/verifier owns correctness; this file presents that immutable DAG
// as ranked, already-certified attacker choices followed by every exact
// defender response.

const PROOF_CHOICE_COLORS = ["#3fb6d9", "#a58ae8", "#e0a23a", "#79cf9a"];
let proofExplorerState = null;
let proofPreviousFocus = null;
let proofBoardBounds = null;
let proofHighlightCenter = null;
let proofView = {x: 0, y: 0, scale: 1};
let proofPan = null;
const proofSavedUrls = new WeakMap();
let proofVerificationSerial = 0;

function proofAttackerChoices(node) {
  if (node.kind !== "attacker_move") return [];
  const alternatives = node.alternatives == null ? [] : node.alternatives;
  if (!Array.isArray(alternatives)) {
    throw new Error("An attacker node has malformed alternatives.");
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
    throw new Error("This result does not contain a proof DAG.");
  }
  const nodes = certificate.nodes;
  const root = Number(certificate.root);
  if (!Number.isInteger(root) || root < 0 || root >= nodes.length) {
    throw new Error("The proof root is out of range.");
  }
  const depths = new Array(nodes.length);
  const visiting = new Set();
  const parents = new Uint32Array(nodes.length);
  let edges = 0;
  let alternativeAttackerNodes = 0;
  let attackerAlternatives = 0;

  for (let id = 0; id < nodes.length; id++) {
    if (nodes[id].kind === "attacker_move") {
      const alternatives = proofAttackerChoices(nodes[id]).length - 1;
      if (alternatives > 0) alternativeAttackerNodes++;
      attackerAlternatives += alternatives;
    }
    for (const child of proofNodeChildren(nodes[id])) {
      if (!Number.isInteger(child) || child < 0 || child >= nodes.length) {
        throw new Error(`Proof node ${id} has an invalid child.`);
      }
      parents[child]++;
      edges++;
    }
  }

  function remaining(id) {
    if (depths[id] !== undefined) return depths[id];
    if (visiting.has(id)) throw new Error("The proof contains a cycle.");
    visiting.add(id);
    const node = nodes[id];
    let depth;
    if (node.kind === "immediate_win" || node.kind === "unstoppable") {
      depth = 1;
    } else if (node.kind === "attacker_move") {
      const bounds = proofAttackerChoices(node).map(choice => 1 + remaining(choice.child));
      for (let index = 1; index < bounds.length; index++) {
        if (bounds[index] < bounds[index - 1]) {
          throw new Error(`Proof node ${id} has attacker choices that are not depth-ranked.`);
        }
      }
      depth = bounds[0];
    } else if (node.kind === "defender_replies" && node.responses.length) {
      depth = Math.max(...node.responses.map(response => remaining(response.child)));
    } else {
      throw new Error(`Proof node ${id} has an unknown or empty shape.`);
    }
    visiting.delete(id);
    depths[id] = depth;
    return depth;
  }

  const maxAttackerTurns = remaining(root);
  if (bundle.verification && bundle.verification.maxAttackerTurns != null
      && Number(bundle.verification.maxAttackerTurns) !== maxAttackerTurns) {
    throw new Error("The downloaded proof summary does not match its DAG.");
  }

  function worstResponseIndex(node) {
    let best = 0;
    for (let i = 1; i < node.responses.length; i++) {
      if (remaining(node.responses[i].child) > remaining(node.responses[best].child)) best = i;
    }
    return best;
  }

  function nearestAlternativePath(start) {
    const queue = [{id: start, path: []}];
    const seen = new Set([start]);
    for (let cursor = 0; cursor < queue.length; cursor++) {
      const current = queue[cursor];
      if (nodes[current.id].kind === "attacker_move"
          && proofAttackerChoices(nodes[current.id]).length > 1) {
        return current.path;
      }
      for (const edge of proofNodeEdges(nodes[current.id])) {
        if (seen.has(edge.child)) continue;
        seen.add(edge.child);
        queue.push({id: edge.child, path: [...current.path, edge]});
      }
    }
    return null;
  }

  return {
    certificate, nodes, root, parents, edges, remaining, worstResponseIndex,
    attackerChoices: proofAttackerChoices, nearestAlternativePath,
    alternativeAttackerNodes, attackerAlternatives, maxAttackerTurns,
  };
}

function applyProofAction(stones, action, player) {
  if (!Array.isArray(action) || action.length < 1 || action.length > 2) {
    throw new Error("A proof action must contain one or two placements.");
  }
  const next = new Map(stones);
  for (const cell of action) {
    if (!Array.isArray(cell) || cell.length !== 2
        || !Number.isInteger(cell[0]) || !Number.isInteger(cell[1])) {
      throw new Error("A proof placement has invalid coordinates.");
    }
    const key = `${cell[0]},${cell[1]}`;
    if (next.has(key)) throw new Error(`The proof tried to replay occupied cell ${key}.`);
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
  throw new Error("The proof bundle has no replayable root position.");
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
    proofExplorerState = {
      bundle, model, position, attacker,
      defender: attacker === "P1" ? "P2" : "P1",
      history: [rootEntry],
    };
    if (savedUrl) proofSavedUrls.set(bundle, savedUrl);
    proofPreviousFocus = document.activeElement;
    document.getElementById("proof-explorer").hidden = false;
    document.body.classList.add("proof-explorer-open");
    renderProofExplorer();
    requestAnimationFrame(() => {
      proofFitBoard();
      document.getElementById("proof-close-btn").focus();
    });
  } catch (error) {
    setForcingStatus(`Could not open proof: ${error.message || error}`, "error");
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
        finish(reject)(new Error(message.error || "Certificate verification failed."));
        return;
      }
      if (message.type !== "verified") return;
      if (!verificationSummariesMatch(bundle.verification, message.summary)) {
        finish(reject)(new Error("The saved proof summary does not match the verified certificate."));
        return;
      }
      finish(resolve)(message.summary);
    };
    worker.onerror = event => finish(reject)(
      new Error(event.message || "The proof-verification worker crashed."),
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
  setForcingStatus("Loading the saved proof…");
  try {
    const response = await fetch(`${URL_PREFIX}/api/proofs/${encodeURIComponent(proofId)}`);
    if (!response.ok) {
      throw new Error(response.status === 404
        ? "That saved proof does not exist on this server."
        : `The server returned HTTP ${response.status}.`);
    }
    const bundle = await response.json();
    setForcingStatus("Verifying every branch of the saved proof in your browser…");
    const summary = await verifySavedProof(bundle);
    setForcingStatus(
      `Verified saved proof: ${summary.dagNodes.toLocaleString()} nodes, `
      + `worst-case ${summary.maxAttackerTurns} attacker turns.`,
      "win",
    );
    openProofExplorerBundle(bundle, {savedUrl: location.href.split("#")[0]});
  } catch (error) {
    setForcingStatus(`Could not open saved proof: ${error.message || error}`, "error");
  }
}

async function copyProofText(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const field = document.createElement("textarea");
  field.value = text;
  field.setAttribute("readonly", "");
  field.style.position = "fixed";
  field.style.opacity = "0";
  document.body.appendChild(field);
  field.select();
  const copied = document.execCommand("copy");
  field.remove();
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
      : message === "Saving…" ? "Saving…" : "Share proof";
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
  proofPan = null;
  if (proofPreviousFocus && typeof proofPreviousFocus.focus === "function") proofPreviousFocus.focus();
  proofPreviousFocus = null;
}

function currentProofEntry() {
  return proofExplorerState.history[proofExplorerState.history.length - 1];
}

function formatProofAction(action) {
  return action.map(cell => `[${cell[0]},${cell[1]}]`).join(" + ");
}

function proofPlayerClass(player) {
  return player === "P1" ? "P1 · orange" : "P2 · blue";
}

function proofChoiceHtml({letter, action, copy, bound, color, onclick, badge = "", badgeClass = ""}) {
  return `<button class="proof-choice" style="--choice-color:${color}" onclick="${onclick}">`
    + `<span class="proof-choice-letter">${letter}</span>`
    + `<span class="proof-choice-main"><b>${formatProofAction(action)}</b><small>${copy}</small></span>`
    + `<span class="proof-choice-bound">win ≤${bound} turn${bound === 1 ? "" : "s"}`
    + (badge ? `<span class="proof-choice-badge ${badgeClass}">${badge}</span>` : "")
    + `</span></button>`;
}

function proofStepTags(entry, node, remaining) {
  const model = proofExplorerState.model;
  const tags = [
    `<span class="proof-step-tag strong">win within ≤${remaining} attacker turn${remaining === 1 ? "" : "s"}</span>`,
    `<span class="proof-step-tag">node ${entry.nodeId.toLocaleString()}</span>`,
  ];
  if (model.parents[entry.nodeId] > 1) {
    tags.push(`<span class="proof-step-tag">transposition · ${model.parents[entry.nodeId]} incoming paths</span>`);
  }
  if (node.kind === "defender_replies") {
    tags.push(`<span class="proof-step-tag">all ${node.responses.length} exact replies shown</span>`);
  } else if (node.kind === "attacker_move") {
    const count = model.attackerChoices(node).length;
    tags.push(`<span class="proof-step-tag">${count} certified attack${count === 1 ? "" : "s"} retained</span>`);
  }
  return `<div class="proof-step-tags">${tags.join("")}</div>`;
}

function renderProofExplorer() {
  if (!proofExplorerState) return;
  const {model, attacker, defender, history} = proofExplorerState;
  const entry = currentProofEntry();
  const node = model.nodes[entry.nodeId];
  const remaining = entry.shownWin ? 0 : model.remaining(entry.nodeId);
  const summary = document.getElementById("proof-explorer-summary");
  summary.textContent = `${model.nodes.length.toLocaleString()} nodes · `
    + `${model.attackerAlternatives.toLocaleString()} alternatives at `
    + `${model.alternativeAttackerNodes.toLocaleString()} attack nodes · `
    + `≤${model.maxAttackerTurns} attacker turns`;
  document.getElementById("proof-attacker-swatch").style.background = attacker === "P1" ? "#f08a3c" : "#3fb6d9";
  document.getElementById("proof-defender-swatch").style.background = defender === "P1" ? "#f08a3c" : "#3fb6d9";
  document.getElementById("proof-attacker-legend").textContent = `${attacker} attacker`;
  document.getElementById("proof-defender-legend").textContent = `${defender} defender`;
  document.getElementById("proof-back-btn").disabled = history.length <= 1;
  document.getElementById("proof-node-label").textContent = entry.shownWin
    ? "certified completion" : `node ${entry.nodeId.toLocaleString()} / ${model.nodes.length.toLocaleString()}`;
  document.getElementById("proof-progress-label").textContent = entry.shownWin
    ? `${entry.attackerTurnsPlayed} attacker turns · complete`
    : `${entry.attackerTurnsPlayed} attacker turns played · ≤${remaining} remain`;
  const progress = entry.shownWin ? 100
    : Math.min(98, entry.attackerTurnsPlayed / model.maxAttackerTurns * 100);
  document.getElementById("proof-progress-bar").style.width = `${progress}%`;

  const card = document.getElementById("proof-step-card");
  const choices = document.getElementById("proof-choices");
  const worstButton = document.getElementById("proof-worst-btn");
  const alternativesButton = document.getElementById("proof-alternatives-btn");
  let cardHtml = "";
  let choicesHtml = "";
  let accent = "var(--brass)";
  const alternativesHere = !entry.shownWin && node.kind === "attacker_move"
    && model.attackerChoices(node).length > 1;
  const nearestAlternatives = alternativesHere ? [] : model.nearestAlternativePath(entry.nodeId);
  alternativesButton.hidden = model.attackerAlternatives === 0;
  alternativesButton.disabled = alternativesHere || !nearestAlternatives;
  alternativesButton.textContent = alternativesHere ? "Alternatives here"
    : nearestAlternatives ? `Find alternatives (${model.alternativeAttackerNodes.toLocaleString()})`
      : "No later alternatives";

  if (entry.shownWin) {
    accent = "var(--good)";
    cardHtml = `<div class="proof-step-kicker">Certificate complete</div>`
      + `<div class="proof-step-title">Certified win</div>`
      + `<div class="proof-step-copy">The highlighted completion makes six. Use Back to inspect another defense or Reset to return to the root.</div>`;
    worstButton.disabled = true;
    worstButton.textContent = "Proof complete";
  } else if (node.kind === "attacker_move") {
    accent = attacker === "P1" ? "var(--p1)" : "var(--p2)";
    const attacks = model.attackerChoices(node);
    const hasAlternatives = attacks.length > 1;
    cardHtml = `<div class="proof-step-kicker">Attacker · ${proofPlayerClass(attacker)}</div>`
      + `<div class="proof-step-title">${hasAlternatives ? `${attacks.length} certified attacking moves` : "Certified attacking move"}</div>`
      + (hasAlternatives
        ? `<div class="proof-step-copy">The normal PDS-PN run retained these winning attacks. They are ranked by independently verified worst-case depth; other winning moves may still exist.</div>`
        : `<div class="proof-step-copy">This run retained only one certified attack at this position. That does not mean it is the only winning move. Use Find alternatives to jump to a branching attacker position elsewhere in the proof.</div>`)
      + proofStepTags(entry, node, remaining);
    const attackBounds = attacks.map(choice => 1 + model.remaining(choice.child));
    choicesHtml = attacks.map((choice, index) => {
      const bound = attackBounds[index];
      const tiedShortest = index > 0 && bound === attackBounds[0];
      const depthRank = 1 + new Set(
        attackBounds.slice(0, index).filter(candidate => candidate < bound),
      ).size;
      return proofChoiceHtml({
        letter: String(index + 1), action: choice.action,
        color: index === 0 ? (attacker === "P1" ? "#f08a3c" : "#3fb6d9")
          : PROOF_CHOICE_COLORS[index % PROOF_CHOICE_COLORS.length],
        copy: index === 0
          ? "Recommended · shortest verified bound retained by this solve"
          : tiedShortest ? "Certified alternative · tied for the shortest verified bound"
            : `Certified alternative · verified depth rank ${depthRank}`,
        bound,
        badge: index === 0 ? "recommended" : tiedShortest ? "tied shortest" : "",
        badgeClass: "recommended",
        onclick: `proofExplorerPlayAttacker(${index})`,
      });
    }).join("");
    worstButton.disabled = false;
    worstButton.textContent = "Play recommended move →";
  } else if (node.kind === "defender_replies") {
    accent = defender === "P1" ? "var(--p1)" : "var(--p2)";
    const plural = node.responses.length === 1 ? "response" : "responses";
    cardHtml = `<div class="proof-step-kicker">Defender · ${proofPlayerClass(defender)}</div>`
      + `<div class="proof-step-title">Choose a defense</div>`
      + `<div class="proof-step-copy">The verifier recomputed ${node.responses.length} exact non-losing ${plural}. Choose any reply to inspect how the certified attack continues.</div>`
      + proofStepTags(entry, node, remaining);
    const worstIndex = model.worstResponseIndex(node);
    const worstBound = model.remaining(node.responses[worstIndex].child);
    const worstTies = node.responses.filter(response => model.remaining(response.child) === worstBound).length;
    choicesHtml = node.responses.map((response, index) => {
      const bound = model.remaining(response.child);
      const isWorst = bound === worstBound;
      return proofChoiceHtml({
        letter: String.fromCharCode(65 + index), action: response.action,
        color: PROOF_CHOICE_COLORS[index % PROOF_CHOICE_COLORS.length],
        copy: isWorst
          ? (worstTies > 1 ? "Ties for the largest remaining bound" : "This branch maximizes the remaining bound")
          : "Also completely covered by the proof",
        bound, badge: isWorst ? "longest defense" : "", badgeClass: "longest",
        onclick: `proofExplorerChooseDefense(${index})`,
      });
    }).join("");
    worstButton.disabled = false;
    worstButton.textContent = "Choose longest defense →";
  } else if (node.kind === "immediate_win") {
    accent = "var(--good)";
    cardHtml = `<div class="proof-step-kicker">Attacker · ${proofPlayerClass(attacker)}</div>`
      + `<div class="proof-step-title">Complete the six</div>`
      + `<div class="proof-step-copy">The verifier found a legal completion during the current attacker turn. Show it on the board to finish this branch.</div>`
      + proofStepTags(entry, node, remaining);
    choicesHtml = proofChoiceHtml({
      letter: "✓", action: node.action, color: "#79cf9a", copy: "Show the certified winning completion",
      bound: 1, onclick: "proofExplorerPlayAttacker()",
    });
    worstButton.disabled = false;
    worstButton.textContent = "Show winning move →";
  } else if (node.kind === "unstoppable") {
    accent = "var(--good)";
    cardHtml = `<div class="proof-step-kicker">Terminal proof node</div>`
      + `<div class="proof-step-title">Unstoppable fork</div>`
      + `<div class="proof-step-copy">There is no two-placement defense covering every winning completion. Whatever ${defender} plays, ${attacker} completes a six on the next attacker turn.</div>`
      + proofStepTags(entry, node, remaining);
    worstButton.disabled = true;
    worstButton.textContent = "No defense remains";
  }
  card.style.setProperty("--proof-accent", accent);
  card.innerHTML = cardHtml;
  choices.innerHTML = choicesHtml;
  renderProofBreadcrumbs();
  drawProofBoard();
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
    setForcingStatus(`Proof replay failed: ${error.message || error}`, "error");
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
      `A${turn} · #${index + 1}`);
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
    `D${String.fromCharCode(65 + index)}`);
}

function proofExplorerWorstCase() {
  if (!proofExplorerState) return;
  const entry = currentProofEntry();
  if (entry.shownWin) return;
  const node = proofExplorerState.model.nodes[entry.nodeId];
  if (node.kind === "attacker_move" || node.kind === "immediate_win") {
    proofExplorerPlayAttacker();
  } else if (node.kind === "defender_replies") {
    proofExplorerChooseDefense(proofExplorerState.model.worstResponseIndex(node));
  }
}

function proofExplorerFindAlternatives() {
  if (!proofExplorerState) return;
  const state = proofExplorerState;
  const entry = currentProofEntry();
  const path = state.model.nearestAlternativePath(entry.nodeId);
  if (!path) return;
  let stones = entry.stones;
  let attackerTurnsPlayed = entry.attackerTurnsPlayed;
  try {
    for (const step of path) {
      const player = step.role === "attacker" ? state.attacker : state.defender;
      stones = applyProofAction(stones, step.action, player);
      if (step.role === "attacker") attackerTurnsPlayed++;
      state.history.push({
        nodeId: step.child,
        stones,
        attackerTurnsPlayed,
        label: step.role === "attacker"
          ? `A${attackerTurnsPlayed} · #${step.index + 1}`
          : `D${String.fromCharCode(65 + step.index)}`,
        lastAction: {action: step.action, player},
        shownWin: false,
      });
    }
    renderProofExplorer();
    requestAnimationFrame(proofFitBoard);
  } catch (error) {
    closeProofExplorer();
    setForcingStatus(`Proof replay failed: ${error.message || error}`, "error");
  }
}

function proofExplorerBack() {
  if (!proofExplorerState || proofExplorerState.history.length <= 1) return;
  proofExplorerState.history.pop();
  renderProofExplorer();
}

function proofExplorerReset() {
  if (!proofExplorerState) return;
  proofExplorerState.history.splice(1);
  renderProofExplorer();
  requestAnimationFrame(proofFitBoard);
}

function proofExplorerJump(index) {
  if (!proofExplorerState || index < 0 || index >= proofExplorerState.history.length - 1) return;
  proofExplorerState.history.splice(index + 1);
  renderProofExplorer();
}

function renderProofBreadcrumbs() {
  const history = proofExplorerState.history;
  document.getElementById("proof-breadcrumbs").innerHTML = history.map((entry, index) => {
    const current = index === history.length - 1;
    return `<button class="proof-crumb${current ? " current" : ""}"`
      + (current ? " disabled" : ` onclick="proofExplorerJump(${index})"`)
      + `>${entry.label}</button>`;
  }).join("");
}

function proofPendingHighlights(node, entry) {
  if (entry.shownWin) return [];
  if (node.kind === "attacker_move") {
    return proofExplorerState.model.attackerChoices(node).map((choice, index) => ({
      action: choice.action,
      color: index === 0
        ? (proofExplorerState.attacker === "P1" ? "#f08a3c" : "#3fb6d9")
        : PROOF_CHOICE_COLORS[index % PROOF_CHOICE_COLORS.length],
      label: String(index + 1),
    }));
  }
  if (node.kind === "immediate_win") {
    return [{action: node.action, color: proofExplorerState.attacker === "P1" ? "#f08a3c" : "#3fb6d9", label: ""}];
  }
  if (node.kind === "defender_replies") {
    return node.responses.map((response, index) => ({
      action: response.action,
      color: PROOF_CHOICE_COLORS[index % PROOF_CHOICE_COLORS.length],
      label: String.fromCharCode(65 + index),
    }));
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
  const highlights = proofPendingHighlights(node, entry);
  const cellSet = new Set();
  const seeds = [];
  for (const key of entry.stones.keys()) {
    const [q, r] = key.split(",").map(Number);
    seeds.push([q, r]);
  }
  for (const highlight of highlights) seeds.push(...highlight.action);
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

  const focusPoints = [];
  for (const highlight of highlights) {
    if (highlight.action.length === 2) {
      const a = axialToPixel(highlight.action[0][0], highlight.action[0][1]);
      const b = axialToPixel(highlight.action[1][0], highlight.action[1][1]);
      body += `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="${highlight.color}" stroke-width="${S * .09}" stroke-dasharray="${S * .18} ${S * .13}" opacity=".7" pointer-events="none"/>`;
    }
    highlight.action.forEach(([q, r], cellIndex) => {
      const point = axialToPixel(q, r);
      focusPoints.push(point);
      const label = highlight.label ? `${highlight.label}${cellIndex + 1}` : `${cellIndex + 1}`;
      body += `<circle cx="${point.x}" cy="${point.y}" r="${S * .46}" fill="#0d0f0e" fill-opacity=".82" stroke="${highlight.color}" stroke-width="2.8" pointer-events="none"/>`;
      body += `<text x="${point.x}" y="${point.y + 1}" fill="${highlight.color}" font-family="ui-monospace,monospace" font-size="${Math.round(S * .36)}" font-weight="700" text-anchor="middle" dominant-baseline="middle" pointer-events="none">${label}</text>`;
    });
  }
  if (entry.shownWin) {
    const winLength = Number((proofExplorerState.position.config || {}).win_length || 6);
    body += proofWinningLineSvg(entry.stones, proofExplorerState.attacker, winLength);
  }
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

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    buildProofModel, applyProofAction, normalizeProofPosition, proofAttackerChoices,
  };
}
