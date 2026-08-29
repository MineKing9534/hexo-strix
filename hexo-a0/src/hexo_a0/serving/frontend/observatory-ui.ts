import {html, render, type TemplateResult} from "lit";

type Side = "P1" | "P2" | "random";

interface AppActions {
    resign(): void;
    copyHtttx(): void;
    analyzeThisGame(): void;
    openModal(): void;
    goToAnalysis(): void;
    goToPlay(): void;
    toggleTopbarMenu(event: Event): void;
    toggleAnalysisSheet(event: Event): void;
    openHdsImport(): void;
    closeHdsImport(): void;
    openProofLab(): void;
    closeProofLab(): void;
    convertHds(): void;
    analysisInputChanged(): void;
    editAnalysisSource(): void;
    copyAnalysisHtttx(): void;
    cancelAnalysisSourceEdit(): void;
    saveAutomaticAnalysis(): void;
    saveAutomaticForcing(): void;
    saveDisplayPreferences(): void;
    savePositionNumbering(): void;
    updateForcingSolverUi(): void;
    updateForcingEffortUi(): void;
    solveCurrentForcing(): void;
    cancelForcingSolve(): void;
    findBetterDefence(): void;
    cancelBetterDefence(): void;
    tryBetterDefence(): void;
    openProofExplorer(): void;
    shareForcingCertificate(): void;
    downloadForcingCertificate(): void;
    loadGame(): void;
    saveAnalysisStrength(): void;
    selectAnalysisModel(): void;
    selectPlayModel(): void;
    analyzeCurrentPosition(): void;
    analyzeWholeGame(): void;
    rerenderCurrentAnalysis(): void;
    analysisUndo(): void;
    returnToMainline(): void;
    onAnalysisEvalPointerMove(event: PointerEvent): void;
    onAnalysisEvalPointerDown(event: PointerEvent): void;
    onAnalysisEvalPointerUp(event: PointerEvent): void;
    onAnalysisEvalPointerLeave(): void;
    onAnalysisEvalClick(event: PointerEvent): void;
    onAnalysisEvalKeydown(event: KeyboardEvent): void;
    proofExplorerBack(): void;
    proofExplorerReset(): void;
    proofExplorerToggleShortestLine(): void;
    proofExplorerWorstCase(): void;
    proofFitBoard(): void;
    closeProofExplorer(): void;
    proofZoom(factor: number): void;
    proofSetShowLine(visible: boolean): void;
    selectSide(side: Side): void;
    startGame(): void;
}

declare global {
  interface Window extends AppActions {}
}

type ActionName = keyof AppActions;

function invoke<K extends ActionName>(name: K, ...args: Parameters<AppActions[K]>): void {
  const action = window[name];
  if (typeof action !== "function") {
    throw new Error(`UI action ${String(name)} is not available`);
  }
  (action as (...values: Parameters<AppActions[K]>) => void)(...args);
}

function toggleSheetFromKeyboard(event: KeyboardEvent): void {
  if (event.key !== "Enter" && event.key !== " ") return;
  event.preventDefault();
  invoke("toggleAnalysisSheet", event);
}

const topbar = (): TemplateResult => html`
  <div id="topbar">
    <h1><span class="mark">He<span class="mark-x">X</span>O</span><span class="mark-tag">Observatory</span></h1>
    <span id="view-label">Analysis</span>
    <span id="status">Loading…</span>
    <span id="difficulty-badge" hidden></span>
    <button id="resign-btn" hidden @click=${() => invoke("resign")}>Resign</button>
    <button id="copy-htttx-btn" hidden @click=${() => invoke("copyHtttx")}>Copy game record</button>
    <button id="analyze-game-btn" hidden @click=${() => invoke("analyzeThisGame")}>Analyze this game</button>
    <button id="new-game-btn" @click=${() => invoke("openModal")}>New game</button>
    <button id="analysis-btn" @click=${() => invoke("goToAnalysis")}>Analysis</button>
    <button id="play-btn" @click=${() => invoke("goToPlay")}>&larr; Play</button>
    <button id="topbar-menu-btn" type="button" aria-haspopup="true" aria-expanded="false"
      aria-label="Open menu" @click=${(event: Event) => invoke("toggleTopbarMenu", event)}>
      <svg viewBox="0 0 18 18" aria-hidden="true" focusable="false">
        <rect x="2" y="4" width="14" height="1.6" rx="0.8" fill="currentColor"></rect>
        <rect x="2" y="8.2" width="14" height="1.6" rx="0.8" fill="currentColor"></rect>
        <rect x="2" y="12.4" width="14" height="1.6" rx="0.8" fill="currentColor"></rect>
      </svg>
    </button>
    <div id="topbar-menu" role="menu" aria-label="More actions"></div>
    <a id="gh-link" href="https://github.com/SootyOwl/hexo-strix" target="_blank"
      rel="noopener noreferrer" title="View source on GitHub" aria-label="View source on GitHub">
      <svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true" focusable="false"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>
      <span class="gh-label">Source</span>
    </a>
  </div>
`;

const analysisControls = (): TemplateResult => html`
  <div id="analysis-controls">
    <div id="analysis-sheet-handle" data-label="Controls" aria-label="Toggle controls panel"
      role="button" tabindex="0" @click=${(event: Event) => invoke("toggleAnalysisSheet", event)}
      @keydown=${toggleSheetFromKeyboard}></div>
    <div class="analysis-mode-tabs" role="tablist" aria-label="Analysis tools">
      <button id="analysis-mode-analysis" type="button" role="tab" aria-selected="true"
        aria-controls="analysis-controls-body" @click=${() => invoke("closeProofLab")}>Analysis</button>
      <button id="proof-lab-launch" type="button" role="tab" aria-selected="false"
        aria-controls="proof-lab-drawer" disabled @click=${() => invoke("openProofLab")}>
        <span class="proof-lab-launch-icon" aria-hidden="true">◇</span> Proof lab
      </button>
    </div>
    <div id="analysis-info"></div>
    <div id="analysis-position-browser" aria-label="Position navigation">
      <div id="analysis-eval-wrap" hidden>
        <canvas id="analysis-eval-bar" width="320" height="48" tabindex="0" role="slider"
          aria-label="Game position timeline" aria-valuemin="1" aria-valuemax="1" aria-valuenow="1"
          @pointerdown=${(event: PointerEvent) => invoke("onAnalysisEvalPointerDown", event)}
          @pointermove=${(event: PointerEvent) => invoke("onAnalysisEvalPointerMove", event)}
          @pointerup=${(event: PointerEvent) => invoke("onAnalysisEvalPointerUp", event)}
          @pointercancel=${(event: PointerEvent) => invoke("onAnalysisEvalPointerUp", event)}
          @pointerleave=${() => invoke("onAnalysisEvalPointerLeave")}
          @click=${(event: PointerEvent) => invoke("onAnalysisEvalClick", event)}
          @keydown=${(event: KeyboardEvent) => invoke("onAnalysisEvalKeydown", event)}></canvas>
        <div id="analysis-eval-preview" hidden></div>
      </div>
      <div id="analysis-navigation" class="row" hidden>
        <button id="analysis-previous-position" class="analysis-nav-icon" type="button"
          aria-label="Previous position" title="Previous position" @click=${() => invoke("analysisUndo")}>
          <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M9 4 3 10l6 6M4 10h13"/></svg>
        </button>
        <button id="analysis-latest-mainline" class="analysis-nav-icon" type="button"
          aria-label="Latest position in the game" title="Latest position in the game"
          @click=${() => invoke("returnToMainline")}>
          <svg viewBox="0 0 20 20" aria-hidden="true"><path d="m5 4 6 6-6 6M14 4v12"/></svg>
        </button>
      </div>
    </div>
    <div id="analysis-controls-body" role="tabpanel" aria-labelledby="analysis-mode-analysis">
      <div id="analysis-source-summary" hidden>
        <span><strong id="analysis-source-title">Loaded game</strong><small id="analysis-source-meta"></small></span>
        <span class="analysis-source-actions">
          <button id="analysis-copy-htttx" type="button" class="secondary-button" @click=${() => invoke("copyAnalysisHtttx")}>Copy as HTTTX</button>
          <button type="button" class="secondary-button" @click=${() => invoke("editAnalysisSource")}>Change</button>
        </span>
      </div>
      <div id="analysis-setup">
        <button id="hds-import-trigger" type="button" @click=${() => invoke("openHdsImport")}>
          Import from Hexo sandbox <span aria-hidden="true">&rarr;</span>
        </button>
        <label class="analysis-record-field" for="analysis-htttx">
          <span class="field-label">Paste a game record (HTTTX)</span>
          <span id="analysis-record-hint" class="field-hint">Open the game now. Analysis starts only when you ask for it.</span>
          <textarea id="analysis-htttx" rows="4" placeholder="version[1];\n1. [1,0][2,0];\n..."
            aria-describedby="analysis-record-hint"
            @input=${() => invoke("analysisInputChanged")}></textarea>
        </label>
        <div class="analysis-setup-actions">
          <button id="analysis-source-cancel" type="button" class="secondary-button" hidden
            @click=${() => invoke("cancelAnalysisSourceEdit")}>Cancel</button>
          <button id="analysis-load-btn" class="primary-button" @click=${() => invoke("loadGame")}>Load game</button>
        </div>
      </div>
      <div class="analysis-run-group">
        <div class="analysis-run-actions">
          <button id="analysis-position-btn" @click=${() => invoke("analyzeCurrentPosition")} disabled>Analyze position</button>
          <button id="analysis-game-btn" @click=${() => invoke("analyzeWholeGame")} disabled>Analyze full game</button>
        </div>
      </div>
      <details class="analysis-advanced analysis-settings">
        <summary>
          <span>Settings</span>
          <small id="analysis-settings-status">Standard · auto off</small>
        </summary>
        <div class="analysis-settings-body">
          <section class="analysis-settings-section" aria-labelledby="analysis-search-settings-title">
            <h3 id="analysis-search-settings-title">Analysis</h3>
            <label id="analysis-model-field" class="analysis-strength-field" for="analysis-model" hidden>
              <span class="field-label">Strix version</span>
              <span class="field-hint">Choose which trained model evaluates this position.</span>
              <select id="analysis-model" @change=${() => invoke("selectAnalysisModel")}></select>
            </label>
            <label class="analysis-strength-field" for="analysis-strength">
              <span class="field-label">Analysis effort</span>
              <span id="analysis-strength-hint" class="field-hint">Higher settings examine more possible continuations and take longer. Instant gives an estimate without searching ahead.</span>
              <select id="analysis-strength" aria-describedby="analysis-strength-hint" @change=${() => invoke("saveAnalysisStrength")}>
                <option value="network">Instant · no search</option>
                <option value="quick">Quick</option>
                <option value="standard" selected>Standard</option>
                <option value="strong">Strong</option>
                <option value="deep">Deep</option>
              </select>
            </label>
            <label class="analysis-setting-toggle" for="analysis-auto-branch">
              <input id="analysis-auto-branch" type="checkbox"
                @change=${() => invoke("saveAutomaticAnalysis")}>
              <span><strong>Analyze new moves automatically</strong><small>Start analysis after you place a hex</small></span>
            </label>
            <label class="analysis-setting-toggle" for="analysis-auto-forcing">
              <input id="analysis-auto-forcing" type="checkbox" checked
                @change=${() => invoke("saveAutomaticForcing")}>
              <span><strong>Check for forced wins</strong><small>Look for a win the opponent cannot stop</small></span>
            </label>
          </section>
          <section class="analysis-settings-section" aria-labelledby="analysis-display-settings-title">
            <h3 id="analysis-display-settings-title">Board overlays</h3>
            <div class="analysis-display-options-body">
              <label><input type="checkbox" id="analysis-heatmap" checked @change=${() => invoke("saveDisplayPreferences")}> Suggested moves</label>
              <label><input type="checkbox" id="analysis-forcing" checked @change=${() => invoke("saveDisplayPreferences")}> Winning lines</label>
              <label><input type="checkbox" id="analysis-threats" @change=${() => invoke("saveDisplayPreferences")}> Threats to answer</label>
            </div>
          </section>
          <section class="analysis-settings-section" aria-labelledby="analysis-numbering-settings-title">
            <h3 id="analysis-numbering-settings-title">Position numbering</h3>
            <label class="analysis-strength-field" for="analysis-numbering">
              <span class="field-label">Numbering</span>
              <span class="field-hint">Number each placement, or group each full round of both players.</span>
              <select id="analysis-numbering" @change=${() => invoke("savePositionNumbering")}>
                <option value="ply" selected>Ply (1, 2, 3…)</option>
                <option value="round">Round (1, 1, 1, 2…)</option>
              </select>
            </label>
          </section>
        </div>
      </details>
      <div id="analysis-progress">
        <div id="analysis-progress-track"><div id="analysis-progress-bar"></div></div>
        <div id="analysis-progress-label"></div>
      </div>
      <div id="analysis-movetree"></div>
      <details class="analysis-advanced analysis-reading-guide">
        <summary>How to read analysis</summary>
        <div id="analysis-caveat">The score shows who the computer expects to win: positive favours P1 and negative favours P2. Point to or choose the graph to view a position. Darker suggested moves are preferred. Choose any empty hex to try that move. At the end of a turn: ★ best, ✓ good, ? mistake, ✗ blunder.</div>
      </details>
    </div>
    ${proofLabDrawer()}
  </div>
`;

const hdsImportDialog = (): TemplateResult => html`
  <dialog id="hds-import-dialog" aria-labelledby="hds-import-title"
    @click=${(event: Event) => {
      if (event.target === event.currentTarget) invoke("closeHdsImport");
    }}>
    <form @submit=${(event: Event) => { event.preventDefault(); invoke("convertHds"); }}>
      <header class="hds-dialog-header">
        <div>
          <h2 id="hds-import-title">Import from Hexo sandbox</h2>
          <p>Paste the position's hexo.did.science link or short code.</p>
        </div>
        <button class="dialog-close" type="button" aria-label="Close import dialog"
          @click=${() => invoke("closeHdsImport")}>Close</button>
      </header>
      <label for="hds-input"><span class="field-label">Sandbox link or code</span>
        <input id="hds-input" type="text" inputmode="url" autocomplete="off"
          placeholder="https://hexo.did.science/sandbox/5knldz6">
      </label>
      <div id="hds-status" role="status" aria-live="polite"></div>
      <footer class="hds-dialog-actions">
        <button class="secondary-button" type="button" @click=${() => invoke("closeHdsImport")}>Cancel</button>
        <button class="primary-button" type="submit">Import position</button>
      </footer>
    </form>
  </dialog>
`;

const forcingControls = (): TemplateResult => html`
  <div id="analysis-forcing-depth-control" class="proof-lab-form">
    <section class="proof-lab-settings" aria-labelledby="proof-search-settings-title">
      <h3 id="proof-search-settings-title">Search settings</h3>
      <label for="analysis-forcing-engine">Search method
        <span id="analysis-forcing-engine-hint" class="field-hint">The default first proves a win, then rules out every shorter win. It saves all checked replies and the best-defence line.</span>
        <select id="analysis-forcing-engine" aria-describedby="analysis-forcing-engine-hint" @change=${() => invoke("updateForcingSolverUi")}>
          <option value="pdspn-shortest" selected>Prove the shortest win · PDS-PN</option>
          <option value="pdspn">Find and explore a win · PDS-PN</option>
          <option value="idtt">Bounded shortest check · IDTT</option>
        </select>
      </label>
      <label for="analysis-forcing-width">Moves to consider
        <select id="analysis-forcing-width">
          <option value="wide" selected>Broad · all legal moves</option>
          <option value="tight">Direct only · immediate threats</option>
        </select>
      </label>
      <label id="analysis-forcing-depth-row" for="analysis-forcing-depth">
        <span id="analysis-forcing-depth-label">Longest win to check</span>
        <span class="analysis-depth-input"><input id="analysis-forcing-depth" type="number" min="1" max="60" value="25" step="1" inputmode="numeric"> turns</span>
      </label>
      <label id="analysis-forcing-effort-row" for="analysis-forcing-effort">
        <span class="proof-effort-heading"><span>Search effort</span><output id="analysis-forcing-effort-label" for="analysis-forcing-effort">Standard</output></span>
        <input id="analysis-forcing-effort" type="range" min="0" max="5" value="1" step="1"
          aria-describedby="analysis-forcing-effort-hint" @input=${() => invoke("updateForcingEffortUi")}>
        <span id="analysis-forcing-effort-hint" class="field-hint">Good default for most positions.</span>
      </label>
    </section>
    <div class="analysis-solver-actions">
      <button id="analysis-solve-forcing-btn" @click=${() => invoke("solveCurrentForcing")}>Check for a forced win</button>
      <button id="analysis-cancel-forcing-btn" @click=${() => invoke("cancelForcingSolve")} hidden>Stop search</button>
      <button id="analysis-explore-certificate-btn" @click=${() => invoke("openProofExplorer")} hidden>View all replies</button>
      <button id="analysis-share-certificate-btn" @click=${() => invoke("shareForcingCertificate")} hidden>Copy result link</button>
      <button id="analysis-download-certificate-btn" @click=${() => invoke("downloadForcingCertificate")} hidden>Download result</button>
      <span id="proof-share-status" class="proof-share-status" role="status" aria-live="polite"></span>
    </div>
    <div id="analysis-forcing-status" role="status" aria-live="polite">Ready. This search runs on your device.</div>
    <details class="analysis-advanced proof-lab-help">
      <summary>How the search works</summary>
      <div class="proof-lab-help-body">
        <p id="analysis-solver-help">First finds and verifies a forced win. Then reuses that proof to rule out every shorter win. The saved best-defence line shows the replies that delay the win longest.</p>
        <p>Search effort controls how long the solver may keep trying. PDS-PN automatically races several complementary branch strategies. Broad search considers every legal move; direct-only search is faster but considers only immediate threats.</p>
      </div>
    </details>
  </div>
`;

const proofLabDrawer = (): TemplateResult => html`
  <aside id="proof-lab-drawer" hidden role="tabpanel" aria-labelledby="proof-lab-launch">
    <header class="proof-lab-header">
      <div>
        <div class="proof-lab-title-row">
          <h2 id="proof-lab-title">Forced-win proof lab</h2>
          <span class="analysis-local-badge">on this device</span>
        </div>
        <p id="proof-lab-position">Selected analysis position</p>
        <p class="proof-lab-intro">Check whether the player to move can force a win that the opponent cannot stop.</p>
      </div>
    </header>
    <section id="proof-defence-review" class="proof-defence-review" aria-labelledby="proof-defence-review-title">
      <div>
        <h3 id="proof-defence-review-title">How could I have defended?</h3>
        <p id="proof-defence-review-copy">Walk back through this lost replay and find the latest defence that breaks or delays the forced win.</p>
      </div>
      <div class="proof-defence-review-actions">
        <button id="proof-find-defence-btn" class="secondary-button" type="button" @click=${() => invoke("findBetterDefence")}>Find a better defence</button>
        <button id="proof-stop-defence-btn" class="secondary-button" type="button" hidden @click=${() => invoke("cancelBetterDefence")}>Stop</button>
      </div>
      <div id="proof-defence-status" class="proof-defence-status" role="status" aria-live="polite"></div>
      <div id="proof-defence-result" class="proof-defence-result" hidden></div>
    </section>
    ${forcingControls()}
  </aside>
`;

const analysisPanel = (): TemplateResult => html`
  <div id="analysis-panel" hidden>
    ${analysisControls()}
    <div id="analysis-board-container">
      <div id="analysis-empty-state">
        <strong>Load a game to explore it</strong>
        <span>Loading is instant and does not start the analysis engine.</span>
      </div>
      <div id="gauge-wrap" hidden>
        <div class="gauge-poles">
          <span class="pole pole-p1">● P1 <b id="gauge-v1">+0.00</b></span>
          <span class="pole pole-p2"><b id="gauge-v2">−0.00</b> P2 ●</span>
        </div>
        <div class="gauge"><div class="gauge-zero"></div><div class="gauge-needle" id="gauge-needle"></div></div>
        <div class="gauge-scale"><span>P1 +1.0</span><span>EVEN</span><span>P2 +1.0</span></div>
      </div>
      <svg id="analysis-board"></svg>
      <div id="analysis-thinking" role="status" aria-live="polite" aria-atomic="true" hidden>
        <span class="analysis-thinking-mark" aria-hidden="true"><i></i><i></i><i></i></span>
        <span id="analysis-thinking-label">Checking position…</span>
      </div>
      <div id="board-legend" hidden>
        <span><i class="sw sw-p1"></i>P1 to move</span>
        <span><i class="sw sw-p2"></i>P2 to move</span>
        <span><i class="sw sw-pick"></i>top suggestion</span>
      </div>
    </div>
  </div>
`;

const proofExplorer = (): TemplateResult => html`
  <div id="proof-explorer" role="dialog" aria-modal="true" aria-labelledby="proof-explorer-title" hidden>
    <section id="proof-board-container" aria-label="Proof position board">
      <svg id="proof-board" aria-label="HeXO proof board"></svg>
      <div class="proof-explorer-actions" aria-label="Proof explorer actions">
        <button id="proof-share-btn" @click=${() => invoke("shareForcingCertificate")}
          title="Save this result and copy its link">Copy link</button>
        <button @click=${() => invoke("downloadForcingCertificate")}>Download</button>
        <button id="proof-close-btn" class="proof-close" @click=${() => invoke("closeProofExplorer")}
          aria-label="Close proof explorer">Close</button>
      </div>
      <div class="proof-board-tools" aria-label="Proof board tools">
        <label class="proof-board-toggle" for="proof-show-line">
          <input id="proof-show-line" type="checkbox"
            @change=${(event: Event) => invoke("proofSetShowLine", (event.target as HTMLInputElement).checked)}>
          <span>Show winning line</span>
        </label>
        <div class="proof-board-zoom" aria-label="Board zoom controls">
          <button @click=${() => invoke("proofZoom", 1.25)} aria-label="Zoom in">+</button>
          <button @click=${() => invoke("proofZoom", 0.8)} aria-label="Zoom out">−</button>
          <button @click=${() => invoke("proofFitBoard")}>Fit</button>
        </div>
      </div>
      <div class="proof-board-legend">
        <span><i id="proof-attacker-swatch" class="proof-sw"></i><span id="proof-attacker-legend">winning side</span></span>
        <span><i id="proof-defender-swatch" class="proof-sw"></i><span id="proof-defender-legend">defending side</span></span>
        <span><i class="proof-sw proof-sw-choice"></i>previewed move</span>
      </div>
      <aside class="proof-explorer-panel" aria-label="Proof navigation">
        <header class="proof-explorer-heading">
          <span class="proof-explorer-kicker">Checked winning strategy</span>
          <h2 id="proof-explorer-title">Explore the win</h2>
          <span id="proof-explorer-summary"></span>
        </header>
        <nav class="proof-history-actions" aria-label="Proof history">
          <button id="proof-back-btn" @click=${() => invoke("proofExplorerBack")} title="Go back one step">&larr; Back</button>
          <button @click=${() => invoke("proofExplorerReset")} title="Return to the first position">Start again</button>
        </nav>
        <div class="proof-progress-copy"><span id="proof-progress-label"></span><span id="proof-node-label"></span></div>
        <div class="proof-progress-track"><div id="proof-progress-bar"></div></div>
        <div id="proof-optimization-note" class="proof-optimization-note" hidden></div>
        <div id="proof-step-card"></div>
        <div class="proof-path-heading"><span>Proof path</span><small><span class="proof-hover-hint">hover to preview · </span>choose to follow</small></div>
        <div id="proof-tree" class="proof-tree" role="tree" aria-label="Positions and available branches"></div>
        <div class="proof-panel-actions">
          <button id="proof-shortest-line-btn" @click=${() => invoke("proofExplorerToggleShortestLine")} hidden>Longest defence</button>
          <button id="proof-worst-btn" class="proof-primary" @click=${() => invoke("proofExplorerWorstCase")}
            title="Follow the reply that delays the win longest">Choose longest defence &rarr;</button>
        </div>
        <details class="proof-explorer-note">
          <summary>How to read this proof</summary>
          <p>On the winning side's turn, each branch shown is a move that this search proved will win. On the other side's turn, every checked reply is shown. “Longest defence” follows the reply that delays the win for the most turns.</p>
        </details>
      </aside>
    </section>
  </div>
`;

const newGameDialog = (): TemplateResult => html`
  <div id="modal-bg">
    <div id="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title">
      <h2 id="modal-title">New game</h2>
      <label id="play-model-field" for="play-model" hidden>
        <span class="field-label">Opponent</span>
        <span class="field-optional">Choose a Strix version</span>
        <select id="play-model" @change=${() => invoke("selectPlayModel")}></select>
      </label>
      <div id="bot-stats">
        <div id="bot-stats-current">Loading the bot's record…</div>
        <div id="bot-stats-alltime"></div>
      </div>
      <label for="modal-name"><span class="field-label">Name</span><span class="field-optional">Optional</span><input id="modal-name" type="text" maxlength="64" autocomplete="off"></label>
      <label for="modal-elo"><span class="field-label">Your rating (Elo)</span><span class="field-optional">Optional · enter your own estimate</span><input id="modal-elo" type="number" min="0" max="3500" placeholder="1500" autocomplete="off" inputmode="numeric"></label>
      <fieldset class="side-fieldset">
        <legend>Side</legend>
        <div class="side-row">
          <button class="side-btn" data-side="P1" @click=${() => invoke("selectSide", "P1")}><span class="stone stone-p1">●</span>P1 <span class="side-colour">orange</span></button>
          <button class="side-btn selected" data-side="random" @click=${() => invoke("selectSide", "random")}><span class="stone">?</span>Random</button>
          <button class="side-btn" data-side="P2" @click=${() => invoke("selectSide", "P2")}><span class="stone stone-p2">●</span>P2 <span class="side-colour">blue</span></button>
        </div>
      </fieldset>
      <label id="diff-label" hidden>Search effort</label>
      <div id="diff-row" class="diff-row" hidden></div>
      <button id="start-btn" @click=${() => invoke("startGame")}>Start game</button>
    </div>
  </div>
`;

const app = (): TemplateResult => html`
  ${topbar()}
  <div id="board-container"><svg id="board"></svg></div>
  ${analysisPanel()}
  ${proofExplorer()}
  ${hdsImportDialog()}
  ${newGameDialog()}
`;

class HexoObservatoryApp extends HTMLElement {
  connectedCallback(): void {
    render(app(), this);
  }
}

customElements.define("hexo-observatory-app", HexoObservatoryApp);
