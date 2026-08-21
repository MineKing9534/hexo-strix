class HexoMoveList extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({mode: "open"});
    this._rounds = [];
    this._leadingVariations = [];
    this._activeId = null;
  }

  set rounds(value) {
    this._rounds = Array.isArray(value) ? value : [];
    this.render();
  }

  get rounds() { return this._rounds; }

  set leadingVariations(value) {
    this._leadingVariations = Array.isArray(value) ? value : [];
    this.render();
  }

  get leadingVariations() { return this._leadingVariations; }

  set activeId(value) {
    this._activeId = value == null ? null : String(value);
    this.render();
  }

  get activeId() { return this._activeId; }

  connectedCallback() { this.render(); }

  _coordinate(move) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "coordinate";
    button.dataset.moveId = String(move.id);
    button.setAttribute("aria-label", `Move ${move.q}, ${move.r}`);
    if (String(move.id) === this._activeId) {
      button.classList.add("active");
      button.setAttribute("aria-current", "step");
    }
    const parts = ["[", move.q, ",", move.r, "]"];
    for (const [index, value] of parts.entries()) {
      const span = document.createElement("span");
      span.textContent = String(value);
      span.setAttribute("aria-hidden", "true");
      if (index === 1 || index === 3) span.className = "number";
      button.appendChild(span);
    }
    button.addEventListener("click", () => this.dispatchEvent(new CustomEvent("move-select", {
      detail: {id: move.id}, bubbles: true, composed: true,
    })));
    return button;
  }

  _missed(move) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "missed";
    button.dataset.moveId = String(move.id);
    button.title = "This move gave up a forced win. Show the winning line.";
    button.setAttribute("aria-label", `Win missed at ${move.q}, ${move.r}. Show the winning line.`);
    button.innerHTML = `<svg viewBox="0 0 12 14" aria-hidden="true"><path d="M7.2 0 1.5 8h3.7L4.3 14l6.2-8H6.7L7.2 0Z"/></svg>`;
    button.addEventListener("click", () => this.dispatchEvent(new CustomEvent("missed-win-select", {
      detail: {id: move.id}, bubbles: true, composed: true,
    })));
    return button;
  }

  _placement(move) {
    const placement = document.createElement("div");
    placement.className = "placement";
    if (!move) {
      placement.classList.add("empty");
      placement.setAttribute("aria-hidden", "true");
      return placement;
    }
    placement.appendChild(this._coordinate(move));
    return placement;
  }

  _turn(moves, player) {
    const turn = document.createElement("div");
    turn.className = "turn";
    turn.setAttribute("aria-label", `${player} turn`);
    const list = Array.isArray(moves) ? moves.slice(0, 2) : [];
    while (list.length < 2) list.push(null);
    for (const move of list) turn.appendChild(this._placement(move));

    const status = document.createElement("span");
    status.className = "turn-status";
    const missedMove = list.find(move => move?.missedWin);
    if (missedMove) status.appendChild(this._missed(missedMove));
    const qualityData = [...list].reverse().find(move => move?.quality)?.quality;
    if (qualityData) {
      const quality = document.createElement("span");
      quality.className = "quality";
      quality.textContent = qualityData.icon;
      quality.title = qualityData.label;
      quality.setAttribute("aria-label", qualityData.label);
      if (qualityData.color) quality.style.color = qualityData.color;
      status.appendChild(quality);
    }
    turn.appendChild(status);
    return turn;
  }

  _variation(data) {
    const card = document.createElement("div");
    card.className = "variation";
    card.style.setProperty("--nesting", String(Math.min(Number(data.nesting) || 0, 3)));
    const heading = document.createElement("div");
    heading.className = "variation-heading";
    heading.textContent = data.label || "Alternative line";
    card.appendChild(heading);
    for (const turnData of data.turns || []) {
      const line = document.createElement("div");
      line.className = "variation-line";
      const side = document.createElement("span");
      side.className = "variation-side";
      side.textContent = turnData.player;
      line.appendChild(side);
      line.appendChild(this._turn(turnData.moves, turnData.player));
      card.appendChild(line);
    }
    return card;
  }

  render() {
    if (!this.shadowRoot) return;
    this.shadowRoot.innerHTML = `<style>
      :host {
        --ml-ink: var(--ink, #0d0f0e);
        --ml-panel: var(--panel-2, #1b201f);
        --ml-control: var(--control, #111514);
        --ml-hover: var(--control-hover, #20251f);
        --ml-line: var(--line, #2a302e);
        --ml-text: var(--text, #ece6da);
        --ml-muted: var(--muted, #8b8f88);
        --ml-faint: var(--faint, #5f635d);
        --ml-brass: var(--brass, #c9a35e);
        --ml-phos: var(--phos, #f2b65a);
        --ml-mono: var(--mono, "Martian Mono", ui-monospace, monospace);
        display: block;
        min-width: 0;
        container-type: inline-size;
        color: var(--ml-text);
        font-family: var(--ml-mono);
      }
      * { box-sizing: border-box; }
      button { font: inherit; }
      .sheet {
        display: grid;
        grid-template-columns: 30px minmax(0, 1fr) minmax(0, 1fr);
        min-width: 0;
        background: var(--ml-panel);
        border-radius: 4px;
        overflow: clip;
      }
      .head {
        position: sticky;
        top: 0;
        z-index: 2;
        min-width: 0;
        padding: 4px 7px;
        border-bottom: 1px solid var(--ml-line);
        background: var(--ml-panel);
        color: var(--ml-muted);
        font: 600 10px/1.2 var(--ml-mono);
      }
      .head:not(:first-child) { text-align: center; }
      .round-number {
        display: grid;
        place-items: center end;
        padding: 4px 6px 4px 2px;
        color: var(--ml-faint);
        font: 500 11px/1 var(--ml-mono);
        font-variant-numeric: tabular-nums;
      }
      .round-number.alt, .turn.alt { background: color-mix(in srgb, var(--ml-brass) 4%, transparent); }
      .turn {
        display: grid;
        grid-template-columns: max-content 30px;
        grid-template-rows: repeat(2, 21px);
        justify-content: center;
        column-gap: 2px;
        min-width: 0;
        padding: 2px 4px;
        border-inline-start: 1px solid color-mix(in srgb, var(--ml-line) 70%, transparent);
      }
      .placement {
        grid-column: 1;
        display: grid;
        align-items: center;
        min-width: 0;
      }
      .placement.empty::before {
        content: "·";
        padding-inline-start: 4px;
        color: color-mix(in srgb, var(--ml-faint) 55%, transparent);
      }
      .coordinate {
        display: inline-grid;
        grid-template-columns: auto 3ch auto 3ch auto;
        align-items: baseline;
        justify-content: start;
        width: max-content;
        min-width: 0;
        margin: 0;
        padding: 1px 3px;
        border: 0;
        border-radius: 4px;
        background: transparent;
        color: #b8bdb2;
        font: 500 12px/1.35 var(--ml-mono);
        font-variant-numeric: tabular-nums;
        cursor: pointer;
      }
      .coordinate .number { text-align: right; }
      .coordinate:hover { background: var(--ml-hover); color: var(--ml-text); }
      .coordinate.active { background: var(--ml-brass); color: var(--ml-ink); font-weight: 650; }
      .coordinate:focus-visible, .missed:focus-visible {
        outline: 2px solid var(--ml-phos);
        outline-offset: 1px;
      }
      .turn-status {
        grid-column: 2;
        grid-row: 1 / -1;
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: center;
        gap: 3px;
        min-width: 0;
      }
      .missed {
        display: inline-grid;
        place-items: center;
        width: 17px;
        height: 17px;
        margin: 0;
        padding: 0;
        border: 1px solid color-mix(in srgb, var(--ml-phos) 52%, var(--ml-line));
        border-radius: 4px;
        background: color-mix(in srgb, var(--ml-phos) 13%, var(--ml-control));
        color: var(--ml-phos);
        cursor: pointer;
      }
      .missed svg { position: static; width: 9px; height: 11px; fill: currentColor; }
      .missed:hover { background: color-mix(in srgb, var(--ml-phos) 24%, var(--ml-control)); }
      .quality {
        width: 12px;
        text-align: center;
        font: 650 11px/1 var(--ml-mono);
      }
      .variation {
        grid-column: 1 / -1;
        margin: 4px 6px 5px calc(6px + var(--nesting) * 8px);
        padding: 5px 6px;
        border: 1px solid var(--ml-line);
        border-radius: 4px;
        background: var(--ml-control);
      }
      .variation-heading { margin-bottom: 3px; color: var(--ml-faint); font: 500 9px/1.4 var(--ml-mono); }
      .variation-line { display: grid; grid-template-columns: 24px minmax(0, 1fr); align-items: center; }
      .variation-line .turn { justify-content: start; border-inline-start: 0; padding-block: 1px; }
      .variation-side { color: var(--ml-brass); font: 600 9px/1 var(--ml-mono); }
      @container (max-width: 340px) {
        .sheet { grid-template-columns:27px minmax(0,1fr) minmax(0,1fr); }
        .turn { grid-template-columns:max-content 30px; padding-inline:3px; }
        .coordinate { grid-template-columns: auto 2.5ch auto 2.5ch auto; font-size: 11px; padding-inline: 2px; }
      }
    </style><div class="sheet" role="table" aria-label="Game moves">
      <div class="head" role="columnheader">#</div>
      <div class="head" role="columnheader">P1</div>
      <div class="head" role="columnheader">P2</div>
    </div>`;
    const sheet = this.shadowRoot.querySelector(".sheet");
    for (const variation of this._leadingVariations) sheet.appendChild(this._variation(variation));
    for (const [index, round] of this._rounds.entries()) {
      const alt = index % 2 === 1;
      const number = document.createElement("div");
      number.className = `round-number${alt ? " alt" : ""}`;
      number.setAttribute("role", "rowheader");
      number.textContent = `${round.number}.`;
      sheet.appendChild(number);
      for (const player of ["P1", "P2"]) {
        const turn = this._turn(round[player], player);
        if (alt) turn.classList.add("alt");
        turn.setAttribute("role", "cell");
        sheet.appendChild(turn);
      }
      for (const variation of round.variations || []) sheet.appendChild(this._variation(variation));
    }
    queueMicrotask(() => this.shadowRoot?.querySelector(".coordinate.active")?.scrollIntoView({block: "nearest"}));
  }
}

if (!customElements.get("hexo-move-list")) customElements.define("hexo-move-list", HexoMoveList);
