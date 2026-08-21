# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **Primary:** the HeXO community — people who play infinite hex tic-tac-toe and want a strong opponent and a way to understand their games. They arrive through the public web app (served over Tailscale Funnel, no accounts).
- **Secondary:** the author, a solo researcher who uses the same engine and tooling for training, evaluation, and analysis.

## Product Purpose

HeXO Observatory lets visitors play HeXO against the Strix AlphaZero engine and analyze games. It exists to give the HeXO community a genuinely useful tool: a strong opponent, the best available analysis, and the only public proof/solver tool. Success means the community actually uses it to play and to understand positions.

## Positioning

- An extremely strong HeXO play bot — perhaps the strongest, though that title is currently contested.
- The best analysis tool that currently exists for HeXO.
- The only publicly available proof/solver tool for HeXO (forced-win search).
- Runs on-device in the browser via WASM, so analysis does not queue behind shared server compute.

## Operating Context

- Played in a web browser on desktop and mobile.
- Served over Tailscale Funnel; no accounts, no matchmaking.
- Completed games are recorded to SQLite; a token-gated `/admin` page exposes them.
- Interoperates with the wider HeXO ecosystem: HTTTX game records, import from the hexo.did.science sandbox (a third-party site), and the SealBot evaluation opponent.

## Capabilities and Constraints

- Play against Strix at selectable difficulty tiers.
- Analysis: eval bar, move heatmap, threat overlay, move tree, and per-position or whole-game analysis.
- Proof lab: forced-win search (PDS-PN, IDTT, DFPN, PNS solvers) with shareable and downloadable certificates.
- Browser-side WASM inference runs bot moves and analysis on-device; server-side inference remains as a compatibility fallback.
- Solo research project on a single-GPU ROCm APU; research-grade rough edges are expected.
- Terminology: HeXO (the game), Strix (the bot), HTTTX (the game-record format).

## Brand Commitments

- **HeXO** is the name of the game, not the author's creation.
- **Strix** is the bot's name.
- Current product name: **HeXO Observatory** (the author likes it). Open decision: **"Strix's Roost" / "Strix's Nest"** is an appealing alternative that keeps the owl theme — undecided, not yet chosen.
- Owl theme (Strix is an owl genus).

## Evidence on Hand

- Trained checkpoints (champion models) and recorded games in SQLite.
- A research corpus and analysis notes under `docs/research/`.
- No fabricated testimonials, benchmarks, or user counts; future work must not invent these.

## Product Principles

- Be a genuinely useful tool to the HeXO community, not a showcase.
- Prefer plain language over repo-specific jargon, following ISO 24495-1:2023 in spirit.
- Work well on desktop and mobile.
- Be honest about the engine's limits (for example, analysis accuracy far from the board center).

## Accessibility & Inclusion

- Must work on desktop and mobile.
- Plain language (ISO 24495-1:2023) in spirit, avoiding repo-specific jargon.
