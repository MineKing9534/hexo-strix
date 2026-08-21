---
name: HeXO Observatory
description: Play and analyse infinite hex tic-tac-toe against the Strix engine.
colors:
  aged-brass: "#c9a35e"
  phosphor: "#f2b65a"
  observatory-ink: "#0d0f0e"
  panel: "#15191a"
  panel-2: "#1b201f"
  line: "#2a302e"
  line-bright: "#3a423f"
  warm-parchment: "#ece6da"
  muted: "#8b8f88"
  faint: "#5f635d"
  p1-orange: "#f08a3c"
  p2-blue: "#3fb6d9"
  best-gold: "#f2c14e"
  good-green: "#79cf9a"
  warn-amber: "#e0a23a"
  bad-red: "#e25c5c"
  control: "#111514"
  control-hover: "#181d1b"
  secondary-text: "#aeb2aa"
  placeholder: "#9ca098"
typography:
  display:
    fontFamily: "Instrument Serif, serif"
    fontWeight: 400
  body:
    fontFamily: "Hanken Grotesk, system-ui, sans-serif"
    fontWeight: 400
  label:
    fontFamily: "Martian Mono, ui-monospace, monospace"
    fontWeight: 400
rounded:
  sm: "4px"
  md: "6px"
  lg: "12px"
components:
  button-primary:
    backgroundColor: "{colors.aged-brass}"
    textColor: "{colors.observatory-ink}"
    rounded: "{rounded.md}"
  button-primary-hover:
    backgroundColor: "{colors.phosphor}"
    textColor: "{colors.observatory-ink}"
  button-secondary:
    backgroundColor: "#232826"
    textColor: "#d8d2c6"
    rounded: "{rounded.md}"
  input:
    backgroundColor: "{colors.control}"
    textColor: "{colors.warm-parchment}"
    rounded: "{rounded.md}"
---

# Design System: HeXO Observatory

## Overview

**Creative North Star: "The Brass Astrolabe"**

HeXO Observatory is a dark instrument panel for reading a game. Its personality is precise, nocturnal, warm-brass, instrument-grade, and quietly confident. The interface recedes behind the board — the artifact being studied — and every control, label, and readout behaves like a calibrated part of a measuring instrument rather than a decorative web widget.

The system is built on three voices: a serif display face for the wordmark and headings, a humanist sans for body and controls, and a monospace for labels, coordinates, and data. Brass is the single accent, used sparingly against near-black ink; phosphor is its brighter, hotter sibling reserved for hover, focus, and the "live" signal. A faint honeycomb watermark and recurring hexagonal geometry tie the whole surface back to the game itself.

**Key Characteristics:**
- Dark, near-black ink ground with warm off-white text.
- One brass accent, used sparingly; phosphor for the live/hover/focus signal.
- Serif display + humanist sans + monospace label trio.
- Flat surfaces with tonal layering; shadows only for lifted overlays.
- Hexagonal motif (honeycomb watermark, hex stones, hex clip-path markers).
- Precise, deliberate controls with visible press and focus states.

## Colors

The palette is a dark instrument panel: warm brass and phosphor accents against layered near-black neutrals, with a small set of semantic colors for game state and verdicts.

### Primary
- **Aged Brass** (`#c9a35e`): The single accent. Wordmark tag, primary buttons, selected states, focus rings, and the honeycomb watermark. Its rarity is the point.
- **Phosphor** (`#f2b65a`): The hotter sibling of brass. Hover on primary buttons, the focus ring, the "thinking" dots, the gauge needle, and the last-move halo. Signals "live" or "active".

### Neutral
- **Observatory Ink** (`#0d0f0e`): The page background and the darkest surface.
- **Panel** (`#15191a`): Primary surface — topbar, modals, control panels.
- **Panel 2** (`#1b201f`): Secondary surface — stat cards, settings groups, move tree.
- **Line** (`#2a302e`): Default border and divider.
- **Line Bright** (`#3a423f`): Brighter border for interactive or floating surfaces.
- **Warm Parchment** (`#ece6da`): Primary text.
- **Muted** (`#8b8f88`): Secondary text, hints, captions.
- **Faint** (`#5f635d`): Tertiary text, disabled and de-emphasized labels.

### Semantic
- **P1 Orange** (`#f08a3c`): Player 1 stones and P1 readouts.
- **P2 Blue** (`#3fb6d9`): Player 2 stones and P2 readouts.
- **Best Gold** (`#f2c14e`): The best-move marker.
- **Good Green** (`#79cf9a`): Winning verdicts and confirmed results.
- **Warn Amber** (`#e0a23a`): Threats and warnings.
- **Bad Red** (`#e25c5c`): Errors, blunders, and destructive actions.

### Control
- **Control** (`#111514`): Recessed input and toggle background.
- **Control Hover** (`#181d1b`): Hover state for recessed controls.
- **Secondary Text** (`#aeb2aa`): Field labels and secondary copy.
- **Placeholder** (`#9ca098`): Input placeholder text.

### Named Rules
**The One Voice Rule.** Brass is the only accent and is used on a small fraction of any screen. If a second accent starts competing with it, the instrument-panel calm is lost.

## Typography

**Display Font:** Instrument Serif (with serif fallback)
**Body Font:** Hanken Grotesk (with system-ui, sans-serif fallback)
**Label/Mono Font:** Martian Mono (with ui-monospace, monospace fallback)

**Character:** A serif display face gives the wordmark and headings a quiet, editorial authority; the humanist sans keeps body and controls legible and warm; the monospace carries coordinates, data, and labels with instrument precision.

### Hierarchy
- **Display** (400, 30px wordmark / 20–24px headings, 1.0–1.2 line-height): The "HeXO" wordmark and section headings. Serif, with the "X" in P1 orange italic.
- **Headline** (400, 20–24px, 1.2): Screen labels and empty-state titles.
- **Title** (600, 15–18px, 1.3): Panel and dialog titles.
- **Body** (400, 12–14px, 1.45): Body copy, controls, and descriptions.
- **Label** (400, 8.5–11px, 1.5–3px letter-spacing, uppercase): Mono labels, tags, coordinates, and data readouts.

### Named Rules
**The Three Voices Rule.** Serif speaks (wordmark, headings), sans explains (body, controls), mono measures (labels, coordinates, data). Never swap a role.

## Layout

The app is a full-viewport shell: a slim topbar over a board that fills the remaining space, with the analysis controls as a floating side panel (desktop) or a bottom sheet (mobile). Density is comfortable; the board is the hero and controls stay out of its way.

- **Topbar:** slim (8px vertical padding), holds the wordmark, screen label, status, and contextual actions.
- **Board:** fills the viewport, pan/zoom, with a honeycomb watermark behind it.
- **Analysis controls:** a 360px floating panel on the left (desktop); collapses to a bottom sheet on mobile.
- **Breakpoints:** ≤480px (phone: overflow menu, full-width modal, bottom sheet), 481–768px (tablet: tighter topbar, sheet starts collapsing), >768px (desktop: side-by-side).
- **Spacing rhythm:** small, consistent gaps (4–16px); no large decorative whitespace.

## Elevation & Depth

The system is **flat at rest** and conveys depth through **tonal layering** — ink → panel → panel 2 → control — rather than shadows. Shadows appear only on lifted overlays (modals, floating panels, the proof explorer) to separate them from the board.

### Shadow Vocabulary
- **Overlay** (`box-shadow: 0 8px 32px rgba(0,0,0,0.5)`): Modals and dialogs.
- **Floating panel** (`box-shadow: 0 16px 42px rgba(0,0,0,0.34)`): The proof explorer panel.
- **Tooltip** (`box-shadow: 0 4px 8px rgba(0,0,0,0.35)`): The "thinking" status chip.

### Named Rules
**The Flat-By-Default Rule.** Surfaces are flat at rest. Shadows appear only as a response to elevation (modal, floating panel), never as decoration on resting cards.

## Shapes

The form language is gently rounded and hexagonal. Corners are small (4px) to medium (6px), with a larger 12px radius reserved for modals and dialogs. The hexagon recurs as the system's signature silhouette: hex stones, hex clip-path markers in the proof tree, and the honeycomb watermark.

- **Radius:** 4px (buttons, chips, small controls), 6px (inputs, cards, panels), 12px (modals, dialogs).
- **Borders:** 1px hairlines in `line` / `line-bright`; brass borders mark selection and focus.
- **Signature geometry:** the hexagon (stones, markers, watermark).

## Components

### Buttons
- **Shape:** 4–6px radius, no border on primary.
- **Primary:** Aged Brass background, Observatory Ink text, bold, 6px radius. Hover → Phosphor. Active → 1px press-down.
- **Secondary:** dark `#232826` background, light text. Hover → brass border.
- **Ghost:** transparent with a `line` border; hover → brass border and text.
- **Focus:** 2px phosphor outline, 2px offset.

### Inputs / Fields
- **Style:** recessed `control` background, 1px `control-border`, 6px radius, 42px min-height.
- **Focus:** brass border shift.
- **Placeholder:** `placeholder` color.
- **Select:** custom brass chevron, appearance-none.

### Cards / Containers
- **Corner Style:** 6px radius.
- **Background:** `panel` or `panel-2`.
- **Border:** 1px `line` (or `line-bright` when interactive).
- **Shadow:** none at rest (see Elevation).

### Chips / Badges
- **Style:** small (2–8px padding), 10px radius, mono uppercase labels.
- **Difficulty badge:** brass text on a dark brass-tinted background.
- **Local badge:** green text on a dark green-tinted background, pill radius.

### Navigation
- **Topbar:** slim, brass wordmark, serif screen label, contextual buttons.
- **Tabs:** segmented control (ink background, `line` border); selected tab gets `control-hover` background and light text.
- **Mobile:** overflow menu (burger) at ≤480px; analysis controls become a bottom sheet.

### Signature: The Gauge
A vertical advantage rail on the board's right edge: a 12px-wide gradient track (P1 orange → ink → P2 blue) with a phosphor needle and a center zero line. It reads the game's evaluation at a glance, like a physical instrument.

### Signature: The Stone
Flat hexagons (no bevel) in P1 orange or P2 blue; empty hexes are near-invisible with a faint brass stroke. The most recent move gets a phosphor ring with a slow pulsing halo.

## Do's and Don'ts

### Do:
- **Do** keep brass rare — one accent, used on a small fraction of any screen.
- **Do** use the three-voice type system: serif for headings, sans for body, mono for labels and data.
- **Do** keep surfaces flat at rest; reserve shadows for lifted overlays.
- **Do** use the hexagon as the recurring signature (stones, markers, watermark).
- **Do** give every interactive control a visible focus ring (2px phosphor) and a press-down state.

### Don't:
- **Don't** introduce a second accent color that competes with brass.
- **Don't** add bevels, gradients, or 3D treatment to stones — they are flat.
- **Don't** use shadows as decoration on resting cards.
- **Don't** swap the type roles (serif for data, mono for headings).
- **Don't** use large decorative whitespace; the board is the hero.
