# HeXO one-sided leaf prover: Lean design kernel

This is a small Lean 4 sidecar for designing a **sound, incomplete, one-sided
MCTS leaf prover**. It is not a rewrite of `hexo-solver`, and it is not intended
to prove that a position has no win.

The contract is deliberately asymmetric:

```text
ProvenWin  => the side to move wins under the real game semantics
Unknown    => anything else (including budget exhaustion)
```

That is exactly the useful leaf-evaluation contract: only `ProvenWin` may
replace the neural value. Search restrictions may lose proofs but must never
invent one.

## Build

The package is pinned to Lean 4.19 and has no dependency beyond Lean's standard
library.

```bash
cd hexo-lean
lake build
```

## Scope

| File | Role |
|---|---|
| `Hexo/Rules.lean` | Infinite axial coordinates, finite sparse positions, radius legality, and live turn state |
| `Hexo/Windows.lean` | Line windows, wins through the placed cell, exact placement transition semantics, and the incremental dirty-window kernel |
| `Hexo/Threats.lean` | Ordered completion gaps, the threat hypergraph, hitting sets, and defender-cover reduction |
| `Hexo/Prover.lean` | Placement-bounded game-theoretic `ForcesWin` and the proof-carrying `ProvenWin | Unknown` API |

The model matches these engine rules:

- the board is infinite but a position contains finitely many stones;
- a placement is legal when empty and within hex distance `radius` of any
  existing stone;
- a normal turn has two **ordered** placements;
- after the first placement, the new stone expands the legal region for the
  second placement;
- a win is checked after each placement, so a first-placement win ends the
  turn immediately;
- a win is checked before the move-limit draw;
- after a nonterminal second placement, the player changes and the next player
  receives two placements.

The initial `(0, 0, P1)` stone is an engine-construction detail, not a theorem
about every solver position, so the formal semantics also accepts arbitrary
finite positions. `ValidLiveState` is the leaf-prover boundary: it requires a
nonempty collision-free position, valid rules, and unused move quota.

## What is proved now

There are no axioms, `sorry`, or admitted theorems.

1. **Placement lookup and radius monotonicity.** A placed stone owns its cell;
   other cell lookups are unchanged. Existing reachable cells stay reachable,
   and every cell inside the new stone's radius becomes reachable. The latter
   is the formal reason a compound move cannot generally be normalized to an
   unordered coordinate pair.

2. **Exact dirty-window set.** A placement belongs to exactly `winLength`
   windows on each of three axes. `Window.affectedBy` therefore has exactly
   `3 * winLength` entries, and a size-`winLength` window is in that list iff it
   contains the placed cell.

3. **Incremental window-kernel correctness.** `patchKernel_correct` proves that
   recomputing those affected windows and retaining every other cached window
   produces exactly the same observations as a full rescan of the new board.
   A Rust implementation may refine each observation into counts, gap masks,
   and bucket links; those derived fields inherit the same dirty set.

4. **The combinatorial defender reduction.** If no set of at most two cells
   covers the completion hypergraph (`B >= 3`), every two-placement defense
   leaves an edge unhit. `defender_cover_reduction` composes this fact with an
   explicit persistence obligation: an unhit completion must remain a legal
   completion after that ordered defense.

5. **One-sided result soundness.** `Verdict (ForcesWin ...)` can contain either
   a proof of the game-semantic win or `Unknown`. There is intentionally no
   `NoWin` constructor. `Verdict.sound` checks the result contract.

## What is specified, but not proved yet

The following are definitions or isolated proof obligations, not completed
end-to-end solver-correctness claims:

- `Step` specifies engine-equivalent placement transitions, including dynamic
  second-placement legality, mid-turn wins, and move-limit draws. A differential
  theorem against Rust `GameState::apply_move` needs a serialized shared corpus.
- `CompletionEdge` specifies a clean winning window with one or two exact gaps
  and at least one legal **ordered** way to fill them.
- `ForcesWin` is the full placement-level minimax meaning: existential attacker
  choices and universal legal defender replies. An executable threat-search
  algorithm has not yet been proved to refine it.
- The geometric persistence premise used by `defender_cover_reduction` is
  exposed rather than assumed globally. Proving it requires an ordered,
  legal `placeMany` relation and a theorem that an unhit gap set is neither
  occupied nor made less reachable by defender stones.
- No completeness claim is intended. Candidate, depth, node, locality, and
  relevance restrictions are allowed to turn a real win into `Unknown`.

## Proposed leaf-prover algorithm

The next production algorithm should be turn-level threat search with a small
proof certificate, not a general solver that spends time proving `No`.

At an attacker node:

1. Try legal ordered one- or two-placement completions, checking for a win after
   the first placement.
2. Generate a heuristic subset of ordered forcing turns from the incremental
   window index. Omitting a candidate is safe because failure returns `Unknown`.
3. Apply a candidate incrementally and build the attacker's next-turn
   completion hypergraph `E`.
4. Reject the branch if the defender has an ordered immediate win (or the move
   quota can draw before the certificate's win).
5. Classify the minimum hitting set of `E`:
   - `B >= 3`: two defender stones cannot hit every edge; certify using an
     unhit-edge/persistence proof.
   - `B = 2`: enumerate every *playable ordered permutation* of every minimum
     two-cell cover and require all child proofs to succeed.
   - `B <= 1`: return `Unknown` in the fully-forcing prover. The defender has a
     spare placement, so cover-only search is not exhaustive.

The certificate only needs attacker turns, the completion edges relied upon,
and a child for each legal minimum cover. A small independent checker should
replay it against the exact placement semantics. MCTS receives `+1` only when
certificate checking succeeds.

### Incremental representation

Use a window identity `(axis, start)` and retain, for each active window:

- P1/P2 occupancy counts;
- an ordered gap mask/list;
- clean/completion/threat bucket membership;
- reverse gap-to-window links for hot-cell generation.

Occupancy changes dirty only `3 * winLength` windows. Radius reach counts can
still be updated over the new stone's hex disk, as the Rust board already does.
Do **not** eagerly rebucket every window merely because a gap became reachable:
keep structural window state independent of reachability, and test the one- or
two-gap `PlayableOrder` using O(1) reach queries when materializing an edge.
That avoids a disk-times-window invalidation wave.

## Rust audit notes that affect soundness/completeness

These are design constraints found while reading `hexo-engine` and the current
IDTT VCF kernel; they are not claims that the existing corpus contains a wrong
verdict.

- `CellSet2` sorts a two-placement move and uses that order for legality. The
  generic game semantics are ordered, but the current tight generator only
  emits pairs after every member passed the pre-turn reachability filter. Both
  orders are therefore legal for every emitted pair, so normalization is sound
  within the current restricted search tree.
- Tight-radius window scanning requires every gap to be reachable on the
  pre-move board. This is sound for a one-sided prover: dropping a chained-reach
  candidate can lose a proof but cannot manufacture one. The stronger
  completeness claim that a newly reachable second placement can never help a
  quiet two-stone builder create next-turn completions is not established here;
  it needs either a locality proof or a finite counterexample search. Do not use
  this filter as part of an exhaustive defender-reduction theorem until that
  distinction is discharged.
- The current forcing state does not carry `move_count`/`max_moves`. A leaf
  certificate must either model draw, as this package does, or prove a horizon
  guard showing that every certified line finishes before the move limit.
- `B >= 3` alone is insufficient: the defender must not be able to win first,
  draw first, or invalidate the purported completion through an unmodeled
  legality effect. These conditions are visible in the formal theorem boundary.

## Next milestones

1. Prove emitted-pair order-insensitivity from the pre-turn reachability gate;
   separately prove or refute completeness of that gate for chained-reach
   two-stone builders.
2. Prove ordered `placeMany` preserves an unhit completion edge, then remove
   the `survives` parameter from the defender-cover theorem.
3. Define certificate syntax and a total checker; prove checker success implies
   `ForcesWin`.
4. Add an executable finite-window enumerator and prove it equals the
   `CompletionEdge` specification.
5. Export shared JSON fixtures from Rust and differential-test legality,
   transitions, windows, gaps, covers, and certificates.
6. Implement the incremental Rust kernel behind a leaf-only flag, verify
   verdict parity/certificate replay, then benchmark and SPRT it.

Only after milestones 1–4 should a pruning rule be promoted from “heuristic
candidate ordering” to “formally exhaustive defender reduction.”
