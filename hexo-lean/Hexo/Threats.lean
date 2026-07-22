import Hexo.Windows

namespace Hexo

/-- Sequential playability. This is deliberately a list, not an unordered pair. -/
def PlayableOrder : Position → Player → Nat → List Coord → Prop
  | _, _, _, [] => True
  | p, who, radius, cell :: rest =>
      Legal p radius cell ∧ PlayableOrder (p.place cell who) who radius rest

def emptyCells (p : Position) (w : Window) : List Coord :=
  w.cells.filter fun cell => p.ownerAt cell = none

/-- A clean line that the attacker can finish on its next (at most two-placement)
turn, including radius legality in some explicit placement order. -/
def CompletionEdge (rules : Rules) (p : Position) (attacker : Player)
    (edge : List Coord) : Prop :=
  ∃ w,
    w.size = rules.winLength ∧
    CleanWindow p attacker w ∧
    edge = emptyCells p w ∧
    edge ≠ [] ∧
    edge.length ≤ 2 ∧
    ∃ order, order.Perm edge ∧
      PlayableOrder p attacker rules.placementRadius order

def Hits (placements edge : List Coord) : Prop :=
  ∃ cell, cell ∈ placements ∧ cell ∈ edge

def Covers (edges : List (List Coord)) (placements : List Coord) : Prop :=
  ∀ edge ∈ edges, Hits placements edge

/-- `B ≥ 3`: no set of at most two defender cells hits every completion. -/
def NoTwoCover (edges : List (List Coord)) : Prop :=
  ∀ placements, placements.length ≤ 2 → ¬ Covers edges placements

theorem uncovered_edge_of_noTwoCover (edges : List (List Coord)) (placements : List Coord)
    (hcover : NoTwoCover edges) (hsize : placements.length ≤ 2) :
    ∃ edge, edge ∈ edges ∧ ¬ Hits placements edge := by
  classical
  apply Classical.byContradiction
  intro hnone
  apply hcover placements hsize
  intro edge hedge
  apply Classical.byContradiction
  intro hunhit
  exact hnone ⟨edge, hedge, hunhit⟩

/-- The reusable logical core of the defender-cover reduction. Geometry and turn
semantics enter only through `survives`: an unhit pre-defense completion must
remain an executable immediate completion after the ordered defender reply. -/
theorem defender_cover_reduction
    (edges : List (List Coord))
    (afterDefense : List Coord → Position)
    (attacker : Player)
    (rules : Rules)
    (hcover : NoTwoCover edges)
    (survives : ∀ defense edge,
      defense.length ≤ 2 → edge ∈ edges → ¬ Hits defense edge →
      CompletionEdge rules (afterDefense defense) attacker edge) :
    ∀ defense, defense.length ≤ 2 →
      ∃ edge, edge ∈ edges ∧ CompletionEdge rules (afterDefense defense) attacker edge := by
  intro defense hsize
  rcases uncovered_edge_of_noTwoCover edges defense hcover hsize with
    ⟨edge, hedge, hunhit⟩
  exact ⟨edge, hedge, survives defense edge hsize hedge hunhit⟩

/-! Regression example for the exact-cover semantics used by Rust. -/

example (a b : Coord) : Covers [[a], [b]] [a, b] := by
  simp [Covers, Hits]

end Hexo
