import Std

namespace Hexo

/-! Infinite axial hex coordinates, represented by finite stone lists. -/

abbrev Coord := Int × Int

inductive Player where
  | p1
  | p2
  deriving Repr, DecidableEq

namespace Player

def other : Player → Player
  | p1 => p2
  | p2 => p1

@[simp] theorem other_other (p : Player) : p.other.other = p := by
  cases p <;> rfl

@[simp] theorem other_ne (p : Player) : p.other ≠ p := by
  cases p <;> decide

end Player

structure Stone where
  coord : Coord
  player : Player
  deriving Repr, DecidableEq

/-- A board has infinitely many cells but only finitely many occupied cells. -/
abbrev Position := List Stone

namespace Position

/-- Engine/solver inputs contain at most one stone per coordinate. -/
def WellFormed (p : Position) : Prop := (p.map Stone.coord).Nodup

def ownerAt : Position → Coord → Option Player
  | [], _ => none
  | stone :: rest, cell =>
      if stone.coord = cell then some stone.player else ownerAt rest cell

def empty (p : Position) (cell : Coord) : Prop := p.ownerAt cell = none

def occupied (p : Position) (cell : Coord) : Prop := ∃ who, p.ownerAt cell = some who

/-- Search make: legal callers ensure the coordinate is empty. -/
def place (p : Position) (cell : Coord) (who : Player) : Position :=
  { coord := cell, player := who } :: p

@[simp] theorem ownerAt_place_same (p : Position) (cell : Coord) (who : Player) :
    (p.place cell who).ownerAt cell = some who := by
  simp [place, ownerAt]

@[simp] theorem ownerAt_place_ne (p : Position) (cell query : Coord) (who : Player)
    (h : cell ≠ query) :
    (p.place cell who).ownerAt query = p.ownerAt query := by
  simp [place, ownerAt, h]

@[simp] theorem place_not_empty (p : Position) (cell : Coord) (who : Player) :
    ¬ (p.place cell who).empty cell := by
  simp [empty]

end Position

namespace Coord

def add (a b : Coord) : Coord := (a.1 + b.1, a.2 + b.2)
def sub (a b : Coord) : Coord := (a.1 - b.1, a.2 - b.2)
def scale (n : Int) (a : Coord) : Coord := (n * a.1, n * a.2)

/-- Axial-coordinate hex distance: max(|dq|, |dr|, |dq+dr|). -/
def hexDist (a b : Coord) : Nat :=
  let dq := a.1 - b.1
  let dr := a.2 - b.2
  max dq.natAbs (max dr.natAbs (dq + dr).natAbs)

@[simp] theorem hexDist_self (a : Coord) : hexDist a a = 0 := by
  simp [hexDist]

end Coord

structure Rules where
  winLength : Nat
  placementRadius : Nat
  maxMoves : Nat
  deriving Repr, DecidableEq

def ValidRules (r : Rules) : Prop :=
  2 ≤ r.winLength ∧ 1 ≤ r.placementRadius ∧ 1 ≤ r.maxMoves

def Reachable (p : Position) (radius : Nat) (cell : Coord) : Prop :=
  ∃ stone ∈ p, Coord.hexDist stone.coord cell ≤ radius

def Legal (p : Position) (radius : Nat) (cell : Coord) : Prop :=
  p.empty cell ∧ Reachable p radius cell

theorem reachable_after_place (p : Position) (radius : Nat) (cell anchor : Coord)
    (who : Player) (h : Reachable p radius cell) :
    Reachable (p.place anchor who) radius cell := by
  rcases h with ⟨stone, hstone, hdist⟩
  exact ⟨stone, by simp [Position.place, hstone], hdist⟩

theorem reachable_from_new_stone (p : Position) (radius : Nat) (cell anchor : Coord)
    (who : Player) (hdist : Coord.hexDist anchor cell ≤ radius) :
    Reachable (p.place anchor who) radius cell := by
  exact ⟨{ coord := anchor, player := who }, by simp [Position.place], hdist⟩

inductive PlacementsLeft where
  | one
  | two
  deriving Repr, DecidableEq

structure LiveState where
  position : Position
  toMove : Player
  placementsLeft : PlacementsLeft
  movesPlayed : Nat
  deriving Repr

/-- Domain invariant for states handed to the leaf prover. Engine games always
have a nonempty, collision-free board and have not consumed the draw quota. -/
def ValidLiveState (rules : Rules) (s : LiveState) : Prop :=
  ValidRules rules ∧
  s.position ≠ [] ∧
  s.position.WellFormed ∧
  s.movesPlayed < rules.maxMoves

inductive PlacementOutcome where
  | won (winner : Player) (position : Position) (movesPlayed : Nat)
  | draw (position : Position) (movesPlayed : Nat)
  | live (state : LiveState)
  deriving Repr

end Hexo
