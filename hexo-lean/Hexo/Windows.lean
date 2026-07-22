import Hexo.Rules

namespace Hexo

inductive Axis where
  | q
  | r
  | diag
  deriving Repr, DecidableEq

namespace Axis

def delta : Axis → Coord
  | q => (1, 0)
  | r => (0, 1)
  | diag => (1, -1)

def all : List Axis := [.q, .r, .diag]

end Axis

structure Window where
  start : Coord
  axis : Axis
  size : Nat
  deriving Repr, DecidableEq

namespace Window

def cells (w : Window) : List Coord :=
  (List.range w.size).map fun i =>
    Coord.add w.start (Coord.scale (Int.ofNat i) w.axis.delta)

def contains (w : Window) (cell : Coord) : Prop := cell ∈ w.cells

def through (cell : Coord) (axis : Axis) (size offset : Nat) : Window :=
  { start := Coord.sub cell (Coord.scale (Int.ofNat offset) axis.delta)
    axis := axis
    size := size }

theorem through_contains (cell : Coord) (axis : Axis) (size offset : Nat)
    (hoffset : offset < size) :
    (through cell axis size offset).contains cell := by
  simp only [contains, cells, through, List.mem_map, List.mem_range]
  refine ⟨offset, hoffset, ?_⟩
  cases axis <;>
    apply Prod.ext <;>
    simp [Coord.add, Coord.sub, Coord.scale, Axis.delta] <;>
    omega

def affectedBy (size : Nat) (cell : Coord) : List Window :=
  Axis.all.flatMap fun axis =>
    (List.range size).map fun offset => through cell axis size offset

@[simp] theorem affectedBy_length (size : Nat) (cell : Coord) :
    (affectedBy size cell).length = 3 * size := by
  simp [affectedBy, Axis.all]
  omega

theorem mem_affectedBy_contains (size : Nat) (cell : Coord) (w : Window)
    (h : w ∈ affectedBy size cell) : w.contains cell := by
  simp only [affectedBy, Axis.all, List.mem_flatMap, List.mem_cons,
    List.not_mem_nil, or_false, List.mem_map, List.mem_range] at h
  rcases h with ⟨axis, haxis, offset, hoffset, rfl⟩
  exact through_contains cell axis size offset hoffset

theorem contains_mem_affectedBy (size : Nat) (cell : Coord) (w : Window)
    (hsize : w.size = size) (h : w.contains cell) :
    w ∈ affectedBy size cell := by
  rcases w with ⟨start, axis, actualSize⟩
  simp only [contains, cells, List.mem_map, List.mem_range] at h
  rcases h with ⟨offset, hoffset, hcell⟩
  change actualSize = size at hsize
  subst actualSize
  rcases start with ⟨startQ, startR⟩
  rcases cell with ⟨cellQ, cellR⟩
  have hwindow :
      { start := (startQ, startR), axis := axis, size := size } =
        through (cellQ, cellR) axis size offset := by
    cases axis <;>
      simp [through, Coord.add, Coord.sub, Coord.scale, Axis.delta, Prod.ext_iff] at hcell ⊢ <;>
      omega
  rw [hwindow]
  simp only [affectedBy, Axis.all, List.mem_flatMap, List.mem_cons,
    List.not_mem_nil, or_false, List.mem_map, List.mem_range]
  cases axis with
  | q => exact ⟨Axis.q, Or.inl rfl, offset, hoffset, rfl⟩
  | r => exact ⟨Axis.r, Or.inr (Or.inl rfl), offset, hoffset, rfl⟩
  | diag => exact ⟨Axis.diag, Or.inr (Or.inr rfl), offset, hoffset, rfl⟩

end Window

def OwnsWindow (p : Position) (who : Player) (w : Window) : Prop :=
  ∀ cell ∈ w.cells, p.ownerAt cell = some who

def CleanWindow (p : Position) (who : Player) (w : Window) : Prop :=
  ∀ cell ∈ w.cells, p.ownerAt cell ≠ some who.other

/-- A winning window created through the just-placed stone. -/
def WinsThrough (p : Position) (who : Player) (winLength : Nat) (cell : Coord) : Prop :=
  ∃ axis offset, offset < winLength ∧
    OwnsWindow p who (Window.through cell axis winLength offset)

/-! Exact placement-level semantics. Sequential use makes a two-stone turn ordered:
the second legality check sees the first stone, and a win stops the turn immediately. -/

inductive Step (rules : Rules) : LiveState → Coord → PlacementOutcome → Prop where
  | win {s : LiveState} {cell : Coord}
      (legal : Legal s.position rules.placementRadius cell)
      (wins : WinsThrough (s.position.place cell s.toMove) s.toMove rules.winLength cell) :
      Step rules s cell
        (.won s.toMove (s.position.place cell s.toMove) (s.movesPlayed + 1))
  | draw {s : LiveState} {cell : Coord}
      (legal : Legal s.position rules.placementRadius cell)
      (notWin : ¬ WinsThrough (s.position.place cell s.toMove) s.toMove rules.winLength cell)
      (quota : rules.maxMoves ≤ s.movesPlayed + 1) :
      Step rules s cell (.draw (s.position.place cell s.toMove) (s.movesPlayed + 1))
  | first {s : LiveState} {cell : Coord}
      (left : s.placementsLeft = .two)
      (legal : Legal s.position rules.placementRadius cell)
      (notWin : ¬ WinsThrough (s.position.place cell s.toMove) s.toMove rules.winLength cell)
      (underQuota : s.movesPlayed + 1 < rules.maxMoves) :
      Step rules s cell (.live {
        position := s.position.place cell s.toMove
        toMove := s.toMove
        placementsLeft := .one
        movesPlayed := s.movesPlayed + 1 })
  | second {s : LiveState} {cell : Coord}
      (left : s.placementsLeft = .one)
      (legal : Legal s.position rules.placementRadius cell)
      (notWin : ¬ WinsThrough (s.position.place cell s.toMove) s.toMove rules.winLength cell)
      (underQuota : s.movesPlayed + 1 < rules.maxMoves) :
      Step rules s cell (.live {
        position := s.position.place cell s.toMove
        toMove := s.toMove.other
        placementsLeft := .two
        movesPlayed := s.movesPlayed + 1 })

/-! A formally checked dirty-window kernel. Production can store richer counters;
the proof establishes that exactly the `3 * winLength` windows containing a new
cell need recomputation. -/

abbrev WindowSnapshot := List (Coord × Option Player)
abbrev WindowKernel := Window → WindowSnapshot

def observeWindow (p : Position) (w : Window) : WindowSnapshot :=
  w.cells.map fun cell => (cell, p.ownerAt cell)

def KernelCorrect (size : Nat) (p : Position) (kernel : WindowKernel) : Prop :=
  ∀ w, w.size = size → kernel w = observeWindow p w

theorem observe_place_unaffected (p : Position) (who : Player) (placed : Coord)
    (w : Window) (h : ¬ w.contains placed) :
    observeWindow (p.place placed who) w = observeWindow p w := by
  unfold observeWindow Window.contains at *
  have go : ∀ cells : List Coord, placed ∉ cells →
      cells.map (fun cell => (cell, (p.place placed who).ownerAt cell)) =
      cells.map (fun cell => (cell, p.ownerAt cell)) := by
    intro cells hmem
    induction cells with
    | nil => rfl
    | cons cell rest ih =>
        simp only [List.mem_cons, not_or] at hmem
        simp [Position.ownerAt_place_ne, hmem.1, ih hmem.2]
  exact go w.cells h

def patchKernel (size : Nat) (p : Position) (placed : Coord) (who : Player)
    (old : WindowKernel) : WindowKernel :=
  fun w =>
    if w ∈ Window.affectedBy size placed
    then observeWindow (p.place placed who) w
    else old w

theorem patchKernel_correct (size : Nat) (p : Position) (placed : Coord) (who : Player)
    (old : WindowKernel) (hold : KernelCorrect size p old) :
    KernelCorrect size (p.place placed who) (patchKernel size p placed who old) := by
  intro w hsize
  classical
  by_cases haffected : w ∈ Window.affectedBy size placed
  · simp [patchKernel, haffected]
  · have houtside : ¬ w.contains placed := by
      intro hcontains
      exact haffected (Window.contains_mem_affectedBy size placed w hsize hcontains)
    simp [patchKernel, haffected, hold w hsize, observe_place_unaffected p who placed w houtside]

end Hexo
