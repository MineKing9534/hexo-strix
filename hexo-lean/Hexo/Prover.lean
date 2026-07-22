import Hexo.Threats

namespace Hexo

/-- Placement-bounded, game-semantic forced win. The attacker chooses placements;
the defender branch is universal over every legal placement. Draw and defender-win
outcomes satisfy neither branch, so neither can be certified as an attacker win. -/
def ForcesWin (rules : Rules) (attacker : Player) : Nat → LiveState → Prop
  | 0, _ => False
  | fuel + 1, s =>
      (s.toMove = attacker ∧
        ((∃ cell position moves,
            Step rules s cell (.won attacker position moves)) ∨
         (∃ cell next,
            Step rules s cell (.live next) ∧
            ForcesWin rules attacker fuel next))) ∨
      (s.toMove = attacker.other ∧
        ∀ cell, Legal s.position rules.placementRadius cell →
          ∃ next, Step rules s cell (.live next) ∧
            ForcesWin rules attacker fuel next)

/-- One-sided leaf-prover result. `unknown` covers exhaustion, disproval within a
restricted threat language, and every unsupported case; there is intentionally no
`NoWin` constructor. -/
inductive Verdict (goal : Prop) where
  | provenWin (proof : goal)
  | unknown

namespace Verdict

def Sound {goal : Prop} : Verdict goal → Prop
  | .provenWin _ => goal
  | .unknown => True

theorem sound {goal : Prop} (verdict : Verdict goal) : verdict.Sound := by
  cases verdict with
  | provenWin proof => exact proof
  | unknown => trivial

end Verdict

/-- Contract for an executable leaf prover. Rust may erase the proof object, but
every `provenWin` produced by the Lean oracle carries `ForcesWin` evidence. -/
abbrev LeafProver (rules : Rules) (attacker : Player) :=
  (fuel : Nat) → (state : LiveState) → ValidLiveState rules state →
    Verdict (ForcesWin rules attacker fuel state)

end Hexo
