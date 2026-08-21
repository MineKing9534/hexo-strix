const assert = require("node:assert/strict");

// The proof explorer's sample line is a sequence of turns, each with 1 or 2 cells
// (depending on moves_remaining at the start of that turn). Under round
// numbering, round k spans turns 2k-2 and 2k-1 — so a cell in turnIndex T gets
// round = floor(T / 2) + 1. This works regardless of whether a turn has 1 or 2
// cells (e.g. a turn that started with moves_remaining=1).
function roundOfTurn(turnIndex) { return Math.floor(turnIndex / 2) + 1; }

const cases = [
  [0, 1, "P1's 1st turn (round 1)"],
  [1, 1, "P2's 1st turn (round 1, paired with turn 0)"],
  [2, 2, "P1's 2nd turn (round 2)"],
  [3, 2, "P2's 2nd turn (round 2, paired with turn 2)"],
  [4, 3, "P1's 3rd turn (round 3)"],
  [5, 3, "P2's 3rd turn (round 3, paired)"],
  [7, 4, "Even-indexed turn, 4th pair"],
];

let pass = 0, fail = 0;
for (const [turnIndex, expectedRound, label] of cases) {
  try {
    assert.equal(roundOfTurn(turnIndex), expectedRound, "turn " + turnIndex + ": " + label);
    console.log("OK:", label);
    pass++;
  } catch (e) {
    console.log("FAIL:", label, "-", e.message);
    fail++;
  }
}

if (fail) {
  console.log(fail + " failures, " + pass + " passed");
  process.exit(1);
}
console.log("OK proof explorer round labeling: " + pass + " cases (paired turns respect moves_remaining per turn)");
