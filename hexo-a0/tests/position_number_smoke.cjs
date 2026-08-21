const assert = require("node:assert/strict");
const fs = require("node:fs");
const src = fs.readFileSync(__dirname + "/../src/hexo_a0/serving/static/analysis.js", "utf8");

// Extract positionNumber function text
const startIdx = src.indexOf("function positionNumber(depth) {");
assert.ok(startIdx > 0, "could not find positionNumber in analysis.js");
let endIdx = startIdx;
let depth = 0;
while (endIdx < src.length) {
  if (src[endIdx] === "{") depth++;
  else if (src[endIdx] === "}") {
    depth--;
    if (depth === 0) { endIdx++; break; }
  }
  endIdx++;
}
const fnSrc = src.slice(startIdx, endIdx);

// Substitute the global positionNumbering() call with a known literal so we
// can test both modes in isolation.
function buildFn(mode) {
  const swapped = fnSrc.replace(/positionNumbering\(\)/g, JSON.stringify(mode));
  return new Function("depth", swapped + "\nreturn positionNumber(depth);");
}

const plyFn = buildFn("ply");
const roundFn = buildFn("round");

// Standard HeXO: every turn places 2 stones; the seed (depth 0) is P1's (0,0).
// Under round numbering, round 1 spans P2's 1st+2nd and P1's 1st+2nd (depths 1..4)
// for a total of 4 placements; every subsequent round also holds 4.
const cases = [
  [0, 1, 0, "seed: ply 1, round 0 (no placement)"],
  [1, 2, 1, "P2 1st: round 1"],
  [2, 3, 1, "P2 2nd: round 1"],
  [3, 4, 1, "P1 1st: round 1"],
  [4, 5, 1, "P1 2nd: round 1 (round 1 = 4 placements)"],
  [5, 6, 2, "P2 1st: round 2"],
  [8, 9, 2, "P1 2nd: round 2"],
  [9, 10, 3, "P2 1st: round 3"],
  [12, 13, 3, "P1 2nd: round 3"],
];

let pass = 0, fail = 0;
for (const [d, expectedPly, expectedRound, label] of cases) {
  try {
    assert.equal(plyFn(d), expectedPly, "ply(" + d + "): " + label);
    assert.equal(roundFn(d), expectedRound, "round(" + d + "): " + label);
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
console.log("OK positionNumber round-mode: " + pass + " cases (4-placement rounds under standard 2-stone opener)");
