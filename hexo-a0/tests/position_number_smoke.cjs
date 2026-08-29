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

// A forcing PV is relative to the attacker at the selected position. Its first
// displayed round is the rest of that attacker turn plus the defender's full
// reply; later attacker+defender rounds have the usual four placements.
assert.equal((src.match(/forcingLineNumber\(i, startDepth\)/g) || []).length, 2,
  "attacker and defender overlay labels must use the current position phase");
const forcingStart = src.indexOf("function forcingLineNumber(index, startDepth) {");
assert.ok(forcingStart > 0, "could not find forcingLineNumber in analysis.js");
let forcingEnd = forcingStart;
let forcingDepth = 0;
while (forcingEnd < src.length) {
  if (src[forcingEnd] === "{") forcingDepth++;
  else if (src[forcingEnd] === "}") {
    forcingDepth--;
    if (forcingDepth === 0) { forcingEnd++; break; }
  }
  forcingEnd++;
}
const forcingSrc = src.slice(forcingStart, forcingEnd);
function buildForcingFn(mode, absolutePositionNumber) {
  const swapped = forcingSrc.replace(/positionNumbering\(\)/g, JSON.stringify(mode));
  const compiled = new Function("index", "startDepth", "positionNumber",
    swapped + "\nreturn forcingLineNumber(index, startDepth);");
  return (index, startDepth) => compiled(index, startDepth, absolutePositionNumber);
}
try {
  const forcingRound = buildForcingFn("round", roundFn);
  const forcingPly = buildForcingFn("ply", plyFn);
  assert.deepEqual(Array.from({length: 8}, (_, i) => forcingRound(i, 0)),
    [1, 1, 1, 1, 2, 2, 2, 2], "full round from the seed");
  assert.deepEqual(Array.from({length: 8}, (_, i) => forcingRound(i, 1)),
    [1, 1, 1, 2, 2, 2, 2, 3], "one placement already used in the current turn");
  assert.deepEqual(Array.from({length: 8}, (_, i) => forcingRound(i, 2)),
    [1, 1, 1, 1, 2, 2, 2, 2], "fresh attacker turn starts a full forcing round");
  assert.deepEqual(Array.from({length: 8}, (_, i) => forcingRound(i, 3)),
    [1, 1, 1, 2, 2, 2, 2, 3], "halfway through the attacker turn leaves three cells in round 1");
  assert.deepEqual(Array.from({length: 8}, (_, i) => forcingRound(i, 4)),
    [1, 1, 1, 1, 2, 2, 2, 2], "new round boundary");
  assert.deepEqual(Array.from({length: 8}, (_, i) => forcingPly(i, 3)),
    [1, 2, 3, 4, 5, 6, 7, 8], "ply labels remain relative");
  console.log("OK: forcing-line rounds respect the attacker turn phase");
  pass++;
} catch (e) {
  console.log("FAIL: forcing-line numbering -", e.message);
  fail++;
}

if (fail) {
  console.log(fail + " failures, " + pass + " passed");
  process.exit(1);
}
console.log("OK positionNumber round-mode: " + pass + " cases (4-placement rounds under standard 2-stone opener)");
