const assert = require("node:assert/strict");
const { proofOptimizationDescription } = require("../src/hexo_a0/serving/static/proof-explorer.js");

const cases = [
  {
    label: "tightest: shortestCertified && cert matches the new upper bound",
    bundle: {
      verification: {maxAttackerTurns: 18, dagNodes: 100, proofEdges: 200},
      optimization: {method: "pdspn-shortest-v1", bestUpperDepth: 18, excludedThroughDepth: 17, shortestCertified: true},
    },
    expectShort: "shortest win: 18 turns",
    expectFullIncludes: "Every reply shown here was checked on this device",
    expectFullExcludes: ["saved replies shown here prove a win within"],
  },
  {
    label: "defensive: shortestCertified && cert stale (older bundle)",
    bundle: {
      verification: {maxAttackerTurns: 21, dagNodes: 100, proofEdges: 200},
      optimization: {method: "pdspn-shortest-v1", bestUpperDepth: 18, excludedThroughDepth: 17, shortestCertified: true},
    },
    expectFullIncludes: "saved replies still prove a win within 21 turns",
    expectFullExcludes: ["Every reply shown here was checked on this device"],
  },
  {
    label: "search upper only: budget exhausted, not shortestCertified",
    bundle: {
      verification: {maxAttackerTurns: 21, dagNodes: 100, proofEdges: 200},
      optimization: {method: "pdspn-shortest-v1", bestUpperDepth: 18, excludedThroughDepth: 10, shortestCertified: false},
    },
    expectFullIncludes: "shortest win takes between 11 and 18 turns",
    expectFullExcludes: ["cannot force one sooner"],
  },
  {
    label: "search upper with no lower bound excluded",
    bundle: {
      verification: {maxAttackerTurns: 30, dagNodes: 100, proofEdges: 200},
      optimization: {method: "pdspn-shortest-v1", bestUpperDepth: 18, excludedThroughDepth: 0, shortestCertified: false},
    },
    expectFullIncludes: "did not prove whether a faster win exists",
    expectFullExcludes: ["cannot force one sooner"],
  },
  {
    label: "no optimization metadata returns null",
    bundle: {verification: {maxAttackerTurns: 18}},
    expectNull: true,
  },
];

let pass = 0;
let fail = 0;
for (const c of cases) {
  const desc = proofOptimizationDescription(c.bundle);
  try {
    if (c.expectNull) {
      assert.equal(desc, null, c.label);
    } else {
      assert.ok(desc && typeof desc.short === "string", c.label + ": got no description");
      if (c.expectShort) assert.equal(desc.short, c.expectShort, c.label + ": short");
      if (c.expectFullIncludes) assert.ok(
        desc.full.includes(c.expectFullIncludes),
        c.label + ": full should include " + JSON.stringify(c.expectFullIncludes) + ", got " + JSON.stringify(desc.full),
      );
      for (const s of c.expectFullExcludes || []) {
        assert.ok(
          !desc.full.includes(s),
          c.label + ": full should NOT include " + JSON.stringify(s) + ", got " + JSON.stringify(desc.full),
        );
      }
    }
    console.log("OK:", c.label);
    pass++;
  } catch (e) {
    console.log("FAIL:", c.label, "-", e.message);
    fail++;
  }
}
if (fail) {
  console.log(fail + " failures, " + pass + " passed");
  process.exit(1);
}
console.log("OK proofOptimizationDescription: " + pass + " cases");
