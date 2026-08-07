const assert = require("node:assert/strict");
const {
  buildProofModel,
  applyProofAction,
  normalizeProofPosition,
} = require("../src/hexo_a0/serving/static/proof-explorer.js");

// Child nodes precede parents, matching certificates emitted by the Rust
// builder. The root has a two-turn recommended attack and a three-turn
// already-certified alternative.
const bundle = {
  format: "hexo-pdspn-proof-bundle-v1",
  position: {
    stones: [[0, 0, "P1"]],
    attacker: "P1",
    placements_remaining: 2,
    config: {win_length: 6, placement_radius: 8, max_moves: 400},
  },
  verification: {dagNodes: 6, proofEdges: 6, maxAttackerTurns: 2},
  certificate: {
    version: 1,
    width: "tight",
    root: 5,
    nodes: [
      {kind: "immediate_win", action: [[4, 0], [5, 0]]},
      {kind: "unstoppable"},
      {kind: "defender_replies", responses: [
        {action: [[2, 0], [3, 0]], child: 0},
        {action: [[2, 1], [3, 1]], child: 1},
      ]},
      {kind: "attacker_move", action: [[4, 1], [5, 1]], child: 1},
      {kind: "defender_replies", responses: [
        {action: [[2, 2], [3, 2]], child: 3},
      ]},
      {kind: "attacker_move", action: [[0, 1], [1, 1]], child: 2,
        alternatives: [{action: [[0, 2], [1, 2]], child: 4}]},
    ],
  },
};

const model = buildProofModel(bundle);
assert.equal(model.maxAttackerTurns, 2);
assert.equal(model.edges, 6);
assert.equal(model.remaining(2), 1);
assert.equal(model.worstResponseIndex(model.nodes[2]), 0, "ties use stable certificate order");
assert.equal(model.parents[0], 1);
assert.equal(model.alternativeAttackerNodes, 1);
assert.equal(model.attackerAlternatives, 1);
assert.deepEqual(model.nearestAlternativePath(model.root), [],
  "the root itself is already an alternative attacker node");
assert.deepEqual(
  model.attackerChoices(model.nodes[5]).map(choice => 1 + model.remaining(choice.child)),
  [2, 3],
  "attacker alternatives retain their verified worst-case ranking",
);

const stones = new Map([["0,0", "P1"]]);
const afterAttack = applyProofAction(stones, model.nodes[5].action, "P1");
assert.equal(stones.size, 1, "replay must not mutate the prior breadcrumb state");
assert.equal(afterAttack.get("1,1"), "P1");
const afterDefense = applyProofAction(afterAttack, model.nodes[2].responses[0].action, "P2");
assert.equal(afterDefense.get("3,0"), "P2");
assert.throws(() => applyProofAction(afterDefense, [[0, 0]], "P2"), /occupied cell/);

const oldPosition = normalizeProofPosition({
  stonesFlat: [0, 0, 1, 2, 3, 2],
  toMove: "P1", movesRemaining: 2,
  winLength: 6, placementRadius: 8, maxMoves: 400,
});
assert.deepEqual(oldPosition.stones, [[0, 0, "P1"], [2, 3, "P2"]]);
assert.equal(oldPosition.placements_remaining, 2);

const mismatch = structuredClone(bundle);
mismatch.verification.maxAttackerTurns = 3;
assert.throws(() => buildProofModel(mismatch), /summary does not match/);

const misranked = structuredClone(bundle);
const primary = {
  action: misranked.certificate.nodes[5].action,
  child: misranked.certificate.nodes[5].child,
};
const alternative = misranked.certificate.nodes[5].alternatives[0];
misranked.certificate.nodes[5].action = alternative.action;
misranked.certificate.nodes[5].child = alternative.child;
misranked.certificate.nodes[5].alternatives = [primary];
assert.throws(() => buildProofModel(misranked), /not depth-ranked/);

console.log("OK proof explorer model: ranked attacks, worst defense, replay, compatibility");
