// Exercise the production browser worker against the real exported model.
import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {fileURLToPath, pathToFileURL} from "node:url";
import {dirname, resolve} from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const workerUrl = pathToFileURL(resolve(here, "../src/hexo_a0/serving/static/inference-worker.js"));
workerUrl.search = "?v=smoke";
const weightsUrl = pathToFileURL(resolve(here, "../../exports/strixbot-rel2.safetensors"));

globalThis.self = globalThis;
self.location = workerUrl;
const nativeFetch = globalThis.fetch;
globalThis.fetch = async input => {
  const url = input instanceof URL ? input : new URL(String(input));
  if (url.protocol !== "file:") return nativeFetch(input);
  return new Response(await readFile(fileURLToPath(url)), {
    status: 200,
    headers: {"Content-Type": url.pathname.endsWith(".wasm")
      ? "application/wasm" : "application/octet-stream"},
  });
};
const posted = [];
self.postMessage = message => posted.push(message);
await import(workerUrl.href);

await self.onmessage({data: {
  type: "analyzeGame", requestId: 1, modelUrl: weightsUrl.href,
  moves: [[0, 0], [1, 0], [0, 1]],
  config: {win_length: 6, placement_radius: 6, max_moves: 400},
  sims: 8, mActions: 4,
}});
const completed = posted.at(-1);
assert.equal(completed.type, "game", completed.error || "worker did not complete");
assert.equal(completed.result.trajectory.length, 3);
for (const entry of completed.result.trajectory) {
  assert.equal(entry.legal.length, entry.q_hat.length);
  assert.equal(entry.legal.length, entry.improved_policy.length);
  assert.equal(entry.legal.length, entry.candidate_set.length);
}
assert.deepEqual(completed.result.boundary_indices, [0, 2]);

posted.length = 0;
await self.onmessage({data: {
  type: "bestMove", requestId: 2, modelUrl: weightsUrl.href,
  position: {
    config: {win_length: 6, placement_radius: 6, max_moves: 400},
    stones: [{q: 0, r: 0, player: 1}], to_move: 2, moves_remaining: 2,
  },
  sims: 8, mActions: 4,
}});
assert.equal(posted.at(-1).type, "bestMove");
assert.ok(Number.isInteger(posted.at(-1).result.move.q));
console.log("OK browser-local inference worker: play + trajectory analysis");
