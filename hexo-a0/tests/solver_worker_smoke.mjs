// Exercise the exact browser-target glue and Observatory worker protocol under
// Node. A tiny file:// fetch shim stands in for the HTTP static server.
import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {fileURLToPath, pathToFileURL} from "node:url";
import {dirname, resolve} from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const workerUrl = pathToFileURL(resolve(
  here, "../src/hexo_a0/serving/static/solver-worker.js",
));
workerUrl.search = "?v=smoke";

globalThis.self = globalThis;
self.location = workerUrl;
const nativeFetch = globalThis.fetch;
globalThis.fetch = async (input, init) => {
  const url = input instanceof URL ? input : new URL(String(input));
  if (url.protocol !== "file:") return nativeFetch(input, init);
  const bytes = await readFile(fileURLToPath(url));
  return new Response(bytes, {
    status: 200,
    headers: {"Content-Type": url.pathname.endsWith(".wasm")
      ? "application/wasm" : "text/javascript"},
  });
};

const posted = [];
self.postMessage = message => posted.push(message);
await import(workerUrl.href);

const position = {
  winLength: 4,
  placementRadius: 8,
  maxMoves: 300,
  toMove: "P1",
  movesRemaining: 2,
  stonesFlat: [0, 0, 1, 1, 0, 1, 2, 0, 1, 5, 5, 2],
};

// With a fixture argument, this doubles as a reproducible long-puzzle check of
// the exact deployed worker/glue. Example:
//   node solver_worker_smoke.mjs ../../scripts/fixtures/forcing_puzzles/0l4291i_live.json pdspn tight 60 100000000 5000
const fixturePath = process.argv[2];
if (fixturePath) {
  const fixture = JSON.parse(await readFile(resolve(process.cwd(), fixturePath), "utf8"));
  if (process.argv[3] === "verify") {
    const positionFixturePath = process.argv[4];
    const root = fixture.position || (positionFixturePath
      ? JSON.parse(await readFile(resolve(process.cwd(), positionFixturePath), "utf8"))
      : null);
    assert.ok(root, "a standalone certificate report needs a position-fixture argument");
    const config = root.config || {win_length: 6, placement_radius: 8, max_moves: 400};
    const verifyPosition = {
      winLength: config.win_length,
      placementRadius: config.placement_radius,
      maxMoves: config.max_moves,
      toMove: root.attacker,
      movesRemaining: root.placements_remaining,
      stonesFlat: root.stones.flatMap(([q, r, player]) => [q, r, player === "P1" ? 1 : 2]),
    };
    posted.length = 0;
    await self.onmessage({data: {
      type: "verify",
      requestId: "fixture-verify",
      position: verifyPosition,
      certificate: fixture.certificate,
    }});
    assert.equal(posted.length, 1, "fixture verifier should post exactly one response");
    assert.equal(posted[0].type, "verified", posted[0].error || "fixture verifier error");
    if (fixture.verification) assert.deepEqual(posted[0].summary, fixture.verification);
    console.log(JSON.stringify(posted[0].summary, null, 2));
    process.exit(0);
  }
  const engine = process.argv[3] || "pdspn";
  const width = process.argv[4] || "tight";
  const depthCap = Number(process.argv[5] || 60);
  const nodeBudget = process.argv[6] || "100000000";
  const leafNodeBudget = process.argv[7] || "5000";
  const certificatePath = process.argv[8];
  const certificateReport = certificatePath
    ? JSON.parse(await readFile(resolve(process.cwd(), certificatePath), "utf8"))
    : null;
  const fixturePosition = {
    winLength: 6,
    placementRadius: 8,
    maxMoves: 400,
    toMove: fixture.attacker,
    movesRemaining: fixture.placements_remaining,
    stonesFlat: fixture.stones.flatMap(([q, r, player]) => [q, r, player === "P1" ? 1 : 2]),
  };
  posted.length = 0;
  console.error(
    `Running ${engine} ${width} through Observatory WASM: depth/recovery=${depthCap}, budget=${nodeBudget}, leaf=${leafNodeBudget}`,
  );
  await self.onmessage({data: {
    type: "solve",
    requestId: "fixture",
    position: fixturePosition,
    engine,
    width,
    depthCap,
    nodeBudget,
    leafNodeBudget,
    certificate: certificateReport ? certificateReport.certificate : null,
  }});
  assert.equal(posted.length, 1, "fixture should post exactly one response");
  const message = posted[0];
  assert.equal(message.type, "result", message.error || "fixture worker error");
  const {certificate, ...summary} = message.result;
  console.log(JSON.stringify({
    ...summary,
    certificate: certificate ? {
      version: certificate.version,
      width: certificate.width,
      root: certificate.root,
      nodes: certificate.nodes.length,
    } : null,
  }, null, 2));
  process.exit(message.result.kind === "win" ? 0 : 2);
}

let pdspnCertificate = null;
for (const [requestId, engine] of ["idtt", "pdspn"].entries()) {
  posted.length = 0;
  await self.onmessage({data: {
    type: "solve",
    requestId,
    position,
    engine,
    width: "wide",
    depthCap: 6,
    nodeBudget: "20000",
    leafNodeBudget: "5000",
  }});
  assert.equal(posted.length, 1, `${engine} should post exactly one response`);
  const message = posted[0];
  assert.equal(message.type, "result", `${engine}: ${message.error || "no result"}`);
  assert.equal(message.requestId, requestId);
  assert.equal(message.result.kind, "win");
  assert.ok(message.result.elapsedMs >= 0);
  assert.ok(message.result.depth >= 1);
  assert.ok(message.result.pv.length > 0);
  assert.equal(message.result.pv.length, message.result.pvOwners.length);
  if (engine === "idtt") assert.equal(message.result.nodes, "0");
  else assert.ok(BigInt(message.result.nodes) > 0n);
  if (engine === "pdspn") {
    assert.ok(message.result.certificate, "PDS-PN win should include a proof certificate");
    assert.equal(message.result.certificate.version, 1);
    assert.ok(message.result.certificate.nodes.length > 0);
    assert.ok(message.result.certificateSummary.dagNodes > 0);
    assert.ok(message.result.certificateSummary.proofEdges >= 0);
    assert.ok(message.result.certificateSummary.maxAttackerTurns > 0);
    const expectedSummary = message.result.certificateSummary;
    const certificate = message.result.certificate;
    pdspnCertificate = certificate;
    posted.length = 0;
    await self.onmessage({data: {
      type: "verify",
      requestId: "verify-pdspn",
      position,
      certificate,
    }});
    assert.equal(posted.length, 1, "certificate verifier should post exactly one response");
    assert.equal(posted[0].type, "verified", posted[0].error || "certificate verifier error");
    assert.deepEqual(posted[0].summary, expectedSummary);

    const tampered = structuredClone(certificate);
    tampered.root = tampered.nodes.length;
    posted.length = 0;
    await self.onmessage({data: {
      type: "verify",
      requestId: "verify-tampered",
      position,
      certificate: tampered,
    }});
    assert.equal(posted.length, 1, "tampered verifier should post exactly one response");
    assert.equal(posted[0].type, "error", "tampered certificate must be rejected");
  } else {
    assert.equal(message.result.certificate, null);
    assert.equal(message.result.certificateSummary, null);
  }
}

posted.length = 0;
await self.onmessage({data: {
  type: "solve",
  requestId: "pdspn-shortest",
  position,
  engine: "pdspn-shortest",
  width: "wide",
  depthCap: 6,
  nodeBudget: "20000",
  leafNodeBudget: "2000",
  certificate: pdspnCertificate,
}});
assert.equal(posted.length, 1, "PDS-PN shortest should post exactly one response");
assert.equal(posted[0].type, "result", posted[0].error || "PDS-PN shortest error");
assert.equal(posted[0].result.kind, "win");
assert.equal(posted[0].result.shortestCertified, true);
assert.equal(posted[0].result.bestUpperDepth, 1);
assert.equal(posted[0].result.excludedThroughDepth, 0);
assert.ok(posted[0].result.certificate);
assert.ok(Array.isArray(posted[0].result.turns));

console.log("OK Observatory solver worker: IDTT, PDS-PN, shortest PDS-PN, certificate verifier");
