const fs = require("node:fs");
const assert = require("node:assert/strict");

const ui = fs.readFileSync("hexo-a0/src/hexo_a0/serving/static/observatory-ui.js", "utf8");
const analysis = fs.readFileSync("hexo-a0/src/hexo_a0/serving/static/analysis.js", "utf8");
const worker = fs.readFileSync("hexo-a0/src/hexo_a0/serving/static/solver-worker.js", "utf8");

assert.match(ui, /value="pdspn-shortest" selected/);
assert.match(ui, /Prove the shortest win/);
assert.doesNotMatch(ui, /Second yes-or-no check|Find any win · DFPN|value="pns"|value="dfpn"/);
assert.match(ui, /id="analysis-forcing-effort"/);
assert.doesNotMatch(ui, /analysis-forcing-budget|analysis-forcing-leaf-budget|million steps|branch<\/span>/);
assert.match(analysis, /let forcingUiEngine = "pdspn-shortest"/);
assert.match(analysis, /PDS_PORTFOLIO_BRANCH_BUDGETS/);
assert.doesNotMatch(worker, /pns: api\.SolverEngineEnum|dfpn: api\.SolverEngineEnum/);
console.log("proof lab default method smoke: ok");
