const assert = require("node:assert/strict");
const { copyProofText } = require("../src/hexo_a0/serving/static/proof-explorer.js");

// Mock the browser environment. Node 21+ defines a built-in `navigator`
// global, so we mutate it in place rather than reassigning.
const clipboardCalls = [];
const execCommandCalls = [];
let clipboardShouldReject = false;
let execCommandShouldFail = false;

const writeText = async (text) => {
  clipboardCalls.push(text);
  if (clipboardShouldReject) {
    throw new DOMException("Write permission denied.", "NotAllowedError");
  }
};

const installClipboard = () => {
  navigator.clipboard = { writeText };
};

const clearClipboard = () => {
  delete navigator.clipboard;
};

let lastAppended = null;
let lastRemoved = null;
let selectCallCount = 0;
let focusCallCount = 0;
let setSelectionRangeCallCount = 0;
global.document = {
  createElement: (tag) => {
    const field = {
      tagName: tag.toUpperCase(),
      value: "",
      _attrs: {},
      style: {},
      setAttribute(name, value) { this._attrs[name] = value; },
      getAttribute(name) { return this._attrs[name]; },
      select() { selectCallCount++; },
      focus(opts) { focusCallCount++; },
      setSelectionRange(start, end) {
        setSelectionRangeCallCount++;
        this._start = start;
        this._end = end;
      },
      remove() { lastRemoved = this; },
    };
    return field;
  },
  body: {
    appendChild(el) { lastAppended = el; return el; },
  },
  execCommand: (cmd) => {
    execCommandCalls.push(cmd);
    if (execCommandShouldFail) return false;
    return true;
  },
  getSelection: () => null,
};

(async () => {
  installClipboard();

  // Test 1: Modern API succeeds - no fallback
  clipboardShouldReject = false;
  execCommandShouldFail = false;
  clipboardCalls.length = 0;
  execCommandCalls.length = 0;
  await copyProofText("https://example.com/proof/abc1");
  assert.equal(clipboardCalls.length, 1, "modern API called");
  assert.equal(execCommandCalls.length, 0, "fallback not used");
  console.log("test1 OK");

  // Test 2: Modern API rejects - falls back to execCommand
  clipboardShouldReject = true;
  execCommandShouldFail = false;
  clipboardCalls.length = 0;
  execCommandCalls.length = 0;
  await copyProofText("https://example.com/proof/abc2");
  assert.equal(clipboardCalls.length, 1, "modern API tried");
  assert.equal(execCommandCalls.length, 1, "fallback used after rejection");
  assert.equal(execCommandCalls[0], "copy", "execCommand('copy') called");
  console.log("test2 OK");

  // Test 3: Both fail - throws meaningful error
  clipboardShouldReject = true;
  execCommandShouldFail = true;
  clipboardCalls.length = 0;
  execCommandCalls.length = 0;
  let caught = null;
  try {
    await copyProofText("https://example.com/proof/abc3");
  } catch (e) {
    caught = e;
  }
  assert.ok(caught, "should throw when both paths fail");
  assert.match(caught.message, /blocked clipboard access/i);
  assert.equal(clipboardCalls.length, 1);
  assert.equal(execCommandCalls.length, 1);
  console.log("test3 OK");

  // Test 4: No modern API at all - uses fallback directly
  clearClipboard();
  execCommandShouldFail = false;
  execCommandCalls.length = 0;
  await copyProofText("https://example.com/proof/abc4");
  assert.equal(execCommandCalls.length, 1, "fallback used when no modern API");
  installClipboard();
  console.log("test4 OK");

  console.log("OK copyProofText: modern path, fallback after rejection, dual failure surfaces error");
})();
