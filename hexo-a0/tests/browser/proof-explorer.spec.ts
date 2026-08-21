import {expect, test, type Page} from "@playwright/test";

const PROOF_BUNDLE = {
  format: "hexo-pdspn-proof-bundle-v1",
  position: {
    stones: [[-4, 0, "P1"]],
    attacker: "P1",
    placements_remaining: 2,
    config: {win_length: 6, placement_radius: 8, max_moves: 400},
  },
  engine: "pdspn",
  width: "wide",
  verification: {dagNodes: 7, proofEdges: 7, maxAttackerTurns: 2},
  certificate: {
    version: 1,
    width: "wide",
    root: 6,
    nodes: [
      {kind: "unstoppable", threats: [[[0, -1]], [[0, 0]], [[0, 1]]]},
      {kind: "immediate_win", action: [[4, 0], [5, 0]]},
      {kind: "immediate_win", action: [[4, 1], [5, 1]]},
      {
        kind: "attacker_move", action: [[2, 0], [3, 0]], child: 0,
        alternatives: [{action: [[2, 1], [3, 1]], child: 0}],
      },
      {kind: "attacker_move", action: [[2, -1], [3, -1]], child: 1},
      {
        kind: "defender_replies",
        responses: [
          {action: [[0, 1], [1, 1]], child: 3},
          {action: [[0, -1], [1, -1]], child: 4},
        ],
      },
      {
        kind: "attacker_move", action: [[-2, 0], [-1, 0]], child: 0,
        alternatives: [{action: [[-2, 1], [-1, 1]], child: 5}],
      },
    ],
  },
};

async function openProof(page: Page) {
  await page.goto("/analysis");
  await page.evaluate(bundle => {
    (window as typeof window & {openProofExplorerBundle(value: unknown): void})
      .openProofExplorerBundle(bundle);
  }, PROOF_BUNDLE);
  await expect(page.locator("#proof-explorer")).toBeVisible();
}

// Bundle that includes a pdspn-shortest `optimization.sampleLine` so the
// proof explorer's "Show winning line" toggle is enabled. Cells are chosen
// from PROOF_BUNDLE so the board geometry stays compatible.
const SAMPLE_LINE_BUNDLE = {
  ...PROOF_BUNDLE,
  engine: "pdspn-shortest",
  optimization: {
    method: "pdspn-shortest-v1",
    shortestCertified: true,
    bestUpperDepth: 2,
    excludedThroughDepth: 1,
    thresholdProbes: 3,
    sampleLine: [
      {turn: 0, player: "P1", cells: [[2, 0], [3, 0]]},
      {turn: 1, player: "P2", cells: [[0, 1], [1, 1]]},
      {turn: 2, player: "P1", cells: [[4, 0], [5, 0]]},
    ],
  },
};

async function openProofWithSampleLine(page: Page) {
  await page.goto("/analysis");
  await page.evaluate(bundle => {
    (window as typeof window & {openProofExplorerBundle(value: unknown): void})
      .openProofExplorerBundle(bundle);
  }, SAMPLE_LINE_BUNDLE);
  await expect(page.locator("#proof-explorer")).toBeVisible();
}

async function expectSelectedNodeCentered(page: Page) {
  const offset = await page.locator("#proof-tree").evaluate(tree => {
    const selected = tree.querySelector<HTMLElement>("[data-proof-selected]");
    if (!selected) throw new Error("no selected proof node");
    const treeRect = tree.getBoundingClientRect();
    const selectedRect = selected.getBoundingClientRect();
    return Math.abs(
      selectedRect.top + selectedRect.height / 2 - (treeRect.top + treeRect.height / 2),
    );
  });
  expect(offset).toBeLessThanOrEqual(4);
}

test.describe("proof explorer", () => {
  test("previews only the hovered or focused branch on the board", async ({page}, testInfo) => {
    test.skip(testInfo.project.name !== "desktop-chromium", "desktop covers hover and keyboard preview");
    await openProof(page);
    const previewHexes = page.locator("#proof-board .proof-preview-hex");
    const previewDots = page.locator("#proof-board .proof-preview-dot");
    const boardLabels = page.locator("#proof-board text");
    const choices = page.locator("#proof-tree .proof-tree-children .proof-tree-node");

    await expect(previewHexes).toHaveCount(0);
    await expect(previewDots).toHaveCount(0);
    await expect(boardLabels).toHaveCount(0);
    await choices.nth(1).hover();
    await expect(previewHexes).toHaveCount(2);
    await expect(previewDots).toHaveCount(2);
    await expect(boardLabels).toHaveCount(0);

    await page.locator("#proof-step-card").hover();
    await expect(previewHexes).toHaveCount(0);
    await expect(previewDots).toHaveCount(0);

    await choices.first().focus();
    await expect(previewHexes).toHaveCount(2);
    await expect(previewDots).toHaveCount(2);
    await expect(boardLabels).toHaveCount(0);
    await page.locator("#proof-close-btn").focus();
    await expect(previewHexes).toHaveCount(0);
    await expect(previewDots).toHaveCount(0);
  });

  test("uses a centered focused tree with local sibling branches", async ({page}) => {
    await openProof(page);

    await expect(page.getByRole("button", {name: /Other winning moves/i})).toHaveCount(0);
    await expect(page.locator("#proof-tree [data-proof-selected]")).toContainText("Starting position");
    const markerShape = await page.locator("#proof-tree [data-proof-selected] .proof-tree-marker")
      .evaluate(element => getComputedStyle(element).clipPath);
    expect(markerShape).toContain("polygon");
    await expect(page.locator("#proof-tree .proof-tree-children .proof-tree-node")).toHaveCount(2);
    await expectSelectedNodeCentered(page);

    await page.locator("#proof-tree .proof-tree-children .proof-tree-node").nth(1).click();
    await expect(page.locator("#proof-tree [data-proof-selected]")).toContainText("Winning turn 1 · move 2");
    await expect(page.locator("#proof-tree .proof-tree-children .proof-tree-node")).toHaveCount(2);
    await expect(page.locator("#proof-tree .proof-tree-siblings").first()).toContainText("1 other branch here");
    await expectSelectedNodeCentered(page);

    await page.locator("#proof-tree .proof-tree-children .proof-tree-node").nth(1).click();
    await expect(page.locator("#proof-tree [data-proof-selected]")).toContainText("Reply B");
    await expect(page.locator("#proof-tree .proof-tree-siblings")).toHaveCount(2);
    await expectSelectedNodeCentered(page);

    await page.locator("#proof-tree .proof-tree-siblings").first().locator("summary").click();
    await page.locator("#proof-tree .proof-tree-siblings").first().locator("button").click();
    await expect(page.locator("#proof-tree [data-proof-selected]")).toContainText("Winning turn 1 · move 1");
    await expect(page.locator("#proof-tree .proof-tree-children")).toHaveCount(0);
    await expectSelectedNodeCentered(page);
  });

  test("shows the checked winning threats at an unstoppable leaf", async ({page}) => {
    await openProof(page);
    await page.locator("#proof-tree .proof-tree-children .proof-tree-node").first().click();

    await expect(page.locator("#proof-step-card")).toContainText("3 winning threats");
    await expect(page.locator("#proof-step-card")).toContainText(
      "Stopping all of them needs at least three placements",
    );
    await expect(page.locator("#proof-board .proof-terminal-threat")).toHaveCount(3);
    await expect(page.locator("#proof-board .proof-preview-dot")).toHaveCount(3);
  });

  test("floats over a full-size board at every viewport", async ({page}) => {
    await openProof(page);
    const viewport = page.viewportSize()!;
    const board = await page.locator("#proof-board-container").boundingBox();
    const panel = await page.locator(".proof-explorer-panel").boundingBox();
    const actions = await page.locator(".proof-explorer-actions").boundingBox();
    expect(board).not.toBeNull();
    expect(panel).not.toBeNull();
    expect(actions).not.toBeNull();
    expect(board!.x).toBe(0);
    expect(board!.y).toBe(0);
    expect(board!.width).toBe(viewport.width);
    expect(board!.height).toBe(viewport.height);
    expect(panel!.x).toBeGreaterThanOrEqual(7);
    expect(panel!.x + panel!.width).toBeLessThanOrEqual(viewport.width - 7);
    expect(actions!.x + actions!.width).toBeLessThanOrEqual(viewport.width - 7);
    await expect(page.locator("#proof-worst-btn")).toBeVisible();

    if (viewport.width > 768) {
      expect(panel!.x).toBe(16);
      expect(panel!.y).toBe(16);
      expect(panel!.height).toBe(viewport.height - 32);
    } else {
      expect(panel!.y + panel!.height).toBe(viewport.height - 8);
      expect(panel!.width).toBe(viewport.width - 16);
    }

    const overflow = await page.evaluate(() => ({
      root: document.documentElement.scrollWidth,
      body: document.body.scrollWidth,
      viewport: window.innerWidth,
    }));
    expect(overflow.root).toBeLessThanOrEqual(overflow.viewport + 1);
    expect(overflow.body).toBeLessThanOrEqual(overflow.viewport + 1);
  });

  test("copy-result-link falls back to execCommand when clipboard.writeText rejects", async ({page}) => {
    // The async Clipboard API rejects on the live site whenever the user has
    // blocked the site's clipboard permission, the document is not focused, or
    // the call loses its user-activation window. The proof share button must
    // still copy the link via the legacy textarea + execCommand fallback in
    // that case; otherwise the user sees "Share failed: …" and nothing lands
    // in the clipboard.
    await page.addInitScript(() => {
      Object.defineProperty(navigator, "clipboard", {
        configurable: true,
        get: () => ({
          writeText: () => Promise.reject(
            new DOMException("Write permission denied.", "NotAllowedError"),
          ),
        }),
      });
      const w = window as unknown as {__execCommandCalls: string[]};
      w.__execCommandCalls = [];
      const original = document.execCommand.bind(document);
      document.execCommand = (command: string): boolean => {
        w.__execCommandCalls.push(command);
        return original(command);
      };
    });
    await openProof(page);
    await page.locator("#proof-share-btn").click();
    const status = page.locator("#proof-share-status");
    await expect(status).toHaveAttribute("data-state", "ok", {timeout: 10_000});
    await expect(status).toHaveText("Link copied");
    const execCommandCalls = await page.evaluate(
      () => (window as unknown as {__execCommandCalls: string[]}).__execCommandCalls,
    );
    expect(execCommandCalls).toContain("copy");
  });

  test("copy-result-link still works when clipboard.writeText succeeds", async ({page, context}) => {
    // The happy path: when the modern Clipboard API succeeds, the legacy
    // textarea fallback should not be used and the status should report
    // success. Headless chromium refuses navigator.clipboard.writeText
    // without an explicit grant, so we grant clipboard-write for the
    // origin and stub the call to record its invocation.
    await context.grantPermissions(
      ["clipboard-read", "clipboard-write"],
      {origin: "http://127.0.0.1:8766"},
    );
    await page.addInitScript(() => {
      const w = window as unknown as {__modernCalls: string[]; __execCommandCalls: string[]};
      w.__modernCalls = [];
      w.__execCommandCalls = [];
      const stub = {
        writeText: (text: string) => {
          w.__modernCalls.push(text);
          return Promise.resolve();
        },
      };
      Object.defineProperty(navigator, "clipboard", {configurable: true, get: () => stub});
      const original = document.execCommand.bind(document);
      document.execCommand = (command: string): boolean => {
        w.__execCommandCalls.push(command);
        return original(command);
      };
    });
    await openProof(page);
    await page.locator("#proof-share-btn").click();
    const status = page.locator("#proof-share-status");
    await expect(status).toHaveAttribute("data-state", "ok", {timeout: 10_000});
    await expect(status).toHaveText("Link copied");
    const [modernCalls, execCommandCalls] = await page.evaluate(() => {
      const w = window as unknown as {__modernCalls: string[]; __execCommandCalls: string[]};
      return [w.__modernCalls, w.__execCommandCalls];
    });
    expect(modernCalls.length).toBeGreaterThan(0);
    expect(execCommandCalls).not.toContain("copy");
  });

  test("show-winning-line toggle draws and hides the sample line on the board", async ({page}) => {
    // Bundle with a sample line enables the toggle; without one it is
    // disabled (plain PDS-PN produces a cert but no sample line).
    await openProofWithSampleLine(page);
    const toggle = page.locator("#proof-show-line");
    await expect(toggle).toBeEnabled();
    await expect(toggle).toBeChecked();

    // Every cell of the sample line renders as a numbered forcing-pv marker.
    const initialMarkers = await page.locator("#proof-board .proof-sample-line .forcing-pv").count();
    expect(initialMarkers).toBeGreaterThan(0);

    // Flipping the toggle off removes the markers; flipping it back restores them.
    await toggle.uncheck();
    await expect(page.locator("#proof-board .proof-sample-line")).toHaveCount(0);
    await toggle.check();
    await expect(page.locator("#proof-board .proof-sample-line .forcing-pv")).toHaveCount(initialMarkers);
  });

  test("show-winning-line toggle is disabled when no sample line was saved", async ({page}) => {
    // PROOF_BUNDLE has no optimization.sampleLine, so the toggle should be off.
    await openProof(page);
    const toggle = page.locator("#proof-show-line");
    await expect(toggle).toBeDisabled();
    await expect(toggle).not.toBeChecked();
    await expect(page.locator("#proof-board .proof-sample-line")).toHaveCount(0);
  });
});


test.describe("proof lab search method", () => {
  test("defaults to staged shortest PDS-PN and retires PNS and DFPN", async ({page}) => {
    await page.goto("/analysis");
    const methods = page.locator("#analysis-forcing-engine");
    await expect(methods).toHaveValue("pdspn-shortest");
    await expect(methods.locator("option")).toHaveCount(3);
    await expect(methods).not.toContainText("PNS");
    await expect(methods).not.toContainText("DFPN");
    await expect(page.locator("#analysis-forcing-budget, #analysis-forcing-leaf-budget")).toHaveCount(0);
    const effort = page.locator("#analysis-forcing-effort");
    await expect(effort).toHaveValue("1");
    await expect(page.locator("#analysis-forcing-effort-label")).toHaveText("Standard");
    await effort.evaluate((element: HTMLInputElement) => {
      element.value = "3";
      element.dispatchEvent(new Event("input", {bubbles: true}));
    });
    await expect(page.locator("#analysis-forcing-effort-label")).toHaveText("Deep");
    await expect(page.locator("#analysis-forcing-effort-hint")).toContainText("keep your device busy");
  });
});
