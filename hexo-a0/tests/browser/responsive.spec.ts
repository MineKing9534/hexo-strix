import { expect, test, type Page } from "@playwright/test";

const HTTTX = `version[1];
1. [1,0][2,0];
2. [2,1][3,1];
3. [3,2][4,2];`;

async function expectNoHorizontalOverflow(page: Page) {
  const dimensions = await page.evaluate(() => ({
    viewport: window.innerWidth,
    root: document.documentElement.scrollWidth,
    body: document.body.scrollWidth,
  }));
  expect(dimensions.root, "document should fit the viewport").toBeLessThanOrEqual(dimensions.viewport + 1);
  expect(dimensions.body, "body should fit the viewport").toBeLessThanOrEqual(dimensions.viewport + 1);
}

async function openMobileControls(page: Page) {
  const viewport = page.viewportSize();
  if (!viewport || viewport.width > 768) return;
  const controls = page.locator("#analysis-controls");
  if (!(await controls.evaluate(el => el.classList.contains("sheet-open"))))
    await page.locator("#analysis-sheet-handle").click();
  await expect(controls).toHaveClass(/sheet-open/);
}

async function loadReplay(page: Page) {
  await openMobileControls(page);
  await page.getByLabel("Paste a game record (HTTTX)").fill(HTTTX);
  await page.getByRole("button", { name: "Load game" }).click();
  await expect(page.locator("#analysis-info")).toContainText(/not analyzed/i);
}

test.describe("responsive Observatory", () => {
  test("analysis loads and replays without inference", async ({ page }) => {
    const inferenceRequests: string[] = [];
    page.on("request", request => {
      const path = new URL(request.url()).pathname;
      if (path === "/model.safetensors" || path.startsWith("/analyze"))
        inferenceRequests.push(path);
    });

    await page.goto("/analysis");
    await loadReplay(page);

    await expect(page.getByRole("button", { name: "Analyze position" })).toBeEnabled();
    await expect(page.getByRole("button", { name: "Analyze full game" })).toBeEnabled();
    await expect(page.locator("#analysis-eval-bar")).toBeVisible();
    await openMobileControls(page);
    const timeline = page.locator("#analysis-eval-bar");
    const viewport = page.viewportSize()!;
    if (viewport.width > 1100) {
      const transport = await page.locator("#analysis-position-browser").boundingBox();
      expect(transport).not.toBeNull();
      expect(Math.abs(transport!.x + transport!.width / 2 - viewport.width / 2)).toBeLessThanOrEqual(1);
    }
    const fullGameHash = await page.evaluate(() => location.hash);
    await timeline.click({ position: { x: 5, y: 24 } });
    await expect(page.locator("#analysis-info .ro-pos")).toHaveText("1/7");
    expect(await page.evaluate(() => location.hash)).toBe(fullGameHash);
    await timeline.evaluate(element => {
      const rect = element.getBoundingClientRect();
      const emit = (type: string, clientX: number, buttons: number) => element.dispatchEvent(new PointerEvent(type, {
        bubbles: true, pointerId: 7, pointerType: "touch", button: 0, buttons,
        clientX, clientY: rect.top + rect.height / 2,
      }));
      emit("pointerdown", rect.left + 5, 1);
      emit("pointermove", rect.right - 5, 1);
      emit("pointerup", rect.right - 5, 0);
    });
    await expect(page.locator("#analysis-info .ro-pos")).toHaveText("7/7");
    expect(await page.evaluate(() => location.hash)).toBe(fullGameHash);
    await expect(page.locator("#analysis-setup")).toBeHidden();
    await expect(page.locator("#analysis-source-summary")).toContainText("7 positions");
    await page.getByRole("button", { name: "Change" }).click();
    await expect(page.locator("#analysis-setup")).toBeVisible();
    await page.getByRole("button", { name: "Cancel" }).click();
    await expect(page.locator("#analysis-setup")).toBeHidden();
    await expect(page.locator("#analysis-board polygon.hex")).not.toHaveCount(0);
    expect(inferenceRequests).toEqual([]);

    const board = await page.locator("#analysis-board-container").boundingBox();
    expect(board).not.toBeNull();
    expect(board!.width).toBeGreaterThanOrEqual(200);
    expect(board!.height).toBeGreaterThanOrEqual(200);
    await expectNoHorizontalOverflow(page);
  });

  test("full-game analysis rates every completed turn", async ({ page }) => {
    await page.goto("/analysis");
    await loadReplay(page);
    await page.evaluate(() => {
      const app = window as typeof window & Record<string, any>;
      const nodes = window.eval("analysisMain");
      const trajectory = nodes.map((node: Record<string, any>, i: number) => {
        const result = {...node.result, analyzed: true};
        if (i + 1 < nodes.length) {
          result.legal = [nodes[i + 1].move];
          result.policy = [1];
          result.improved_policy = [1];
          result.q_hat = [0.5];
          result.candidate_set = [true];
        }
        return result;
      });
      app.localInference = async (type: string) => {
        if (type !== "analyzeGame") throw new Error(`unexpected ${type}`);
        return {trajectory, boundary_indices: []};
      };
    });

    await page.getByRole("button", { name: "Analyze full game" }).evaluate(
      element => (element as HTMLButtonElement).click(),
    );
    await expect(page.getByRole("button", { name: "Analyze full game" })).toBeEnabled();
    const verdicts = await page.evaluate(() => window.eval(
      "analysisMain.filter(node => isTurnEnd(node)).map(node => node.result.quality?.label || null)",
    ));
    expect(verdicts).toEqual(["best", "best", "best"]);
    await expect(page.locator("#analysis-progress")).toBeHidden();
  });

  test("a second full-game analysis reuses IndexedDB without loading the model", async ({ page }) => {
    test.skip(test.info().project.name !== "desktop-chromium", "one browser covers IndexedDB semantics");
    const modelRequests: string[] = [];
    await page.route("**/model.safetensors*", route => route.fulfill({
      status: 200,
      contentType: "application/octet-stream",
      path: "exports/strixbot-rel2.safetensors",
    }));
    page.on("request", request => {
      if (new URL(request.url()).pathname === "/model.safetensors") modelRequests.push(request.url());
    });
    await page.goto("/analysis");
    await loadReplay(page);
    await page.getByText("Settings", {exact: true}).click();
    await page.locator("#analysis-strength").selectOption("network");
    await page.locator("#analysis-auto-forcing").uncheck();
    await page.evaluate(async () => {
      const app = window as typeof window & Record<string, any>;
      await new Promise<void>((resolve, reject) => {
        const request = indexedDB.deleteDatabase("hexo-local-analysis");
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
      });
      app.analysisProgressMessages = [];
      const original = app.showProgress;
      app.showProgress = (on: boolean, message: string, ...rest: unknown[]) => {
        if (on) app.analysisProgressMessages.push(message);
        return original(on, message, ...rest);
      };
      await app.analyzeWholeGame();
    });
    expect(modelRequests).toHaveLength(1);
    const firstState = await page.evaluate(() => ({
      messages: (window as typeof window & Record<string, any>).analysisProgressMessages,
      ratings: window.eval("analysisMain.filter(node => isTurnEnd(node) && node.result.quality).length"),
    }));
    expect(firstState.ratings).toBe(3);
    expect(firstState.messages.some((message: string) => message.startsWith("Analyzing and rating"))).toBe(true);
    expect(firstState.messages.some((message: string) => message.startsWith("Rating completed"))).toBe(false);

    const secondMessages = await page.evaluate(async () => {
      const app = window as typeof window & Record<string, any>;
      app.analysisProgressMessages.length = 0;
      await app.analyzeWholeGame();
      return app.analysisProgressMessages;
    });
    expect(modelRequests).toHaveLength(1);
    expect(secondMessages).toContain("Loading saved analysis…");
    expect(secondMessages.some((message: string) => message.startsWith("Analyzing and rating"))).toBe(false);
    expect(secondMessages.some((message: string) => message.startsWith("Rating completed"))).toBe(false);
  });

  test("position analysis returns its turn rating in the same operation", async ({ page }) => {
    await page.goto("/analysis");
    await loadReplay(page);
    await page.evaluate(() => {
      const app = window as typeof window & Record<string, any>;
      const nodes = window.eval("analysisMain");
      const node = nodes[nodes.length - 1];
      const start = nodes[node.depth - 2];
      const afterFirst = nodes[node.depth - 1];
      start.result = {
        ...start.result, analyzed: true, legal: [afterFirst.move],
        policy: [1], improved_policy: [1], q_hat: [0.5], candidate_set: [true],
      };
      afterFirst.result = {
        ...afterFirst.result, analyzed: true, legal: [node.move],
        policy: [1], improved_policy: [1], q_hat: [0.5], candidate_set: [true],
      };
      const analyzed = {...node.result, analyzed: true, value: 0.25};
      app.localInference = async (type: string) => {
        if (type !== "analyzePosition") throw new Error(`unexpected ${type}`);
        return analyzed;
      };
    });

    await page.getByRole("button", { name: "Analyze position" }).evaluate(
      element => (element as HTMLButtonElement).click(),
    );
    await expect(page.getByRole("button", { name: "Analyze position" })).toBeEnabled();
    await expect(page.locator("#analysis-info")).toContainText("Best");
    expect(await page.evaluate(() => window.eval("analysisCurrent.result.quality?.label"))).toBe("best");
    await expect(page.locator("#analysis-thinking")).toBeHidden();
  });

  test("primary UI stays inside each configured viewport", async ({ page }) => {
    await page.goto("/analysis");
    await openMobileControls(page);

    await expect(page.locator("#analysis-empty-state")).toBeVisible();
    await expect(page.locator("#analysis-navigation")).toBeHidden();

    const strengthColours = await page.locator("#analysis-strength").evaluate(element => {
      const style = getComputedStyle(element);
      return { background: style.backgroundColor, foreground: style.color };
    });
    expect(strengthColours.background).not.toBe("rgb(255, 255, 255)");
    expect(strengthColours.foreground).not.toBe("rgb(0, 0, 0)");
    await expect(page.locator("#analysis-strength option[value=network]")).toHaveText("Instant · no search");
    await expect(page.locator("#analysis-auto-forcing")).toBeChecked();
    await expect(page.locator("#analysis-auto-branch")).not.toBeChecked();
    await expect(page.locator("#analysis-settings-status")).toHaveText("Standard · auto off");

    const viewport = page.viewportSize()!;
    for (const selector of ["#topbar", "#analysis-controls", "#analysis-board-container"]) {
      const box = await page.locator(selector).boundingBox();
      expect(box, `${selector} should have layout`).not.toBeNull();
      expect(box!.x, `${selector} should not escape left`).toBeGreaterThanOrEqual(-1);
      expect(box!.x + box!.width, `${selector} should not escape right`).toBeLessThanOrEqual(viewport.width + 1);
    }
    await page.evaluate(() => {
      (window as typeof window & { updateGauge(value: number, player: string): void }).updateGauge(0.5, "P1");
    });
    await expect(page.locator("#gauge-wrap")).toBeVisible();
    await expect(page.locator("#gauge-needle")).toHaveAttribute("style", /top: 25/);
    const gauge = await page.locator("#gauge-wrap").boundingBox();
    expect(gauge).not.toBeNull();
    expect(gauge!.x + gauge!.width).toBeLessThanOrEqual(viewport.width + 1);
    if (viewport.width > 768) {
      const board = await page.locator("#analysis-board-container").boundingBox();
      expect(board).not.toBeNull();
      expect(Math.abs(gauge!.y + gauge!.height / 2 - (board!.y + board!.height / 2)))
        .toBeLessThanOrEqual(1);
      expect(gauge!.height).toBeGreaterThanOrEqual(Math.min(300, board!.height - 2));
    }
    await expectNoHorizontalOverflow(page);
  });

  test("mobile controls sheet can reach every section", async ({ page }) => {
    test.skip((page.viewportSize()?.width ?? 9999) > 768, "bottom sheet is the compact layout");
    await page.goto("/analysis");
    await openMobileControls(page);

    const controls = page.locator("#analysis-controls");
    const controlsBody = page.locator("#analysis-controls-body");
    await expect(page.getByRole("button", { name: "Load game" })).toBeVisible();
    await page.getByText("Settings", { exact: true }).click();
    await expect(page.getByText("Analyze new moves automatically", { exact: true })).toBeVisible();
    await expect(page.getByText("Board overlays", { exact: true })).toBeVisible();
    await page.getByText("How to read analysis", { exact: true }).click();
    await controlsBody.evaluate(el => el.scrollTo({ top: el.scrollHeight }));
    await expect(page.locator("#analysis-caveat")).toBeVisible();

    const geometry = await controls.evaluate(el => {
      const rect = el.getBoundingClientRect();
      return { left: rect.left, right: rect.right, bottom: rect.bottom };
    });
    expect(geometry.left).toBeGreaterThanOrEqual(-1);
    expect(geometry.right).toBeLessThanOrEqual(windowWidth(page) + 1);
    expect(geometry.bottom).toBeLessThanOrEqual((page.viewportSize()?.height ?? 0) + 1);
    await expectNoHorizontalOverflow(page);
  });

  test("analysis settings persist and placed hexes can analyze automatically", async ({ page }) => {
    await page.goto("/analysis");
    await openMobileControls(page);
    await page.getByText("Settings", { exact: true }).click();
    await page.locator("#analysis-strength").selectOption("quick");
    await page.locator("#analysis-auto-branch").check();
    await page.locator("#analysis-heatmap").uncheck();
    await expect(page.locator("#analysis-settings-status")).toHaveText("Quick · auto on");

    await page.reload();
    await openMobileControls(page);
    await expect(page.locator("#analysis-strength")).toHaveValue("quick");
    await expect(page.locator("#analysis-auto-branch")).toBeChecked();
    await expect(page.locator("#analysis-heatmap")).not.toBeChecked();
    await expect(page.locator("#analysis-settings-status")).toHaveText("Quick · auto on");

    await loadReplay(page);
    await page.evaluate(() => {
      const app = window as typeof window & Record<string, any>;
      app.analyzePosition = async (moves: number[][], onEstimate?: (result: object) => void) => {
        const result = app.replayEntryAt(moves, moves.length - 1, {
          win_length: 4, placement_radius: 8, max_moves: 120,
        });
        const count = result.legal?.length ?? 0;
        onEstimate?.({value: -0.18, current_player: result.current_player});
        const forcing = {
          winner: result.current_player,
          attacker_is_mover: true,
          first_move: result.legal?.[0] ?? null,
          depth: 1,
          pv: result.legal?.length ? [result.legal[0]] : [],
          pv_owners: result.legal?.length ? [result.current_player] : [],
          line_placements: result.legal?.length ? 1 : 0,
          pv_len: result.legal?.length ? 1 : 0,
          wide: true,
          defense: null,
        };
        onEstimate?.({value: -0.18, current_player: result.current_player, forcing});
        return await new Promise(resolve => {
          app.finishAutomaticAnalysis = () => resolve({
            ...result,
            value: 0.42,
            forcing,
            policy: Array(count).fill(count ? 1 / count : 0),
            improved_policy: Array(count).fill(count ? 1 / count : 0),
            q_hat: Array(count).fill(0.42),
          });
        });
      };
    });
    const move = await page.evaluate(() => {
      return window.eval("analysisCurrent.result.legal[0]") as [number, number];
    });
    await page.evaluate(([q, r]) => {
      (window as typeof window & Record<string, any>).analysisCellClick(q, r);
    }, move);
    await expect(page.locator("#analysis-thinking")).toBeVisible();
    await expect(page.locator("#analysis-thinking-label")).toHaveText("Forced-win check ready · searching moves…");
    await expect(page.locator("#analysis-info")).toContainText("P1 estimate");
    await expect(page.locator("#gauge-wrap")).toBeVisible();
    await expect(page.locator("#forcing-banner")).toContainText("has a forced win");
    await page.evaluate(() => {
      (window as typeof window & Record<string, any>).finishAutomaticAnalysis();
    });
    await expect(page.locator("#analysis-info")).toContainText("+1.00");
    await expect(page.locator("#analysis-info")).toContainText("P1 score");
    await expect(page.locator("#analysis-thinking")).toBeHidden();
    await expect(page.locator("#analysis-progress")).toBeHidden();
  });

  test("short landscape controls remain reachable", async ({ page }) => {
    const viewport = page.viewportSize();
    test.skip(!viewport || viewport.height > 520, "short landscape layout only");
    await page.goto("/analysis");

    const scroller = viewport!.width <= 768
      ? page.locator("#analysis-controls-body")
      : page.locator("#analysis-controls");
    await page.getByText("How to read analysis", { exact: true }).click();
    await scroller.evaluate(el => el.scrollTo({ top: el.scrollHeight }));
    await expect(page.locator("#analysis-caveat")).toBeVisible();
  });

  test("proof lab is a position-scoped board tool", async ({ page }) => {
    await page.goto("/analysis");
    await expect(page.locator("#proof-lab-launch")).toBeVisible();
    await expect(page.locator("#proof-lab-launch")).toBeDisabled();
    await loadReplay(page);
    await expect(page.locator("#proof-lab-launch")).toBeEnabled();

    const viewport = page.viewportSize();
    if (viewport && viewport.width <= 768) {
      await openMobileControls(page);
      await expect(page.locator("#analysis-controls")).toHaveClass(/sheet-open/);
    }

    await expect(page.locator("#analysis-forcing-depth-control")).toBeHidden();
    await page.locator("#proof-lab-launch").click();
    if (viewport && viewport.width <= 768)
      await expect(page.locator("#analysis-controls")).toHaveClass(/sheet-open/);
    await expect(page.locator("#proof-lab-drawer")).toBeVisible();
    await expect(page.locator("#analysis-info")).toBeHidden();
    await expect(page.locator("#analysis-position-browser")).toBeVisible();
    await expect(page.locator("#proof-lab-position")).toContainText(/Position \d+ · P[12] to move/);
    await expect(page.getByRole("button", { name: "Check for a forced win" })).toBeVisible();
    await page.locator("#analysis-mode-analysis").click();
    await expect(page.locator("#proof-lab-drawer")).toBeHidden();
    await expect(page.locator("#analysis-info")).toBeVisible();
  });

  test("an unstoppable opponent win takes precedence over the generic threat", async ({ page }) => {
    await page.goto("/analysis");
    await loadReplay(page);
    await page.evaluate(() => {
      const app = window as typeof window & Record<string, any>;
      const node = window.eval("analysisCurrent");
      const mover = node.result.current_player;
      const winner = mover === "P1" ? "P2" : "P1";
      const first = node.result.legal[0];
      node.result.forcing = {
        winner,
        attacker_is_mover: false,
        first_move: first,
        depth: 3,
        pv: [first],
        pv_owners: [winner],
        line_placements: 1,
        pv_len: 1,
        wide: true,
        defense: {killers: [], pair_anchors: [], best_delay: first, wide: true},
      };
      (document.getElementById("analysis-threats") as HTMLInputElement).checked = true;
      app.rerenderCurrentAnalysis();
    });
    await expect(page.locator("#forcing-banner")).toContainText("has an unstoppable forced win");
    await expect(page.locator("#forcing-banner")).toContainText("can only delay it");
    await expect(page.locator("#forcing-banner")).not.toContainText("threatens a forced win");
  });

  test("a selected whole-game threat lazily gains its defence result", async ({ page }) => {
    await page.goto("/analysis");
    await loadReplay(page);
    await page.evaluate(() => {
      const app = window as typeof window & Record<string, any>;
      const node = window.eval("analysisCurrent");
      const mover = node.result.current_player;
      const winner = mover === "P1" ? "P2" : "P1";
      const first = node.result.legal[0];
      node.result.analyzed = false; // keep this test focused on defence hydration
      node.result.forcing = {
        winner, attacker_is_mover: false, first_move: first, depth: 1,
        pv: [first], pv_owners: [winner], line_placements: 1, pv_len: 1,
        wide: true, defense: null,
      };
      app.localInference = async (type: string, payload: Record<string, any>) => {
        if (type !== "analyzeDefense") throw new Error(`unexpected ${type}`);
        return {
          status: "checked",
          forcing: {
            ...payload.forcing,
            defense_status: "checked",
            defense: {killers: [], pair_anchors: [], best_delay: first, wide: true},
          },
        };
      };
      (document.getElementById("analysis-threats") as HTMLInputElement).checked = true;
      app.setCurrent(node);
    });
    await expect(page.locator("#analysis-thinking-label")).toHaveText("Checking possible defences…");
    await expect(page.locator("#forcing-banner")).toContainText("has an unstoppable forced win");
    await expect(page.locator("#forcing-banner")).toContainText("delays it longest");
    await expect(page.locator("#analysis-board .defense-badge")).not.toHaveCount(0);
  });

  test("a defender in an already lost position gets a forced-position card", async ({ page }) => {
    await page.goto("/analysis");
    await loadReplay(page);
    await page.evaluate(async () => {
      const app = window as typeof window & Record<string, any>;
      const nodes = window.eval("analysisMain");
      const node = nodes[nodes.length - 1];
      const start = nodes[node.depth - 2];
      const afterFirst = nodes[node.depth - 1];
      const first = afterFirst.move;
      const second = node.move;
      const mover = start.result.current_player;
      const winner = mover === "P1" ? "P2" : "P1";
      start.result = {
        ...start.result, analyzed: true, legal: [first], improved_policy: [1],
        q_hat: [-0.99], candidate_set: [true], forcing: {
          winner, attacker_is_mover: false, first_move: first, depth: 3,
          pv: [first], pv_owners: [winner], line_placements: 1, pv_len: 1,
          defense_status: "checked",
          defense: {killers: [], pair_anchors: [], best_delay: first, wide: true},
        },
      };
      afterFirst.result = {
        ...afterFirst.result, analyzed: true, legal: [[9, 9], second], improved_policy: [1, 0],
        q_hat: [-0.99, -1], candidate_set: [true, true],
      };
      node.result = {
        ...node.result, analyzed: true, forcing: {
          winner, attacker_is_mover: true, first_move: first, depth: 3,
          pv: [first], pv_owners: [winner], line_placements: 1, pv_len: 1,
          defense: null,
        },
      };
      node.result.quality = await app.computeTurnQuality(node);
      app.setCurrent(node);
    });
    await expect(page.locator("#analysis-info")).toHaveClass(/vc/);
    await expect(page.locator("#analysis-info")).toContainText("No saving move");
    await expect(page.locator("#analysis-info")).not.toContainText("Blunder");
    await expect(page.locator("#analysis-info")).toContainText("What happened");
    await expect(page.locator("#analysis-info")).toContainText("P1 had a forced win before this turn");
    await expect(page.locator("#analysis-info")).toContainText("No P2 move could stop it");
    await expect(page.locator("#analysis-info")).toContainText("but that line also loses");
    await expect(page.locator("#analysis-info")).not.toContainText("move-quality penalty");
    await expect(page.locator("#analysis-info")).not.toContainText("suggested delay");
  });

  test("keeping a verified forced win is never rated as a blunder", async ({ page }) => {
    await page.goto("/analysis");
    await loadReplay(page);
    await page.evaluate(async () => {
      const app = window as typeof window & Record<string, any>;
      const nodes = window.eval("analysisMain");
      const node = nodes[nodes.length - 1];
      const start = nodes[node.depth - 2];
      const afterFirst = nodes[node.depth - 1];
      const mover = start.result.current_player;
      start.result = {
        ...start.result, analyzed: true, legal: [afterFirst.move],
        improved_policy: [1], q_hat: [0.94], candidate_set: [true],
        forcing: {winner: mover, attacker_is_mover: true, first_move: afterFirst.move},
      };
      afterFirst.result = {
        ...afterFirst.result, analyzed: true, legal: [[-10, 4], node.move],
        improved_policy: [1, 0], q_hat: [0.94, 0.35], candidate_set: [true, true],
      };
      node.result = {
        ...node.result, analyzed: true, forcing: {
          winner: mover, attacker_is_mover: false, first_move: node.move,
          defense: null,
        },
      };
      app.localInference = async (type: string, payload: Record<string, any>) => {
        if (type !== "analyzeDefense") throw new Error(`unexpected ${type}`);
        return {status: "checked", forcing: {
          ...payload.forcing, defense_status: "checked",
          defense: {killers: [], pair_anchors: [], best_delay: [-8, 0]},
        }};
      };
      node.result.quality = await app.computeTurnQuality(node);
      app.setCurrent(node);
    });
    await expect(page.locator("#analysis-info")).toContainText("Winning move");
    await expect(page.locator("#analysis-info")).toContainText("kept a verified forced win");
    await expect(page.locator("#analysis-info")).not.toContainText("Blunder");
    await expect(page.locator("#analysis-info")).not.toContainText("estimated scores");
    expect(await page.evaluate(() => window.eval("analysisCurrent.result.quality.label"))).toBe("winning");
  });

  test("missed wins open a concise explanation and the winning board line", async ({ page }) => {
    await page.goto("/analysis");
    await loadReplay(page);
    await page.evaluate(() => {
      const app = window as typeof window & Record<string, any>;
      const nodes = window.eval("analysisMain");
      const target = nodes[2];
      const missed = nodes[3];
      const first = [5, 3];
      missed.result.missed_win = {
        by: target.result.current_player, at_prefix: 2, first_move: first,
        depth: 2, pv: [first], pv_owners: [target.result.current_player],
        line_placements: 1, pv_len: 1,
      };
      app.renderMoveTree();
    });
    await page.locator(".missed-win-badge").evaluate(element => (element as HTMLElement).click());
    await expect(page.locator("#missed-win-callout")).toContainText("Forced win missed");
    await expect(page.locator("#missed-win-callout")).toContainText("The selected move gave it up");
    await expect(page.locator("#forcing-banner")).toContainText("has a forced win");
    await expect(page.locator("#analysis-board .forcing-pv")).not.toHaveCount(0);
  });

  test("sandbox import is a focused dialog", async ({ page }) => {
    await page.goto("/analysis");
    await openMobileControls(page);
    await expect(page.locator("#hds-import")).toHaveCount(0);

    await page.getByRole("button", { name: "Import from Hexo sandbox" }).click();
    const dialog = page.locator("#hds-import-dialog");
    await expect(dialog).toBeVisible();
    await expect(page.getByLabel("Sandbox link or code")).toBeFocused();
    await expect(page.getByRole("button", { name: "Import position" })).toBeVisible();

    const viewport = page.viewportSize()!;
    const box = await dialog.boundingBox();
    expect(box).not.toBeNull();
    expect(box!.x).toBeGreaterThanOrEqual(-1);
    expect(box!.x + box!.width).toBeLessThanOrEqual(viewport.width + 1);
    expect(box!.y).toBeGreaterThanOrEqual(-1);
    expect(box!.y + box!.height).toBeLessThanOrEqual(viewport.height + 1);
    await page.keyboard.press("Escape");
    await expect(dialog).toBeHidden();
  });

  test("sandbox import loads the converted position", async ({ page }) => {
    await page.goto("/analysis");
    await openMobileControls(page);
    await page.route("**/convert_hds", route => route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({htttx: HTTTX, move_count: 6}),
    }));
    await page.getByRole("button", { name: "Import from Hexo sandbox" }).click();
    const dialog = page.locator("#hds-import-dialog");
    await page.getByLabel("Sandbox link or code").fill("5knldz6");
    await page.getByRole("button", { name: "Import position" }).click();
    await expect(dialog).toBeHidden();
    await expect(page.locator("#analysis-info")).toContainText(/not analyzed/i);
  });

  test("touch drag pans analysis without playing a move", async ({ page }) => {
    test.skip(!test.info().project.use.hasTouch, "touch project only");
    await page.goto("/analysis");
    await loadReplay(page);

    const beforePosition = await page.locator("#analysis-info").textContent();
    const beforeTransform = await page.locator("#analysis-board-group").getAttribute("transform");
    await page.locator("#analysis-board").evaluate(svg => {
      const point = (x: number, y: number) => ({
        identifier: 1, target: svg, clientX: x, clientY: y,
        pageX: x, pageY: y, screenX: x, screenY: y,
      });
      const dispatch = (type: string, touches: object[]) => {
        const event = new Event(type, { bubbles: true, cancelable: true });
        Object.defineProperty(event, "touches", { value: touches });
        svg.dispatchEvent(event);
      };
      dispatch("touchstart", [point(150, 180)]);
      dispatch("touchmove", [point(205, 220)]);
      dispatch("touchend", []);
    });

    await expect.poll(() => page.locator("#analysis-board-group").getAttribute("transform"))
      .not.toBe(beforeTransform);
    await expect(page.locator("#analysis-info")).toHaveText(beforePosition ?? "");
  });

  test("new-game dialog fits and remains actionable", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#modal-bg")).toHaveClass(/show/);
    await expect(page.getByRole("button", { name: "Start game", exact: true })).toBeVisible();
    await expect(page.locator("#diff-row .diff-btn")).toHaveCount(4);
    await expect(page.locator("#diff-row .diff-btn.selected")).toHaveCount(1);
    await expect(page.locator("#diff-label")).toHaveText("Search effort");
    await expect(page.locator("#diff-row .diff-btn")).toHaveText([
      /QuickResponds fastest/,
      /StandardBalances speed and search/,
      /StrongSearches further/,
      /DeepSearches furthest/,
    ]);

    const viewport = page.viewportSize()!;
    const modal = await page.locator("#modal").boundingBox();
    expect(modal).not.toBeNull();
    expect(modal!.x).toBeGreaterThanOrEqual(-1);
    expect(modal!.y).toBeGreaterThanOrEqual(-1);
    expect(modal!.x + modal!.width).toBeLessThanOrEqual(viewport.width + 1);
    expect(modal!.y + modal!.height).toBeLessThanOrEqual(viewport.height + 1);
    const start = await page.getByRole("button", { name: "Start game", exact: true }).boundingBox();
    expect(start).not.toBeNull();
    expect(start!.y + start!.height).toBeLessThanOrEqual(viewport.height + 1);
    await expectNoHorizontalOverflow(page);
  });
});

function windowWidth(page: Page) {
  return page.viewportSize()?.width ?? 0;
}
