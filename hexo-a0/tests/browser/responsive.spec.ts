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
  await page.getByLabel("Paste HTTTX").fill(HTTTX);
  await page.getByRole("button", { name: "Load game" }).click();
  await expect(page.locator("#analysis-info")).toContainText("not analyzed");
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
    await expect(page.locator("#analysis-eval-bar")).toBeHidden();
    await expect(page.locator("#analysis-board polygon.hex")).not.toHaveCount(0);
    expect(inferenceRequests).toEqual([]);

    const board = await page.locator("#analysis-board-container").boundingBox();
    expect(board).not.toBeNull();
    expect(board!.width).toBeGreaterThanOrEqual(200);
    expect(board!.height).toBeGreaterThanOrEqual(200);
    await expectNoHorizontalOverflow(page);
  });

  test("primary UI stays inside each configured viewport", async ({ page }) => {
    await page.goto("/analysis");
    await openMobileControls(page);

    const viewport = page.viewportSize()!;
    for (const selector of ["#topbar", "#analysis-controls", "#analysis-board-container"]) {
      const box = await page.locator(selector).boundingBox();
      expect(box, `${selector} should have layout`).not.toBeNull();
      expect(box!.x, `${selector} should not escape left`).toBeGreaterThanOrEqual(-1);
      expect(box!.x + box!.width, `${selector} should not escape right`).toBeLessThanOrEqual(viewport.width + 1);
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
    await page.getByText("Forced-win proof lab", { exact: true }).click();
    await page.getByText("Display options", { exact: true }).click();
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
    await expect(page.getByRole("button", { name: "Start", exact: true })).toBeVisible();

    const viewport = page.viewportSize()!;
    const modal = await page.locator("#modal").boundingBox();
    expect(modal).not.toBeNull();
    expect(modal!.x).toBeGreaterThanOrEqual(-1);
    expect(modal!.y).toBeGreaterThanOrEqual(-1);
    expect(modal!.x + modal!.width).toBeLessThanOrEqual(viewport.width + 1);
    expect(modal!.y + modal!.height).toBeLessThanOrEqual(viewport.height + 1);
    await expectNoHorizontalOverflow(page);
  });
});

function windowWidth(page: Page) {
  return page.viewportSize()?.width ?? 0;
}
