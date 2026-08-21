import {expect, test} from "@playwright/test";

const PREVIEW = "/static/move-list-preview.html";

test("isolated move-list preview renders all important states", async ({page}) => {
  await page.goto(PREVIEW);
  const list = page.locator("hexo-move-list");
  await expect(list).toBeVisible();
  await expect(list.locator(".round-number")).toHaveCount(11);
  await expect(list.locator(".missed")).toHaveCount(5);
  await expect(list.locator(".coordinate.active")).toHaveCount(1);
  await expect(list.locator(".placement.empty")).toHaveCount(3);
  await expect(list.locator(".missed").first()).toHaveAccessibleName(/Win missed/);
});

test("coordinates, statuses, and player columns never overlap", async ({page}) => {
  await page.goto(PREVIEW);
  const layout = await page.locator("hexo-move-list").evaluate((host: HTMLElement & {shadowRoot: ShadowRoot}) => {
    const root = host.shadowRoot;
    const placements = [...root.querySelectorAll<HTMLElement>(".placement:not(.empty)")];
    const turns = [...root.querySelectorAll<HTMLElement>(".turn")];
    const coordinates = [...root.querySelectorAll<HTMLElement>(".coordinate")];
    const missed = [...root.querySelectorAll<HTMLElement>(".missed")];
    const overlaps = placements.filter(placement => {
      const coordinate = placement.querySelector<HTMLElement>(".coordinate")!.getBoundingClientRect();
      const turn = placement.closest<HTMLElement>(".turn")!;
      const status = turn.querySelector<HTMLElement>(".turn-status")!.getBoundingClientRect();
      const box = placement.getBoundingClientRect();
      return coordinate.right > status.left + 0.5 || coordinate.left < box.left - 0.5 || status.right > turn.getBoundingClientRect().right + 0.5;
    }).length;
    const statusOverlaps = [...root.querySelectorAll<HTMLElement>(".turn-status")].filter(status => {
      const children = [...status.children].map(child => child.getBoundingClientRect());
      return children.some((rect, index) => index > 0 && rect.left < children[index - 1].right - 0.5);
    }).length;
    const overflowingTurns = turns.filter(turn => turn.scrollWidth > turn.clientWidth).length;
    const coordinateTemplates = coordinates.map(el => getComputedStyle(el).gridTemplateColumns);
    const iconsContained = missed.every(button => {
      const outer = button.getBoundingClientRect();
      const icon = button.querySelector("svg")!.getBoundingClientRect();
      return icon.left >= outer.left && icon.right <= outer.right && icon.top >= outer.top && icon.bottom <= outer.bottom;
    });
    return {overlaps, statusOverlaps, overflowingTurns, coordinateTemplates, iconsContained};
  });
  expect(layout.overlaps).toBe(0);
  expect(layout.statusOverlaps).toBe(0);
  expect(layout.overflowingTurns).toBe(0);
  expect(new Set(layout.coordinateTemplates).size).toBe(1);
  expect(layout.iconsContained).toBe(true);
});

test("selection and missed-win actions are independent controls", async ({page}) => {
  await page.goto(PREVIEW);
  const list = page.locator("hexo-move-list");
  const target = list.locator('.coordinate[data-move-id="2"]');
  await target.click();
  await expect(target).toHaveClass(/active/);
  await expect(target).toHaveAttribute("aria-current", "step");

  const missed = list.locator(".missed").first();
  await missed.focus();
  await expect(missed).toBeFocused();
  await missed.press("Enter");
  await expect(target).toHaveClass(/active/);
});
