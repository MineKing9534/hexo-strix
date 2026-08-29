import {expect, test} from "@playwright/test";

const REPLAY = "AQQFBAMEAgIJBgAGBAQGBgUIAggICAoKBAoADAwMDg4GDAgOChQMFgoQBhYBAgIBBQYEAwIAAgMCBAIFBAAGAAEACAAGAgYEBggGAQQCCgIMAgACCAQMAAQIDgEKBAoGCggOBAgGDAYEBhAGBgoEDAAQDAQIDAoDCAIKAQwIDgYEEBAECAEMEAwFDA4QDBIKCBQUCA4MEgwKDBQMDgoOEA4IDhIMChAKCAoUChAOEBAIFhASEg4SEA";

test("Analysis renders the isolated move-list component and supports arrow navigation", async ({page}) => {
  await page.goto(`/analysis#c=${REPLAY}`);
  const list = page.locator("#analysis-movetree hexo-move-list");
  await expect(list).toBeVisible();
  await expect(list.locator(".round-number")).toHaveCount(24);

  const activeBefore = await list.locator(".coordinate.active").getAttribute("data-move-id");
  await page.keyboard.press("ArrowLeft");
  const activePrevious = await list.locator(".coordinate.active").getAttribute("data-move-id");
  expect(activePrevious).not.toBe(activeBefore);
  await page.keyboard.press("ArrowRight");
  await expect(list.locator(".coordinate.active")).toHaveAttribute("data-move-id", activeBefore!);

  await page.locator("#analysis-htttx").dispatchEvent("keydown", {key: "ArrowLeft", bubbles: true});
  await expect(list.locator(".coordinate.active")).toHaveAttribute("data-move-id", activeBefore!);
});

test("Copy as HTTTX copies the full loaded game record", async ({page}) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {writeText: async (text: string) => { (window as typeof window & {__copied?: string}).__copied = text; }},
    });
  });
  await page.goto(`/analysis#c=${REPLAY}`);
  const copy = page.getByRole("button", {name: "Copy as HTTTX"});
  await expect(copy).toBeAttached();
  await copy.evaluate((button: HTMLButtonElement) => button.click());
  await expect.poll(() => page.evaluate(() => (window as typeof window & {__copied?: string}).__copied)).toBeTruthy();
  const copied = await page.evaluate(() => (window as typeof window & {__copied?: string}).__copied);
  expect(copied).toContain("version[1];");
  expect(copied).toContain("46.");
});


test("Loaded-game actions stay on one row with settings expanded", async ({page}) => {
  await page.goto(`/analysis#c=${REPLAY}`);
  const settings = page.locator("details.analysis-settings");
  await settings.evaluate((element: HTMLDetailsElement) => { element.open = true; });
  const copy = page.getByRole("button", {name: "Copy as HTTTX"});
  const change = page.getByRole("button", {name: "Change"});
  const [copyBox, changeBox] = await Promise.all([copy.boundingBox(), change.boundingBox()]);
  expect(copyBox).not.toBeNull();
  expect(changeBox).not.toBeNull();
  expect(Math.abs(copyBox!.y - changeBox!.y)).toBeLessThan(2);
  expect(copyBox!.x + copyBox!.width).toBeLessThanOrEqual(changeBox!.x);
});
