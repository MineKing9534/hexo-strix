import {expect, test} from "@playwright/test";

test("loads a compact replay when the hash changes on an open Analysis page", async ({page}) => {
  await page.goto("/analysis");
  await expect(page.locator("#analysis-htttx")).toHaveValue("");

  // [1,0] zig-zag/varint encodes to bytes [2,0] => base64url "AgA".
  await page.evaluate(() => { location.hash = "#c=AgA"; });

  await expect(page.locator("#analysis-htttx")).toHaveValue(/\[1,0\]/);
  await expect(page.locator("#analysis-empty-state")).toBeHidden();
  await expect(page.locator("#analysis-navigation")).toBeVisible();
});

const WRAPPED_REPLAY = "AQQFBAMEAgIJBgAGBAQGBgUIAggICAoKBAoADAwMDg4GDAgOChQMFgoQBhYBAgIBBQYEAwIAAgMCBAIFBAAGAAEACAAGAgYEBggGAQQCCgIMAgACCAQMAAQIDgEKBAoGCggOBAgGDAYEBhAGBgoEDAAQDAQ%20IDAoDCAIKAQwIDgYEEBAECAEMEAwFDA4QDBIKCBQUCA4MEgwKDBQMDgoOEA4IDhIMChAKCAoUChAOEBAIFhASEg4SEA";

test("loads a fresh compact replay URL containing encoded wrapping whitespace", async ({page}) => {
  await page.goto(`/analysis#c=${WRAPPED_REPLAY}`);

  await expect(page.locator("#analysis-empty-state")).toBeHidden();
  await expect(page.locator("#analysis-navigation")).toBeVisible();
  await expect(page.locator("#analysis-source-meta")).toHaveText("93 positions");
});
