import { defineConfig, devices } from "@playwright/test";

const PORT = 8766;
const webkitProjects = process.env.PLAYWRIGHT_WEBKIT === "1" ? [
  {
    name: "phone-webkit",
    use: { ...devices["iPhone 12"], browserName: "webkit" as const },
  },
  {
    name: "tablet-webkit",
    use: { ...devices["iPad Mini"], browserName: "webkit" as const },
  },
] : [];

export default defineConfig({
  testDir: "./hexo-a0/tests/browser",
  outputDir: "./hexo-a0/tests/browser-results",
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"], ["html", { outputFolder: "hexo-a0/tests/browser-report", open: "never" }]],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    colorScheme: "dark",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  webServer: {
    command: `.venv/bin/python hexo-a0/tests/browser_server.py --port ${PORT}`,
    url: `http://127.0.0.1:${PORT}/analysis`,
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
  projects: [
    {
      name: "desktop-chromium",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1280, height: 800 } },
    },
    {
      name: "phone-chromium",
      use: { ...devices["Pixel 5"] },
    },
    {
      name: "phone-landscape",
      use: {
        ...devices["Pixel 5 landscape"],
        viewport: { width: 851, height: 393 },
      },
    },
    {
      name: "tablet-chromium",
      use: { ...devices["iPad Mini"], browserName: "chromium" },
    },
    ...webkitProjects,
  ],
});
