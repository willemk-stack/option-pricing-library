import { defineConfig, devices } from "@playwright/test";

const baseURL =
  process.env.DOCS_BASE_URL ?? "http://127.0.0.1:8000/option-pricing-library";

export default defineConfig({
  testDir: ".",
  timeout: 60_000,
  expect: {
    timeout: 10_000,
    toHaveScreenshot: {
      maxDiffPixelRatio: 0.01
    }
  },
  use: {
    baseURL,
    trace: "on-first-retry",
    screenshot: "only-on-failure"
  },
  projects: [
    {
      name: "chromium-375",
      use: { ...devices["Desktop Chrome"], viewport: { width: 375, height: 1400 } }
    },
    {
      name: "chromium-768",
      use: { ...devices["Desktop Chrome"], viewport: { width: 768, height: 1400 } }
    },
    {
      name: "chromium-1280",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1280, height: 1600 } }
    },
    {
      name: "chromium-1536",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1536, height: 1800 } }
    }
  ]
});