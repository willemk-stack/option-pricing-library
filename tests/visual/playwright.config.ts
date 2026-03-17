import { defineConfig, devices } from "@playwright/test";
import { widths } from "./targets";

function normalizeBaseURL(rawUrl: string): string {
  return rawUrl.endsWith("/") ? rawUrl : `${rawUrl}/`;
}

const baseURL = normalizeBaseURL(
  process.env.DOCS_BASE_URL || "http://127.0.0.1:8000/option-pricing-library/"
);

export default defineConfig({
  testDir: ".",
  timeout: 60_000,
  snapshotPathTemplate: "{testDir}/{testFilePath}-snapshots/{arg}{ext}",
  expect: {
    timeout: 10_000,
  },
  use: {
    baseURL,
    trace: "on-first-retry",
  },
  webServer: {
    command: "powershell -NoProfile -ExecutionPolicy Bypass -File ..\\..\\scripts\\serve-docs.ps1",
    url: baseURL,
    reuseExistingServer: true,
    timeout: 180_000,
  },
  projects: widths.map((width) => ({
    name: `chromium-${width}`,
    use: {
      ...devices["Desktop Chrome"],
      viewport: {
        width,
        height: 1400,
      },
    },
  })),
});
