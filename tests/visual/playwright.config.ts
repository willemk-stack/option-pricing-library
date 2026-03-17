import { existsSync } from "node:fs";
import { resolve } from "node:path";

import { defineConfig, devices } from "@playwright/test";
import { widths } from "./targets";

function normalizeBaseURL(rawUrl: string): string {
  return rawUrl.endsWith("/") ? rawUrl : `${rawUrl}/`;
}

const baseURL = normalizeBaseURL(
  process.env.DOCS_BASE_URL || "http://127.0.0.1:8000/option-pricing-library/"
);

function resolvePythonCommand(): string {
  const explicitCommand = process.env.PYTHON || process.env.PYTHON_EXECUTABLE;
  if (explicitCommand) {
    return JSON.stringify(explicitCommand);
  }

  const windowsVenvPython = resolve(__dirname, "../../.venv/Scripts/python.exe");
  if (existsSync(windowsVenvPython)) {
    return JSON.stringify(windowsVenvPython);
  }

  const posixVenvPython = resolve(__dirname, "../../.venv/bin/python");
  if (existsSync(posixVenvPython)) {
    return JSON.stringify(posixVenvPython);
  }

  return "python";
}

const pythonCommand = resolvePythonCommand();

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
    command: `${pythonCommand} ../../scripts/serve_docs.py`,
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
