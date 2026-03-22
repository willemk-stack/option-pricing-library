import { existsSync } from "node:fs";
import { resolve } from "node:path";

import { defineConfig, devices } from "@playwright/test";
import { defaultDocsBaseURL, normalizeBaseURL } from "./config";
import { widths } from "./targets";

const baseURL = normalizeBaseURL(process.env.DOCS_BASE_URL || defaultDocsBaseURL);

function resolvePythonCommand(): string {
  const explicitCommand = process.env.PYTHON || process.env.PYTHON_EXECUTABLE;
  if (explicitCommand) {
    return JSON.stringify(explicitCommand);
  }

  const windowsVenvPython = resolve(__dirname, "../../.venv/Scripts/python.exe");
  if (process.platform === "win32" && existsSync(windowsVenvPython)) {
    return JSON.stringify(windowsVenvPython);
  }

  const posixVenvPython = resolve(__dirname, "../../.venv/bin/python");
  if (process.platform !== "win32" && existsSync(posixVenvPython)) {
    return JSON.stringify(posixVenvPython);
  }

  return "python";
}

const pythonCommand = resolvePythonCommand();
const servePrebuiltArg = process.env.SERVE_PREBUILT_SITE === "1" ? " --serve-prebuilt" : "";

export default defineConfig({
  testDir: ".",
  timeout: 60_000,
  retries: process.env.CI ? 1 : 0,
  snapshotPathTemplate: "{testDir}/{testFilePath}-snapshots/{arg}{ext}",
  expect: {
    timeout: 10_000,
  },
  use: {
    baseURL,
    trace: "on-first-retry",
  },
  webServer: {
    command: `${pythonCommand} ../../scripts/serve_docs.py${servePrebuiltArg}`,
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
