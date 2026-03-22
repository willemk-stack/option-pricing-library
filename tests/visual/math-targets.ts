import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

let cachedMathRoutes: string[] | null = null;

function resolvePythonCommand(): string {
  const explicitCommand = process.env.PYTHON || process.env.PYTHON_EXECUTABLE;
  if (explicitCommand) {
    return explicitCommand;
  }

  const windowsVenvPython = resolve(__dirname, "../../.venv/Scripts/python.exe");
  if (process.platform === "win32" && existsSync(windowsVenvPython)) {
    return windowsVenvPython;
  }

  const posixVenvPython = resolve(__dirname, "../../.venv/bin/python");
  if (process.platform !== "win32" && existsSync(posixVenvPython)) {
    return posixVenvPython;
  }

  return "python";
}

export function loadMathRoutes(): string[] {
  if (cachedMathRoutes) {
    return cachedMathRoutes;
  }

  const scriptPath = resolve(
    __dirname,
    "../../scripts/visual_audit/discover_math_routes.py"
  );
  const raw = execFileSync(resolvePythonCommand(), [scriptPath, "--format", "json"], {
    cwd: resolve(__dirname, "../.."),
    encoding: "utf8",
  });

  const parsed = JSON.parse(raw) as unknown;
  if (!Array.isArray(parsed) || parsed.some((route) => typeof route !== "string")) {
    throw new Error("discover_math_routes.py returned an invalid route list.");
  }
  if (parsed.length === 0) {
    throw new Error("No built math routes were discovered.");
  }

  cachedMathRoutes = parsed;
  return cachedMathRoutes;
}
