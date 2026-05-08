import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

type MathRouteSelection = {
  routes: string[];
  skipMessage: string | null;
};

let cachedMathRouteSelection: MathRouteSelection | null = null;

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
  return loadMathRouteSelection().routes;
}

function readReviewPathFilters(): string[] {
  const rawFilters = process.env.MATH_REVIEW_PATHS || process.env.REVIEW_PATHS || "";
  return rawFilters
    .split(",")
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);
}

export function loadMathRouteSelection(): MathRouteSelection {
  if (cachedMathRouteSelection) {
    return cachedMathRouteSelection;
  }

  const scriptPath = resolve(
    __dirname,
    "../../scripts/visual_audit/discover_math_routes.py"
  );
  const reviewPathFilters = readReviewPathFilters();
  const raw = execFileSync(
    resolvePythonCommand(),
    [
      scriptPath,
      "--format",
      "selection-json",
      ...reviewPathFilters.flatMap((path) => ["--review-path", path]),
    ],
    {
      cwd: resolve(__dirname, "../.."),
      encoding: "utf8",
    }
  );

  const parsed = JSON.parse(raw) as unknown;
  if (
    typeof parsed !== "object" ||
    parsed === null ||
    !Array.isArray((parsed as { routes?: unknown }).routes) ||
    (parsed as { routes: unknown[] }).routes.some((route) => typeof route !== "string")
  ) {
    throw new Error("discover_math_routes.py returned an invalid route selection.");
  }

  const selection = parsed as {
    routes: string[];
    message?: unknown;
  };
  const skipMessage = typeof selection.message === "string" ? selection.message : null;
  if (selection.routes.length === 0 && !skipMessage) {
    throw new Error("No built math routes were discovered.");
  }

  cachedMathRouteSelection = {
    routes: selection.routes,
    skipMessage,
  };
  return cachedMathRouteSelection;
}
