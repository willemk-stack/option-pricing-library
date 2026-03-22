from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS_VISUAL_DIR = ROOT / "tests" / "visual"


def _resolve_node() -> str:
    for candidate in ("node.exe", "node"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError("Could not find node on PATH.")


def _playwright_command() -> list[str]:
    cli_path = TESTS_VISUAL_DIR / "node_modules" / "playwright" / "cli.js"
    if not cli_path.exists():
        raise FileNotFoundError(
            "Could not find tests/visual/node_modules/playwright/cli.js. Run `cd tests/visual && npm ci`."
        )
    return [_resolve_node(), str(cli_path)]


def _run(
    command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None
) -> None:
    printable = subprocess.list2cmdline(command)
    print(f"\n> {printable}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> int:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    _run([sys.executable, "scripts/render_readme.py", "--check"], env=env)
    _run([sys.executable, "scripts/render_performance_page.py", "--check"], env=env)
    _run([sys.executable, "scripts/check_docs_source_links.py"], env=env)
    _run([sys.executable, "-m", "mkdocs", "build", "--strict"], env=env)
    _run([sys.executable, "scripts/visual_audit/check_svg_assets.py"], env=env)

    playwright_env = env.copy()
    playwright_env["SKIP_DOCS_PREBUILD"] = "1"
    playwright_env["SERVE_PREBUILT_SITE"] = "1"
    _run(
        [*_playwright_command(), "test", "smoke.spec.ts"],
        cwd=TESTS_VISUAL_DIR,
        env=playwright_env,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
