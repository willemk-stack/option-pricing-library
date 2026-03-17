from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCS_BASE_URL = "http://127.0.0.1:8000/option-pricing-library/"


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("DOCS_BASE_URL", DEFAULT_DOCS_BASE_URL)

    if os.environ.get("SKIP_DOCS_PREBUILD") != "1":
        print("Rendering diagrams...", flush=True)
        run([sys.executable, "scripts/render_d2_diagrams.py"])

        print("Building generated visual assets...", flush=True)
        run(
            [
                sys.executable,
                "scripts/build_visual_artifacts.py",
                "all",
                "--profile",
                "ci",
            ]
        )

    print("Starting MkDocs server...", flush=True)
    run([sys.executable, "-m", "mkdocs", "serve", "-a", "127.0.0.1:8000"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
