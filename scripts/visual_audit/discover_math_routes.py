from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from docs_site_contract import (  # noqa: E402
    load_docs_site_contract,
    verify_prebuilt_site,
)


def html_path_to_route(site_dir: Path, html_path: Path) -> str:
    relative = html_path.relative_to(site_dir).as_posix()
    if relative == "index.html":
        return "/"
    return "/" + relative[: -len("index.html")]


def discover_math_routes(site_dir: Path) -> list[str]:
    routes: list[str] = []
    for html_path in sorted(site_dir.rglob("index.html")):
        text = html_path.read_text(encoding="utf8")
        if 'class="arithmatex"' not in text:
            continue
        routes.append(html_path_to_route(site_dir, html_path))
    return sorted(dict.fromkeys(routes))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover built docs routes that contain MathJax placeholders."
    )
    parser.add_argument(
        "--format",
        choices=("json", "lines"),
        default="json",
        help="Output format.",
    )
    parser.add_argument(
        "--site-dir",
        help="Optional explicit built site directory. Defaults to the docs site contract.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = load_docs_site_contract()
    site_dir = Path(args.site_dir) if args.site_dir else contract.site_dir

    if site_dir == contract.site_dir:
        verify_prebuilt_site(contract)
    elif not site_dir.is_dir():
        raise SystemExit(f"Built site directory does not exist: {site_dir}")

    routes = discover_math_routes(site_dir)

    if args.format == "lines":
        for route in routes:
            print(route, flush=True)
        return 0

    print(json.dumps(routes), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
