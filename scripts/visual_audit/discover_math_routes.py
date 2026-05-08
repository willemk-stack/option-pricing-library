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


def normalize_route(path: str) -> str:
    stripped = path.strip()
    if not stripped:
        return ""
    if stripped == "/":
        return "/"
    return f"/{stripped.strip('/')}/"


def select_math_routes(
    discovered_routes: list[str],
    review_paths: list[str] | None,
) -> tuple[list[str], str | None]:
    normalized_filters = {
        normalized
        for path in review_paths or []
        if (normalized := normalize_route(path))
    }
    if not normalized_filters:
        return discovered_routes, None

    selected = [route for route in discovered_routes if route in normalized_filters]
    if selected:
        return selected, None

    return [], (
        "No selected review paths contain built math routes. "
        f"Review paths: {', '.join(sorted(normalized_filters))}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover built docs routes that contain MathJax placeholders."
    )
    parser.add_argument(
        "--format",
        choices=("json", "lines", "selection-json"),
        default="json",
        help="Output format.",
    )
    parser.add_argument(
        "--site-dir",
        help="Optional explicit built site directory. Defaults to the docs site contract.",
    )
    parser.add_argument(
        "--review-path",
        action="append",
        default=[],
        help="Optional review path filter. Repeat to select multiple routes.",
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
    selected_routes, message = select_math_routes(routes, args.review_path)

    if args.format == "lines":
        for route in selected_routes:
            print(route, flush=True)
        return 0

    if args.format == "selection-json":
        print(
            json.dumps(
                {
                    "routes": selected_routes,
                    "message": message,
                }
            ),
            flush=True,
        )
        return 0

    print(json.dumps(selected_routes), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
