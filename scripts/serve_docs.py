from __future__ import annotations

import argparse
import os
import subprocess
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlsplit, urlunsplit

from docs_site_contract import (
    load_docs_site_contract,
    load_docs_visual_config,
    verify_prebuilt_site,
)

ROOT = Path(__file__).resolve().parents[1]
DOCS_VISUAL_CONFIG = load_docs_visual_config()
DOCS_SITE_CONTRACT = load_docs_site_contract()
DEFAULT_DOCS_BASE_URL = DOCS_VISUAL_CONFIG["docs_base_url"]
DEFAULT_BUILD_PROFILE = DOCS_VISUAL_CONFIG["build_profile"]
SITE_DIR = DOCS_SITE_CONTRACT.site_dir


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


def resolve_bind_address(base_url: str) -> str:
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    return f"{host}:{port}"


def resolve_docs_prefix(base_url: str) -> str:
    path = urlparse(base_url).path.rstrip("/")
    return "" if path == "/" else path


def split_bind_address(bind_address: str) -> tuple[str, int]:
    host, port = bind_address.rsplit(":", 1)
    return host, int(port)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the built MkDocs site for Playwright-based docs validation."
    )
    parser.add_argument(
        "--serve-prebuilt",
        action="store_true",
        help="Serve the existing site/ directory without rebuilding assets or MkDocs.",
    )
    return parser.parse_args()


def should_serve_prebuilt(args: argparse.Namespace) -> bool:
    if args.serve_prebuilt:
        return True

    env_value = os.environ.get("SERVE_PREBUILT_SITE", "").strip().lower()
    return env_value in {"1", "true", "yes", "on"}


class DocsSiteHandler(SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args: Any,
        directory: str,
        docs_prefix: str,
        **kwargs: Any,
    ) -> None:
        self.docs_prefix = docs_prefix
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        if self._handle_prefix_redirect():
            return
        if not self._is_served_path():
            self.send_error(404, "File not found")
            return
        super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802
        if self._handle_prefix_redirect():
            return
        if not self._is_served_path():
            self.send_error(404, "File not found")
            return
        super().do_HEAD()

    def translate_path(self, path: str) -> str:
        stripped_path = self._strip_docs_prefix(path)
        if stripped_path is None:
            stripped_path = "/__missing__"
        return super().translate_path(stripped_path)

    def _handle_prefix_redirect(self) -> bool:
        parsed = urlsplit(self.path)
        request_path = parsed.path or "/"
        if not self.docs_prefix:
            return False

        if request_path == "/":
            target = urlunsplit(
                ("", "", f"{self.docs_prefix}/", parsed.query, parsed.fragment)
            )
            self.send_response(302)
            self.send_header("Location", target)
            self.end_headers()
            return True

        if request_path == self.docs_prefix:
            target = urlunsplit(
                ("", "", f"{self.docs_prefix}/", parsed.query, parsed.fragment)
            )
            self.send_response(301)
            self.send_header("Location", target)
            self.end_headers()
            return True

        return False

    def _is_served_path(self) -> bool:
        parsed = urlsplit(self.path)
        request_path = parsed.path or "/"
        if not self.docs_prefix:
            return True
        return request_path.startswith(f"{self.docs_prefix}/")

    def _strip_docs_prefix(self, raw_path: str) -> str | None:
        parsed = urlsplit(raw_path)
        request_path = parsed.path or "/"

        if not self.docs_prefix:
            stripped_path = request_path
        elif request_path == self.docs_prefix:
            stripped_path = "/"
        elif request_path.startswith(f"{self.docs_prefix}/"):
            stripped_path = request_path[len(self.docs_prefix) :]
        else:
            return None

        return urlunsplit(("", "", stripped_path or "/", parsed.query, parsed.fragment))


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("DOCS_BASE_URL", DEFAULT_DOCS_BASE_URL)
    docs_base_url = os.environ["DOCS_BASE_URL"]
    bind_address = resolve_bind_address(docs_base_url)
    docs_prefix = resolve_docs_prefix(docs_base_url)
    build_profile = os.environ.get("DOCS_VISUAL_BUILD_PROFILE", DEFAULT_BUILD_PROFILE)
    serve_prebuilt = should_serve_prebuilt(args)

    if serve_prebuilt:
        try:
            verify_prebuilt_site(DOCS_SITE_CONTRACT)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        print(
            "Serving existing MkDocs site build from "
            f"{DOCS_SITE_CONTRACT.artifact_name}...",
            flush=True,
        )
    else:
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
                    build_profile,
                ]
            )

        print("Building MkDocs site...", flush=True)
        run([sys.executable, "-m", "mkdocs", "build", "--strict"])

    host, port = split_bind_address(bind_address)
    handler = partial(
        DocsSiteHandler,
        directory=str(SITE_DIR),
        docs_prefix=docs_prefix,
    )
    with ThreadingHTTPServer((host, port), handler) as server:
        print(
            f"Serving built site from {SITE_DIR} at http://{bind_address}{docs_prefix or '/'}",
            flush=True,
        )
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
