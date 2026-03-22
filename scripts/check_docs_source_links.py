from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
HTML_ATTR_PATTERN = re.compile(
    r'<(?P<tag>a|img)\b[^>]*\b(?P<attr>href|src)="(?P<target>[^"]+)"',
    re.IGNORECASE,
)
SCHEME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail when raw HTML links or images in docs markdown do not resolve "
            "relative to the source file."
        )
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Optional markdown files to scan. Defaults to docs/**/*.md.",
    )
    return parser.parse_args()


def iter_markdown_files(files: list[str]) -> list[Path]:
    if not files:
        return sorted(DOCS_DIR.rglob("*.md"))

    selected: list[Path] = []
    for raw_path in files:
        path = Path(raw_path)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        if path.suffix == ".md" and path.exists():
            selected.append(path)
    return sorted({path for path in selected})


def is_internal_relative_target(target: str) -> bool:
    stripped = target.strip()
    if not stripped:
        return False
    if stripped.startswith(("#", "//")):
        return False
    return not SCHEME_PATTERN.match(stripped)


def normalize_target(target: str) -> str:
    return unquote(target.split("#", 1)[0].split("?", 1)[0]).strip()


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def main() -> int:
    args = parse_args()
    issues: list[str] = []

    for markdown_file in iter_markdown_files(args.files):
        text = markdown_file.read_text(encoding="utf8")
        for match in HTML_ATTR_PATTERN.finditer(text):
            target = match.group("target")
            if not is_internal_relative_target(target):
                continue

            normalized_target = normalize_target(target)
            if not normalized_target:
                continue

            resolved_target = (markdown_file.parent / normalized_target).resolve()
            if resolved_target.exists():
                continue

            rel_path = markdown_file.relative_to(ROOT).as_posix()
            issues.append(
                f"{rel_path}:{line_number(text, match.start())}: "
                f"{match.group('tag')} {match.group('attr')}=\"{target}\""
            )

    if not issues:
        print("Docs source link guard passed.")
        return 0

    print(
        "Docs source link guard found raw HTML paths that do not resolve relative "
        "to the markdown source file.\n"
        'Use Markdown links/images inside a `markdown="1"` container when the '
        "URL should be rewritten by MkDocs.\n",
        file=sys.stderr,
    )
    for issue in issues:
        print(f"- {issue}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
