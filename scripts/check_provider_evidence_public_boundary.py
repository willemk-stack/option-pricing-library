from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

# Match actual secret-like values, not safe documentation words such as
# "credentials stay local/private".
SENSITIVE_TEXT_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{20,}"),
    re.compile(
        r"(?i)\b(api[_-]?key|secret|password|token)\b\s*[:=]\s*[\"']?[A-Za-z0-9_./+=-]{16,}"
    ),
)

FORBIDDEN_JSON_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "secret",
    "authorization",
    "auth_header",
    "bearer",
    "token",
    "password",
    "credential",
    "latest_quote",
    "option_chain",
)

ALLOWED_RAW_COUNT_COLUMNS = {"raw_contracts", "raw_contract_count"}


def _walk_json_keys(payload: Any, path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key)
            key_lower = key_text.lower()
            if any(fragment in key_lower for fragment in FORBIDDEN_JSON_KEY_FRAGMENTS):
                hits.append(f"{path}.{key_text}")
            hits.extend(_walk_json_keys(value, f"{path}.{key_text}"))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            hits.extend(_walk_json_keys(value, f"{path}[{index}]"))
    return hits


def _scan_text(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    hits = []
    for pattern in SENSITIVE_TEXT_PATTERNS:
        if pattern.search(text):
            hits.append(f"{path}: matched {pattern.pattern}")
    return hits


def _scan_json(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return [f"{path}: forbidden JSON key {hit}" for hit in _walk_json_keys(payload)]


def _scan_csv(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
    hits = []
    for column in header:
        column_lower = column.lower()
        if column_lower in ALLOWED_RAW_COUNT_COLUMNS:
            continue
        if any(fragment in column_lower for fragment in FORBIDDEN_JSON_KEY_FRAGMENTS):
            hits.append(f"{path}: forbidden CSV column {column}")
    return hits


def scan_public_provider_evidence(root: Path) -> list[str]:
    if not root.exists():
        raise FileNotFoundError(root)

    hits: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".json"}:
            hits.extend(_scan_json(path))
            hits.extend(_scan_text(path))
        elif path.suffix.lower() in {".csv"}:
            hits.extend(_scan_csv(path))
            hits.extend(_scan_text(path))
        elif path.suffix.lower() in {".svg", ".md", ".txt"}:
            hits.extend(_scan_text(path))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail if public provider evidence artifacts contain forbidden payload/secret markers."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("docs/assets/generated/provider_evidence"),
        help="Generated public provider evidence directory.",
    )
    args = parser.parse_args()
    hits = scan_public_provider_evidence(args.root)
    if hits:
        print("Public provider evidence boundary check failed:")
        for hit in hits:
            print(f"- {hit}")
        return 1
    print(f"Public provider evidence boundary check passed: {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
