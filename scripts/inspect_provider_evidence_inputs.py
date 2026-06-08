from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_FILES = (
    "manifest.json",
    "market_data.json",
    "cleaned_quotes.parquet",
    "cleaned_quotes.csv",
    "rejected_quotes.parquet",
    "rejected_quotes.csv",
    "heston_quotes.parquet",
    "heston_quotes.csv",
    "surface_inputs.parquet",
    "heston_fit_summary.csv",
    "warnings.json",
)

SENSITIVE_NAME_FRAGMENTS = (
    "api_key",
    "apikey",
    "secret",
    "authorization",
    "auth_header",
    "bearer",
    "token",
    "password",
    "credential",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        payload
        if isinstance(payload, dict)
        else {"_non_object_json": type(payload).__name__}
    )


def _frame_info(path: Path) -> dict[str, Any]:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        return {}
    return {
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "sensitive_column_name_hits": [
            str(column)
            for column in frame.columns
            if any(
                fragment in str(column).lower() for fragment in SENSITIVE_NAME_FRAGMENTS
            )
        ],
    }


def inspect_bundle(
    bundle_root: Path, provider_summary_json: Path | None = None
) -> dict[str, Any]:
    bundle_root = bundle_root.resolve()
    files = {}
    for name in EXPECTED_FILES:
        path = bundle_root / name
        files[name] = {"exists": path.exists(), "path": str(path)}
        if path.exists() and path.suffix in {".csv", ".parquet"}:
            files[name].update(_frame_info(path))

    manifest = _read_json(bundle_root / "manifest.json")
    warnings = _read_json(bundle_root / "warnings.json")
    provider_summary = (
        _read_json(provider_summary_json) if provider_summary_json else {}
    )

    return {
        "bundle_root": str(bundle_root),
        "provider_summary_json": (
            str(provider_summary_json.resolve()) if provider_summary_json else None
        ),
        "manifest_keys": sorted(manifest.keys()),
        "provider_summary_keys": sorted(provider_summary.keys()),
        "warning_keys": sorted(warnings.keys()),
        "files": files,
        "builder_command": (
            "python scripts/build_provider_evidence_artifacts.py "
            f"--bundle-root {bundle_root} "
            f"--provider-summary-json {provider_summary_json or '<provider-summary-json>'} "
            "--output-dir docs/assets/generated/provider_evidence "
            "--profile release"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a local provider/model-validation bundle without printing "
            "raw rows or provider payload bodies."
        )
    )
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--provider-summary-json", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    payload = inspect_bundle(args.bundle_root, args.provider_summary_json)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.write_text(text, encoding="utf-8")
        print(args.output_json)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
