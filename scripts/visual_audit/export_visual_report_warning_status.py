from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH = ROOT / "artifacts" / "visual-state" / "report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a warning-level workflow environment variable when the visual "
            "state report contains non-blocking findings."
        )
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"Path to the visual state report JSON (default: {DEFAULT_REPORT_PATH}).",
    )
    parser.add_argument(
        "--severity",
        default="medium",
        help="Finding severity that should trigger a warning export (default: medium).",
    )
    parser.add_argument(
        "--env-key",
        default="STEP_VISUAL_STATE_WARNINGS",
        help="Environment variable name to write into GITHUB_ENV (default: STEP_VISUAL_STATE_WARNINGS).",
    )
    parser.add_argument(
        "--count-env-key",
        default="STEP_VISUAL_STATE_WARNING_COUNT",
        help="Environment variable name to store the matching warning count (default: STEP_VISUAL_STATE_WARNING_COUNT).",
    )
    return parser.parse_args()


def append_github_env(name: str, value: str) -> None:
    github_env = os.environ.get("GITHUB_ENV")
    if not github_env:
        raise SystemExit("GITHUB_ENV is required to export workflow warning state")
    with Path(github_env).open("a", encoding="utf8") as handle:
        handle.write(f"{name}={value}\n")


def main() -> int:
    args = parse_args()
    if not args.report_path.exists():
        print(
            f"No visual state report found at {args.report_path}; skipping warning export."
        )
        return 0

    report = json.loads(args.report_path.read_text(encoding="utf8"))
    findings = report.get("findings", [])
    matching = [
        finding for finding in findings if finding.get("severity") == args.severity
    ]

    append_github_env(args.count_env_key, str(len(matching)))
    if matching:
        append_github_env(args.env_key, "warning")
        print(
            f"Exported {args.env_key}=warning because {len(matching)} {args.severity} findings were detected."
        )
    else:
        print(f"No {args.severity} findings detected; no warning status exported.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
