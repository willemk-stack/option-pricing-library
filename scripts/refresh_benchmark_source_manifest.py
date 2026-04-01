from __future__ import annotations

import argparse
from pathlib import Path

try:
    from . import build_benchmark_artifacts as benchmark_artifacts
    from .benchmark_source_scope import is_benchmark_freshness_input
except ImportError:
    import build_benchmark_artifacts as benchmark_artifacts
    from benchmark_source_scope import is_benchmark_freshness_input

ROOT = Path(__file__).resolve().parents[1]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh the benchmark source manifest when pre-commit reports a "
            "benchmark-sensitive change."
        )
    )
    parser.add_argument("files", nargs="*")
    return parser.parse_args(argv)


def _should_refresh(files: list[str]) -> bool:
    if not files:
        return True
    return any(is_benchmark_freshness_input(path, ROOT) for path in files)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not _should_refresh(args.files):
        return 0
    return benchmark_artifacts.main(["--write-source-manifest"])


if __name__ == "__main__":
    raise SystemExit(main())
