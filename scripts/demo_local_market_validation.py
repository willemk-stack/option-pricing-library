from __future__ import annotations

# ruff: noqa: E402
import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FIXTURE_ROOT = ROOT / "tests" / "marketdata" / "fixtures"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from option_pricing.marketdata.bundles import ModelValidationBundleConfig
from option_pricing.marketdata.pipeline import (
    LocalModelValidationPipelineResult,
    run_local_model_validation_pipeline,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the deterministic local market validation demo from the "
            "checked-in synthetic fixture. No live providers or credentials "
            "are used."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/marketdata-demo"),
        help="Local artifact root for Bronze, Silver, Gold, and bundle outputs.",
    )
    parser.add_argument(
        "--run-id",
        default="demo-run",
        help="Deterministic run ID used in local partition paths.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing artifacts for the same local run ID.",
    )
    parser.add_argument(
        "--library-commit",
        default="local-demo",
        help="Library commit label recorded in manifests.",
    )
    heston = parser.add_mutually_exclusive_group()
    heston.add_argument(
        "--skip-heston-smoke",
        dest="run_heston_smoke",
        action="store_false",
        help="Skip the optional Heston smoke check. This is the default.",
    )
    heston.add_argument(
        "--run-heston-smoke",
        dest="run_heston_smoke",
        action="store_true",
        help="Run the optional Heston smoke check.",
    )
    parser.set_defaults(run_heston_smoke=False)
    return parser.parse_args(argv)


def _bundle_artifact_path(
    result: LocalModelValidationPipelineResult,
    filename: str,
) -> Path:
    return result.model_validation_bundle.manifest_path.parent / filename


def _summary_lines(result: LocalModelValidationPipelineResult) -> list[str]:
    run_id = (
        result.local_snapshot.run_id or result.model_validation_bundle.metadata.run_id
    )
    valuation_date = result.local_snapshot.asof.date().isoformat()

    return [
        "Local market validation demo completed.",
        f"underlying: {result.local_snapshot.underlying}",
        f"run_id: {run_id}",
        f"valuation_date: {valuation_date}",
        f"bronze_manifest: {result.bronze_paths.manifest}",
        f"silver_manifest: {result.silver_paths.manifest}",
        f"market_data: {_bundle_artifact_path(result, 'market_data.json')}",
        f"heston_quotes: {_bundle_artifact_path(result, 'heston_quotes.parquet')}",
        f"bundle_manifest: {result.model_validation_bundle.manifest_path}",
        f"warnings: {_bundle_artifact_path(result, 'warnings.json')}",
        f"heston_fit_summary: {_bundle_artifact_path(result, 'heston_fit_summary.csv')}",
        "No live providers or credentials were used.",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_local_model_validation_pipeline(
        storage=args.output_dir,
        run_id=args.run_id,
        fixture_root=FIXTURE_ROOT,
        bundle_config=ModelValidationBundleConfig(
            run_heston_smoke=args.run_heston_smoke,
        ),
        overwrite=args.overwrite,
        library_commit=args.library_commit,
    )

    print("\n".join(_summary_lines(result)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
