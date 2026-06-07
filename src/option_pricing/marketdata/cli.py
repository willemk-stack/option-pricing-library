from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from os import PathLike, fspath
from pathlib import Path

from option_pricing.marketdata.bundles import ModelValidationBundleConfig
from option_pricing.marketdata.config import (
    AlpacaConfig,
    FredConfig,
    MarketDataPolicyConfig,
    PipelineConfig,
    StorageConfig,
)
from option_pricing.marketdata.errors import MarketDataProviderError
from option_pricing.marketdata.pipeline import MarketDataPipeline
from option_pricing.marketdata.provider_confidence import (
    validate_provider_snapshot_bundle,
)
from option_pricing.marketdata.provider_results import (
    provider_snapshot_public_summary,
)

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_RATE_SERIES = "DGS3MO"
DEFAULT_RATE_LOOKBACK_DAYS = 90
DEFAULT_TIMEFRAME = "1Day"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="option-pricing-marketdata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Run local/private marketdata artifact workflows through "
            "MarketDataPipeline."
        ),
        epilog=(
            "Public proof path: use scripts/demo_local_market_validation.py for the "
            "deterministic synthetic fixture workflow.\n"
            "Private provider evidence: snapshot, refresh-daily, backfill-fred, and "
            "backfill-bars may call Alpaca/FRED and write local-only artifacts.\n"
            "Model-ready path: validate-bundle checks saved artifacts before "
            "load_model_validation_bundle -> prepare_heston_market_fit -> "
            "fit_heston_market."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser(
        "snapshot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Fetch, clean, and persist one private provider-backed snapshot.",
        description=(
            "Fetch one Alpaca/FRED-backed snapshot, clean option quotes, and write "
            "Bronze, Silver, Gold, rate-curve, and model-validation bundle artifacts "
            "under a local data root."
        ),
    )
    _add_common_storage_options(snapshot)
    _add_run_id_option(snapshot)
    _add_common_provider_options(snapshot)
    _add_snapshot_query_options(snapshot)
    snapshot.add_argument("--underlying", required=True, help="Underlying symbol.")

    refresh_daily = subparsers.add_parser(
        "refresh-daily",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Run private snapshots for several underlyings and aggregate the run.",
        description=(
            "Run one provider-backed snapshot per underlying, write child artifacts, "
            "and emit one aggregate summary for local/private validation."
        ),
    )
    _add_common_storage_options(refresh_daily)
    _add_common_provider_options(refresh_daily)
    _add_snapshot_query_options(refresh_daily)
    refresh_daily.add_argument(
        "--underlyings",
        nargs="+",
        required=True,
        help="One or more underlying symbols.",
    )
    refresh_daily.add_argument(
        "--run-id-prefix",
        default=None,
        help="Optional aggregate run ID prefix used for generated child run IDs.",
    )

    backfill_fred = subparsers.add_parser(
        "backfill-fred",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Backfill FRED series into local Bronze and Silver storage.",
        description=(
            "Fetch one or more FRED series and persist local Bronze/Silver "
            "backfill artifacts for private evidence roots."
        ),
    )
    _add_common_storage_options(backfill_fred)
    _add_run_id_option(backfill_fred)
    backfill_fred.add_argument(
        "--series",
        nargs="+",
        required=True,
        help="One or more FRED series IDs.",
    )
    backfill_fred.add_argument("--start", required=True, help="Start date.")
    backfill_fred.add_argument("--end", default=None, help="Optional end date.")

    backfill_bars = subparsers.add_parser(
        "backfill-bars",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Backfill Alpaca equity bars into local Bronze and Silver storage.",
        description=(
            "Fetch Alpaca equity bars and persist local Bronze/Silver backfill "
            "artifacts. This does not backfill historical option chains."
        ),
    )
    _add_common_storage_options(backfill_bars)
    _add_run_id_option(backfill_bars)
    _add_common_provider_options(backfill_bars)
    backfill_bars.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="One or more equity symbols.",
    )
    backfill_bars.add_argument("--start", required=True, help="Start timestamp/date.")
    backfill_bars.add_argument("--end", required=True, help="End timestamp/date.")
    backfill_bars.add_argument(
        "--timeframe",
        default=DEFAULT_TIMEFRAME,
        help="Alpaca bar timeframe.",
    )

    validate_bundle = subparsers.add_parser(
        "validate-bundle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Validate saved model-facing artifacts without provider calls.",
        description=(
            "Read already-written market_data.json, cleaned_quotes.parquet, and "
            "heston_quotes.parquet files and verify the provider snapshot bundle "
            "contracts without credentials or live providers."
        ),
    )
    validate_bundle.add_argument(
        "--market-data",
        type=Path,
        required=True,
        help="Path to saved local market_data.json.",
    )
    validate_bundle.add_argument(
        "--cleaned-quotes",
        type=Path,
        required=True,
        help="Path to saved local cleaned_quotes.parquet.",
    )
    validate_bundle.add_argument(
        "--heston-quotes",
        type=Path,
        required=True,
        help="Path to saved local heston_quotes.parquet.",
    )
    validate_bundle.add_argument(
        "--json",
        action="store_true",
        help="Emit one stable, sanitized JSON object instead of human-readable text.",
    )

    return parser.parse_args(argv)


def _add_common_storage_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Local/private root for Bronze, Silver, Gold, and metadata artifacts. "
            "Provider-backed outputs from this root are not redistributable."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing artifacts for the requested partitions.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one stable, sanitized JSON object instead of human-readable text.",
    )
    parser.add_argument(
        "--library-commit",
        default=None,
        help="Optional library commit recorded in manifests and run metadata.",
    )
    parser.add_argument(
        "--policy-config",
        type=Path,
        default=None,
        help="Optional local JSON/TOML/YAML marketdata policy config.",
    )


def _add_run_id_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run ID; generated by the pipeline when omitted.",
    )


def _add_common_provider_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--equity-feed",
        default=None,
        help="Optional Alpaca equity feed override for stock quotes and bars.",
    )
    parser.add_argument(
        "--option-feed",
        default=None,
        help="Optional Alpaca option feed override for option chains.",
    )
    parser.add_argument(
        "--feed",
        default=None,
        help=(
            "Deprecated Alpaca feed alias. For snapshot/refresh it overrides "
            "the option feed only; for backfill-bars it overrides the equity feed."
        ),
    )


def _add_snapshot_query_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--asof",
        default=None,
        help="Optional valuation timestamp accepted by snapshot methods.",
    )
    parser.add_argument(
        "--rate-series",
        default=DEFAULT_RATE_SERIES,
        help="FRED rate series ID used for the risk-free rate.",
    )
    parser.add_argument(
        "--rate-lookback-days",
        type=int,
        default=DEFAULT_RATE_LOOKBACK_DAYS,
        help="FRED lookback window used for snapshot rate selection.",
    )
    parser.add_argument(
        "--curve-series",
        nargs="*",
        default=None,
        help="Zero or more FRED series IDs for the local rate-curve artifact.",
    )
    parser.add_argument(
        "--no-rate-curve",
        action="store_true",
        help="Disable the provider rate-curve artifact for this snapshot run.",
    )
    parser.add_argument(
        "--run-heston-smoke",
        action="store_true",
        help=(
            "Enable the optional model-validation bundle Heston smoke check. "
            "This is compatibility evidence, not calibration-quality evidence."
        ),
    )
    parser.add_argument(
        "--dividend-yield",
        type=float,
        default=0.0,
        help="Dividend yield override used for snapshot market inputs.",
    )
    parser.add_argument(
        "--dividend-yield-source",
        default="assumption",
        help="Dividend yield source label recorded in metadata.",
    )
    parser.add_argument("--expiry-gte", default=None)
    parser.add_argument("--expiry-lte", default=None)
    parser.add_argument("--strike-gte", type=float, default=None)
    parser.add_argument("--strike-lte", type=float, default=None)
    parser.add_argument("--option-type", default=None)
    parser.add_argument(
        "--max-equity-quote-age-seconds",
        type=float,
        default=None,
        help="Warn/fail when the equity quote is older than this many seconds.",
    )
    parser.add_argument(
        "--max-option-quote-age-seconds",
        type=float,
        default=None,
        help="Warn/reject when option quotes are older than this many seconds.",
    )
    parser.add_argument(
        "--quote-freshness-mode",
        choices=("demo_lenient", "end_of_day", "intraday_strict"),
        default="demo_lenient",
        help="Quote freshness mode used by snapshot cleaning and diagnostics.",
    )
    parser.add_argument(
        "--max-quote-age-seconds",
        type=float,
        default=None,
        help="Unified quote max age for intraday_strict mode.",
    )
    parser.add_argument(
        "--allow-prior-session",
        action="store_true",
        help="Allow prior-session quote data in end_of_day mode.",
    )
    parser.add_argument(
        "--stale-quote-action",
        choices=("warn", "reject", "fail"),
        default="warn",
        help="Action for stale option quotes under the selected freshness mode.",
    )
    parser.add_argument(
        "--reject-stale-option-quotes",
        action="store_true",
        help="Move stale accepted option quotes to rejected_quotes.",
    )
    parser.add_argument(
        "--reject-option-quotes-after-asof",
        action="store_true",
        help="Move accepted option quotes after the snapshot asof to rejected_quotes.",
    )
    parser.add_argument(
        "--reject-stale-equity-quote",
        action="store_true",
        help="Fail the snapshot when the equity quote violates freshness policy.",
    )
    parser.add_argument(
        "--min-accepted-contracts",
        type=int,
        default=None,
        help="Minimum accepted option contracts required by the quality policy.",
    )
    parser.add_argument(
        "--min-accepted-calls",
        type=int,
        default=None,
        help="Minimum accepted calls required by the quality policy.",
    )
    parser.add_argument(
        "--min-accepted-puts",
        type=int,
        default=None,
        help="Minimum accepted puts required by the quality policy.",
    )
    parser.add_argument(
        "--min-expiries",
        type=int,
        default=None,
        help="Minimum accepted expiries required by the quality policy.",
    )


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    default_alpaca = AlpacaConfig()
    equity_feed, option_feed = _config_feeds(args, default_alpaca)
    policy_config = (
        MarketDataPolicyConfig()
        if getattr(args, "policy_config", None) is None
        else MarketDataPolicyConfig.from_file(args.policy_config)
    )
    return PipelineConfig(
        alpaca=AlpacaConfig(equity_feed=equity_feed, option_feed=option_feed),
        fred=FredConfig(),
        storage=StorageConfig(root=args.data_root),
        policy=policy_config,
    )


def _config_feeds(
    args: argparse.Namespace,
    default_alpaca: AlpacaConfig,
) -> tuple[str, str]:
    legacy_feed = getattr(args, "feed", None)
    equity_feed = getattr(args, "equity_feed", None) or default_alpaca.equity_feed
    option_feed = getattr(args, "option_feed", None) or default_alpaca.option_feed
    if legacy_feed:
        if getattr(args, "command", None) == "backfill-bars":
            if getattr(args, "equity_feed", None) is None:
                equity_feed = legacy_feed
        elif getattr(args, "option_feed", None) is None:
            option_feed = legacy_feed
    return equity_feed, option_feed


def _build_pipeline(args: argparse.Namespace) -> MarketDataPipeline:
    return MarketDataPipeline(
        _build_config(args),
        bundle_config=ModelValidationBundleConfig(
            run_heston_smoke=bool(getattr(args, "run_heston_smoke", False))
        ),
    )


def _run_command(args: argparse.Namespace) -> tuple[str, object]:
    if args.command == "validate-bundle":
        return (
            "validate-bundle",
            validate_provider_snapshot_bundle(
                market_data_path=args.market_data,
                cleaned_quotes_path=args.cleaned_quotes,
                heston_quotes_path=args.heston_quotes,
            ),
        )

    pipeline = _build_pipeline(args)
    if args.command == "snapshot":
        return (
            "snapshot",
            pipeline.snapshot(
                args.underlying,
                asof=args.asof,
                run_id=args.run_id,
                rate_series_id=args.rate_series,
                expiry_gte=args.expiry_gte,
                expiry_lte=args.expiry_lte,
                strike_gte=args.strike_gte,
                strike_lte=args.strike_lte,
                option_type=args.option_type,
                feed=args.feed,
                equity_feed=args.equity_feed,
                option_feed=args.option_feed,
                dividend_yield=args.dividend_yield,
                dividend_yield_source=args.dividend_yield_source,
                rate_lookback_days=args.rate_lookback_days,
                curve_series_ids=_curve_series_ids(args),
                quality_policy=_quality_policy_payload(args),
                overwrite=args.overwrite,
                library_commit=args.library_commit,
            ),
        )
    if args.command == "refresh-daily":
        return (
            "refresh-daily",
            pipeline.refresh_daily(
                args.underlyings,
                asof=args.asof,
                run_id_prefix=args.run_id_prefix,
                rate_series_id=args.rate_series,
                expiry_gte=args.expiry_gte,
                expiry_lte=args.expiry_lte,
                strike_gte=args.strike_gte,
                strike_lte=args.strike_lte,
                option_type=args.option_type,
                feed=args.feed,
                equity_feed=args.equity_feed,
                option_feed=args.option_feed,
                dividend_yield=args.dividend_yield,
                dividend_yield_source=args.dividend_yield_source,
                rate_lookback_days=args.rate_lookback_days,
                curve_series_ids=_curve_series_ids(args),
                quality_policy=_quality_policy_payload(args),
                overwrite=args.overwrite,
                library_commit=args.library_commit,
            ),
        )
    if args.command == "backfill-fred":
        return (
            "backfill-fred",
            pipeline.backfill_fred(
                args.series,
                start=args.start,
                end=args.end,
                run_id=args.run_id,
                overwrite=args.overwrite,
                library_commit=args.library_commit,
            ),
        )
    if args.command == "backfill-bars":
        return (
            "backfill-bars",
            pipeline.backfill_bars(
                args.symbols,
                start=args.start,
                end=args.end,
                timeframe=args.timeframe,
                feed=args.feed,
                equity_feed=args.equity_feed,
                run_id=args.run_id,
                overwrite=args.overwrite,
                library_commit=args.library_commit,
            ),
        )

    raise ValueError(f"Unsupported command: {args.command}")


def _emit_result(command: str, result: object, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(_result_payload(command, result), indent=2, sort_keys=True))
        return

    if command == "snapshot":
        lines = _snapshot_summary_lines(result)
    elif command == "refresh-daily":
        lines = _refresh_daily_summary_lines(result)
    elif command == "backfill-fred":
        lines = _backfill_summary_lines(result, dataset="fred_series")
    elif command == "backfill-bars":
        lines = _backfill_summary_lines(result, dataset="equity_bars")
    elif command == "validate-bundle":
        lines = _validate_bundle_summary_lines(result)
    else:
        raise ValueError(f"Unsupported command: {command}")
    print("\n".join(lines))


def _result_payload(command: str, result: object) -> dict[str, object]:
    if command == "snapshot":
        return _snapshot_payload(result)
    if command == "refresh-daily":
        return _refresh_daily_payload(result)
    if command == "backfill-fred":
        return _backfill_payload(result, command=command, dataset="fred_series")
    if command == "backfill-bars":
        return _backfill_payload(result, command=command, dataset="equity_bars")
    if command == "validate-bundle":
        payload = _jsonable(result)
        if not isinstance(payload, Mapping):
            raise TypeError("validate-bundle result must be mapping-like")
        return {"command": "validate-bundle", **payload}
    raise ValueError(f"Unsupported command: {command}")


def _snapshot_payload(result: object) -> dict[str, object]:
    main_paths = _snapshot_main_artifact_paths(result)
    return {
        "command": "snapshot",
        "public_summary": provider_snapshot_public_summary(result),
        "underlying": _jsonable(getattr(result, "underlying", None)),
        "asof": _jsonable(getattr(result, "asof", None)),
        "run_id": _jsonable(getattr(result, "run_id", None)),
        "spot": _jsonable(getattr(result, "spot", None)),
        "rate": _jsonable(getattr(result, "rate", None)),
        "rate_source": _jsonable(getattr(result, "rate_source", None)),
        "rate_observation_date": _jsonable(
            getattr(result, "rate_observation_date", None)
        ),
        "rate_series_id": _jsonable(getattr(result, "rate_series_id", None)),
        "selected_rate": _jsonable(
            getattr(result, "selected_rate", getattr(result, "rate", None))
        ),
        "flat_rate": _jsonable(
            getattr(result, "flat_rate", getattr(result, "rate", None))
        ),
        "equity_provider": _jsonable(getattr(result, "equity_provider", None)),
        "equity_feed": _jsonable(getattr(result, "equity_feed", None)),
        "option_provider": _jsonable(getattr(result, "option_provider", None)),
        "option_feed": _jsonable(
            getattr(result, "option_feed", getattr(result, "feed", None))
        ),
        "dividend_yield": _jsonable(getattr(result, "dividend_yield", None)),
        "dividend_yield_source": _jsonable(
            getattr(result, "dividend_yield_source", None)
        ),
        "rate_policy": _jsonable(getattr(result, "rate_policy", {})),
        "dividend_policy": _jsonable(getattr(result, "dividend_policy", {})),
        "option_cleaning_policy": _jsonable(
            getattr(result, "option_cleaning_policy", {})
        ),
        "data_policy": _jsonable(getattr(result, "data_policy", {})),
        "raw_option_contract_count": _jsonable(
            getattr(result, "raw_option_contract_count", None)
        ),
        "normalized_option_contract_count": _jsonable(
            getattr(result, "normalized_option_contract_count", None)
        ),
        "dropped_before_cleaning_count": _jsonable(
            getattr(result, "dropped_before_cleaning_count", None)
        ),
        "accepted_quote_count": _jsonable(
            getattr(result, "accepted_quote_count", None)
        ),
        "rejected_quote_count": _jsonable(
            getattr(result, "rejected_quote_count", None)
        ),
        "provider_rejected_contract_count": _jsonable(
            getattr(result, "provider_rejected_contract_count", None)
        ),
        "quality_policy": _jsonable(getattr(result, "quality_policy", {})),
        "quote_freshness": _jsonable(getattr(result, "quote_freshness", {})),
        "provider_operation_diagnostics": _diagnostics_from_result(result),
        "main_artifact_paths": {
            name: str(path) for name, path in main_paths.items() if path is not None
        },
        "artifact_paths": [str(path) for path in _artifact_paths(result)],
        "warnings": list(_warnings_from_result(result)),
    }


def _refresh_daily_payload(result: object) -> dict[str, object]:
    return {
        "command": "refresh-daily",
        "aggregate_run_id": _jsonable(getattr(result, "aggregate_run_id", None)),
        "child_run_ids": [
            _jsonable(run_id) for run_id in getattr(result, "child_run_ids", ())
        ],
        "underlyings": [
            _jsonable(underlying) for underlying in getattr(result, "underlyings", ())
        ],
        "counts": _refresh_daily_counts_payload(getattr(result, "counts", None)),
        "artifact_paths": [str(path) for path in _artifact_paths(result)],
        "warnings": list(_warnings_from_result(result)),
        "snapshots": [
            _snapshot_payload(snapshot) for snapshot in getattr(result, "results", ())
        ],
    }


def _refresh_daily_counts_payload(counts: object) -> dict[str, object]:
    field_names = (
        "raw_option_contract_count",
        "normalized_option_contract_count",
        "dropped_before_cleaning_count",
        "provider_rejected_contract_count",
        "accepted_quote_count",
        "rejected_quote_count",
    )
    if isinstance(counts, Mapping):
        return {
            name: _jsonable(counts.get(name)) for name in field_names if name in counts
        }
    return {
        name: _jsonable(getattr(counts, name, None))
        for name in field_names
        if getattr(counts, name, None) is not None
    }


def _backfill_payload(
    result: object,
    *,
    command: str,
    dataset: str,
) -> dict[str, object]:
    metadata = getattr(result, "metadata", None)
    stats = getattr(result, "stats", None)
    return {
        "command": command,
        "run_id": _jsonable(getattr(metadata, "run_id", None)),
        "run_ids": [_jsonable(run_id) for run_id in getattr(result, "run_ids", ())],
        "dataset": dataset,
        "row_counts": {
            "rows_in": _jsonable(getattr(stats, "rows_in", 0)),
            "rows_out": _jsonable(getattr(stats, "rows_out", 0)),
        },
        "artifact_paths": [str(path) for path in _artifact_paths(result)],
        "warnings": list(_warnings_from_result(result)),
    }


def _snapshot_summary_lines(result: object) -> list[str]:
    lines = [
        "Market snapshot completed.",
        f"underlying: {getattr(result, 'underlying', None)}",
        f"asof: {_text(getattr(result, 'asof', None))}",
        f"run_id: {getattr(result, 'run_id', None)}",
        f"spot: {getattr(result, 'spot', None)}",
        f"rate: {getattr(result, 'rate', None)}",
        f"rate_source: {getattr(result, 'rate_source', None)}",
        f"rate_observation_date: {_text(getattr(result, 'rate_observation_date', None))}",
        f"rate_series_id: {getattr(result, 'rate_series_id', None)}",
        f"selected_rate: {getattr(result, 'selected_rate', getattr(result, 'rate', None))}",
        f"flat_rate: {getattr(result, 'flat_rate', getattr(result, 'rate', None))}",
        f"equity_provider: {getattr(result, 'equity_provider', None)}",
        f"equity_feed: {getattr(result, 'equity_feed', None)}",
        f"option_provider: {getattr(result, 'option_provider', None)}",
        f"option_feed: {getattr(result, 'option_feed', getattr(result, 'feed', None))}",
        f"dividend_yield: {getattr(result, 'dividend_yield', None)}",
        f"dividend_yield_source: {getattr(result, 'dividend_yield_source', None)}",
        "raw_option_contract_count: "
        f"{getattr(result, 'raw_option_contract_count', None)}",
        "normalized_option_contract_count: "
        f"{getattr(result, 'normalized_option_contract_count', None)}",
        "dropped_before_cleaning_count: "
        f"{getattr(result, 'dropped_before_cleaning_count', None)}",
        f"accepted_quote_count: {getattr(result, 'accepted_quote_count', None)}",
        f"rejected_quote_count: {getattr(result, 'rejected_quote_count', None)}",
        "provider_rejected_contract_count: "
        f"{getattr(result, 'provider_rejected_contract_count', None)}",
        "main_artifact_paths:",
    ]
    main_paths = _snapshot_main_artifact_paths(result)
    lines.extend(
        f"  {name}: {path}" for name, path in main_paths.items() if path is not None
    )
    lines.extend(_warning_lines(_warnings_from_result(result)))
    return lines


def _refresh_daily_summary_lines(result: object) -> list[str]:
    count_lines = [
        f"  {name}: {value}"
        for name, value in _refresh_daily_counts_payload(
            getattr(result, "counts", None)
        ).items()
    ]
    snapshot_lines = [
        "snapshots:",
        *(
            "  - "
            f"{getattr(snapshot, 'underlying', None)} "
            f"run_id={getattr(snapshot, 'run_id', None)} "
            f"accepted={getattr(snapshot, 'accepted_quote_count', None)} "
            f"rejected={getattr(snapshot, 'rejected_quote_count', None)} "
            f"dropped={getattr(snapshot, 'dropped_before_cleaning_count', None)}"
            for snapshot in getattr(result, "results", ())
        ),
    ]
    lines = [
        "Daily refresh completed.",
        f"aggregate_run_id: {getattr(result, 'aggregate_run_id', None)}",
        "underlyings: "
        + ", ".join(str(value) for value in getattr(result, "underlyings", ())),
        "child_run_ids:",
        *(f"  - {run_id}" for run_id in getattr(result, "child_run_ids", ())),
        "counts:",
        *count_lines,
        "artifact_paths:",
        *(f"  - {path}" for path in _artifact_paths(result)),
        *snapshot_lines,
    ]
    lines.extend(_warning_lines(_warnings_from_result(result)))
    return lines


def _backfill_summary_lines(result: object, *, dataset: str) -> list[str]:
    metadata = getattr(result, "metadata", None)
    stats = getattr(result, "stats", None)
    lines = [
        f"{dataset} backfill completed.",
        f"run_id: {getattr(metadata, 'run_id', None)}",
        f"dataset: {dataset}",
        f"rows_in: {getattr(stats, 'rows_in', 0)}",
        f"rows_out: {getattr(stats, 'rows_out', 0)}",
        "artifact_paths:",
    ]
    lines.extend(f"  - {path}" for path in _artifact_paths(result))
    lines.extend(_warning_lines(_warnings_from_result(result)))
    return lines


def _validate_bundle_summary_lines(result: object) -> list[str]:
    return [
        "Provider snapshot bundle validation passed.",
        f"underlying: {getattr(result, 'underlying', None)}",
        "cleaned_quote_count: " f"{getattr(result, 'cleaned_quote_count', None)}",
        f"heston_quote_count: {getattr(result, 'heston_quote_count', None)}",
        f"spot: {getattr(result, 'spot', None)}",
        f"rate: {getattr(result, 'rate', None)}",
        f"dividend_yield: {getattr(result, 'dividend_yield', None)}",
    ]


def _warning_lines(warnings: Sequence[str]) -> list[str]:
    if not warnings:
        return ["warnings: none"]
    return ["warnings:", *(f"  - {warning}" for warning in warnings)]


def _warnings_from_result(result: object) -> tuple[str, ...]:
    warnings: object | None = getattr(result, "warnings", None)
    if warnings is None:
        stats = getattr(result, "stats", None)
        warnings = getattr(stats, "warnings", None)
    if warnings is None:
        return ()
    if isinstance(warnings, str):
        return (warnings,)
    if isinstance(warnings, Sequence):
        return tuple(str(warning) for warning in warnings)
    return (str(warnings),)


def _diagnostics_from_result(result: object) -> list[object]:
    diagnostics = getattr(result, "diagnostics", ())
    out: list[object] = []
    for diagnostic in diagnostics:
        as_dict = getattr(diagnostic, "as_dict", None)
        if callable(as_dict):
            out.append(_jsonable(as_dict()))
        else:
            out.append(_jsonable(diagnostic))
    return out


def _curve_series_ids(
    args: argparse.Namespace,
) -> Sequence[str] | tuple[str, ...] | None:
    if getattr(args, "no_rate_curve", False):
        return ()
    curve_series = getattr(args, "curve_series", None)
    return curve_series


def _quality_policy_payload(args: argparse.Namespace) -> dict[str, object] | None:
    payload: dict[str, object] = {}
    for arg_name in (
        "quote_freshness_mode",
        "max_quote_age_seconds",
        "allow_prior_session",
        "stale_quote_action",
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            payload[arg_name] = value
    max_equity_age = getattr(args, "max_equity_quote_age_seconds", None)
    if max_equity_age is not None:
        payload["max_equity_quote_age_seconds"] = max_equity_age
    max_option_age = getattr(args, "max_option_quote_age_seconds", None)
    if max_option_age is not None:
        payload["max_option_quote_age_seconds"] = max_option_age
    if getattr(args, "reject_stale_option_quotes", False):
        payload["reject_stale_option_quotes"] = True
    if getattr(args, "reject_option_quotes_after_asof", False):
        payload["reject_option_quotes_after_asof"] = True
    if getattr(args, "reject_stale_equity_quote", False):
        payload["reject_stale_equity_quote"] = True
    for arg_name in (
        "min_accepted_contracts",
        "min_accepted_calls",
        "min_accepted_puts",
        "min_expiries",
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            payload[arg_name] = value
    return payload or None


def _artifact_paths(result: object) -> tuple[Path, ...]:
    paths = getattr(result, "artifact_paths", ())
    return tuple(Path(path) for path in paths)


def _snapshot_main_artifact_paths(result: object) -> dict[str, Path | None]:
    bronze_paths = getattr(result, "bronze_paths", None)
    silver_paths = getattr(result, "silver_paths", None)
    gold_paths = getattr(result, "gold_paths", None)
    rate_curve_paths = getattr(result, "rate_curve_paths", None)
    bundle = getattr(result, "model_validation_bundle", None)
    return {
        "bronze_manifest": _optional_path(getattr(bronze_paths, "manifest", None)),
        "silver_manifest": _optional_path(getattr(silver_paths, "manifest", None)),
        "provider_rejected_contracts": _optional_path(
            getattr(silver_paths, "provider_rejected_contracts", None)
        ),
        "market_data": _optional_path(getattr(gold_paths, "market_data", None)),
        "market_manifest": _optional_path(getattr(gold_paths, "market_manifest", None)),
        "rate_curve": _optional_path(getattr(rate_curve_paths, "rate_curve", None)),
        "rate_curve_manifest": _optional_path(
            getattr(rate_curve_paths, "manifest", None)
        ),
        "bundle_manifest": _optional_path(getattr(bundle, "manifest_path", None)),
    }


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    if isinstance(value, PathLike):
        path_value = fspath(value)
        if isinstance(path_value, str):
            return Path(path_value)
    raise TypeError(f"Expected path-like value, got {type(value).__name__}")


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat) and not isinstance(value, str):
        return isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_jsonable(item) for item in value]
    return value


def _text(value: object) -> str:
    jsonable = _jsonable(value)
    return "" if jsonable is None else str(jsonable)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        command, result = _run_command(args)
    except (
        MarketDataProviderError,
        FileNotFoundError,
        ValueError,
        FileExistsError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _emit_result(command, result, as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
