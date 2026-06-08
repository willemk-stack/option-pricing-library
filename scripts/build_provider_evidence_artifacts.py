from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ARTIFACT_SCHEMA_VERSION = "provider_evidence.v1"
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "assets" / "generated" / "provider_evidence"
REBUILD_COMMAND_STUB = (
    "python scripts/build_provider_evidence_artifacts.py "
    "--bundle-root <bundle-root> "
    "--provider-summary-json <provider-public-summary.json> "
    "--output-dir docs/assets/generated/provider_evidence "
    "--profile release"
)
ALLOWED_PUBLIC_COUNT_COLUMNS = {"raw_contracts", "normalized_contracts"}

FORBIDDEN_PUBLIC_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "secret",
    "authorization",
    "auth_header",
    "bearer",
    "token",
    "password",
    "credential",
    "contracts",
    "latest_quote",
    "option_chain",
    "bid",
    "ask",
)
PUBLIC_DATA_FILES = (
    "provider_evidence_manifest.json",
    "provider_public_summary.json",
    "provider_run_metadata.json",
    "provider_quality_policy.json",
    "provider_data_policy.json",
    "provider_quote_cleaning_summary.csv",
    "provider_rejection_reason_counts.csv",
    "provider_expiry_coverage.csv",
    "provider_strike_moneyness_coverage.csv",
    "provider_model_ready_summary.json",
    "provider_heston_fit_summary.csv",
    "provider_heston_parameter_summary.csv",
    "provider_warnings.json",
)
PUBLIC_FIGURE_FILES = (
    "provider_evidence_summary_card.light.svg",
    "provider_evidence_summary_card.dark.svg",
    "provider_pipeline_flow.light.svg",
    "provider_pipeline_flow.dark.svg",
    "provider_quote_cleaning_waterfall.light.png",
    "provider_quote_cleaning_waterfall.dark.png",
    "provider_expiry_strike_coverage.light.png",
    "provider_expiry_strike_coverage.dark.png",
    "provider_heston_fit_summary.light.png",
    "provider_heston_fit_summary.dark.png",
)
CAVEATS = (
    "Sanitized public provider evidence only.",
    "Raw provider payloads, credentials, and full provider-derived quote rows remain local/private.",
    "Heston fit status is workflow evidence, not a trading or production calibration-quality claim.",
)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _first_existing(root: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        direct = root / name
        if direct.exists():
            return direct
    for name in names:
        matches = list(root.rglob(name))
        if matches:
            return matches[0]
    return None


def _read_frame(root: Path, stem: str) -> pd.DataFrame:
    path = _first_existing(root, (f"{stem}.parquet", f"{stem}.csv"))
    if path is None:
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _sanitize_json_public(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        result: dict[str, Any] = {}
        for key, value in payload.items():
            key_text = str(key)
            if any(
                fragment in key_text.lower()
                for fragment in FORBIDDEN_PUBLIC_KEY_FRAGMENTS
            ):
                continue
            result[key_text] = _sanitize_json_public(value)
        return result
    if isinstance(payload, list):
        return [_sanitize_json_public(item) for item in payload]
    return payload


def _assert_no_forbidden_public_keys(payload: Any, *, where: str) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if any(
                fragment in str(key).lower()
                for fragment in FORBIDDEN_PUBLIC_KEY_FRAGMENTS
            ):
                raise ValueError(f"Forbidden public key {key!r} found in {where}")
            _assert_no_forbidden_public_keys(value, where=where)
    elif isinstance(payload, list):
        for item in payload:
            _assert_no_forbidden_public_keys(item, where=where)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _get(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    lower = {str(key).lower(): value for key, value in mapping.items()}
    for key in keys:
        value = lower.get(key.lower())
        if value not in (None, ""):
            return value
    return default


def _get_any(
    *mappings: Mapping[str, Any], keys: Iterable[str], default: Any = None
) -> Any:
    for mapping in mappings:
        value = _get(mapping, *keys, default=None)
        if value not in (None, ""):
            return value
    return default


def _rows(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = manifest.get("rows")
    return rows if isinstance(rows, Mapping) else {}


def _warning_count(payload: Mapping[str, Any]) -> int:
    count = 0
    for value in payload.values():
        if isinstance(value, list):
            count += len(value)
        elif isinstance(value, Mapping):
            count += _warning_count(value)
        elif value:
            count += 1
    return count


def _fit_source(bundle_root: Path, explicit: Path | None) -> pd.DataFrame:
    path = (
        explicit
        if explicit is not None
        else _first_existing(
            bundle_root, ("heston_fit_summary.csv", "provider_heston_fit_summary.csv")
        )
    )
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _fit_value(
    fit: pd.DataFrame,
    manifest: Mapping[str, Any],
    names: Iterable[str],
    default: Any = "",
) -> Any:
    if not fit.empty:
        lower_columns = {str(column).lower(): column for column in fit.columns}
        for name in names:
            column = lower_columns.get(name.lower())
            if column is not None:
                value = fit[column].iloc[0]
                if value not in (None, ""):
                    return value
    smoke = manifest.get("heston_smoke")
    if isinstance(smoke, Mapping):
        return _get(smoke, *names, default=default)
    return default


def _right_column(df: pd.DataFrame) -> str | None:
    return next(
        (
            column
            for column in ("right", "option_type", "cp_flag", "type")
            if column in df.columns
        ),
        None,
    )


def _expiry_column(df: pd.DataFrame) -> str | None:
    return next(
        (
            column
            for column in ("expiry", "expiration", "expiry_date")
            if column in df.columns
        ),
        None,
    )


def _log_moneyness(df: pd.DataFrame) -> pd.Series:
    if "log_moneyness" in df.columns:
        return pd.to_numeric(df["log_moneyness"], errors="coerce")
    if "moneyness" in df.columns:
        m = pd.to_numeric(df["moneyness"], errors="coerce")
        return m.where(m > 0).map(
            lambda value: math.log(value) if pd.notna(value) else float("nan")
        )
    if "strike" in df.columns and "spot" in df.columns:
        ratio = pd.to_numeric(df["strike"], errors="coerce") / pd.to_numeric(
            df["spot"], errors="coerce"
        )
        return ratio.where(ratio > 0).map(
            lambda value: math.log(value) if pd.notna(value) else float("nan")
        )
    return pd.Series([float("nan")] * len(df))


def _quote_summary(
    provider_summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    cleaned: pd.DataFrame,
    rejected: pd.DataFrame,
    heston: pd.DataFrame,
    fit: pd.DataFrame,
) -> pd.DataFrame:
    rows = _rows(manifest)
    accepted = _safe_int(
        _get(
            rows,
            "accepted_quotes",
            "cleaned_quotes",
            "cleaned_count",
            default=len(cleaned),
        )
    )
    rejected_count = _safe_int(
        _get(rows, "rejected_quotes", "rejected_count", default=len(rejected))
    )
    raw = _safe_int(
        _get_any(
            provider_summary,
            manifest,
            keys=("raw_contract_count", "raw_contracts", "raw_count"),
            default=accepted + rejected_count,
        )
    )
    normalized = _safe_int(
        _get_any(
            provider_summary,
            manifest,
            keys=(
                "normalized_contract_count",
                "normalized_contracts",
                "normalized_count",
            ),
            default=accepted + rejected_count,
        )
    )
    selected = _safe_int(
        _fit_value(
            fit,
            manifest,
            ("selected_quote_count", "quote_count", "n_quotes"),
            default=len(heston) or accepted,
        )
    )
    records = [
        ("raw_contracts", raw, raw),
        ("normalized_contracts", raw, normalized),
        ("accepted_quotes", normalized, accepted),
        ("rejected_quotes", normalized, rejected_count),
        ("selected_calibration_rows", accepted, selected),
    ]
    return pd.DataFrame(
        [
            {
                "stage": stage,
                "input_count": int(input_count),
                "output_count": int(output_count),
                "dropped_count": max(int(input_count) - int(output_count), 0),
                "retained_count": int(output_count),
            }
            for stage, input_count, output_count in records
        ]
    )


def _rejection_counts(
    manifest: Mapping[str, Any], rejected: pd.DataFrame
) -> pd.DataFrame:
    reason_col = next(
        (
            column
            for column in ("rejection_reason", "reason", "primary_rejection_reason")
            if column in rejected.columns
        ),
        None,
    )
    if reason_col is not None:
        counts = rejected[reason_col].fillna("unknown").astype(str).value_counts()
        return pd.DataFrame(
            {
                "stage": "quote_cleaning",
                "reason": counts.index,
                "count": counts.astype(int).values,
            }
        )
    candidates = [
        manifest.get("reason_counts"),
        manifest.get("rejection_reason_counts"),
        _rows(manifest).get("reason_counts"),
    ]
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return pd.DataFrame(
                [
                    {
                        "stage": "quote_cleaning",
                        "reason": str(reason),
                        "count": _safe_int(count),
                    }
                    for reason, count in candidate.items()
                ]
            )
    return pd.DataFrame(columns=["stage", "reason", "count"])


def _option_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    right_col = _right_column(out)
    right = (
        out[right_col].fillna("").astype(str).str.upper()
        if right_col
        else pd.Series([""] * len(out))
    )
    out["is_call_public"] = right.str.startswith("C") | right.eq("CALL")
    out["is_put_public"] = right.str.startswith("P") | right.eq("PUT")
    return out


def _expiry_coverage(cleaned: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "expiry",
        "days_to_expiry",
        "raw_contracts",
        "accepted_quotes",
        "calls",
        "puts",
        "min_strike",
        "max_strike",
        "min_log_moneyness",
        "max_log_moneyness",
    ]
    if cleaned.empty:
        return pd.DataFrame(columns=columns)
    df = _option_flags(cleaned)
    expiry_col = _expiry_column(df)
    if expiry_col is None:
        df["expiry"] = "unknown"
        expiry_col = "expiry"
    df["log_moneyness_public"] = _log_moneyness(df)
    if "strike" not in df.columns:
        df["strike"] = float("nan")
    if "days_to_expiry" not in df.columns:
        df["days_to_expiry"] = (
            pd.to_numeric(df["expiry_years"], errors="coerce") * 365
            if "expiry_years" in df.columns
            else float("nan")
        )
    records = []
    for expiry, group in df.groupby(expiry_col, dropna=False):
        records.append(
            {
                "expiry": str(expiry),
                "days_to_expiry": (
                    round(
                        float(
                            pd.to_numeric(
                                group["days_to_expiry"], errors="coerce"
                            ).median()
                        ),
                        6,
                    )
                    if group["days_to_expiry"].notna().any()
                    else ""
                ),
                "raw_contracts": len(group),
                "accepted_quotes": len(group),
                "calls": int(group["is_call_public"].sum()),
                "puts": int(group["is_put_public"].sum()),
                "min_strike": (
                    round(
                        float(pd.to_numeric(group["strike"], errors="coerce").min()), 6
                    )
                    if group["strike"].notna().any()
                    else ""
                ),
                "max_strike": (
                    round(
                        float(pd.to_numeric(group["strike"], errors="coerce").max()), 6
                    )
                    if group["strike"].notna().any()
                    else ""
                ),
                "min_log_moneyness": (
                    round(float(group["log_moneyness_public"].min()), 6)
                    if group["log_moneyness_public"].notna().any()
                    else ""
                ),
                "max_log_moneyness": (
                    round(float(group["log_moneyness_public"].max()), 6)
                    if group["log_moneyness_public"].notna().any()
                    else ""
                ),
            }
        )
    return pd.DataFrame(records, columns=columns)


def _strike_coverage(cleaned: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "bucket",
        "min_log_moneyness",
        "max_log_moneyness",
        "accepted_quotes",
        "calls",
        "puts",
    ]
    labels = ["deep_downside", "downside", "atm", "upside", "deep_upside"]
    if cleaned.empty:
        return pd.DataFrame(columns=columns)
    df = _option_flags(cleaned)
    df["log_moneyness_public"] = _log_moneyness(df)
    df["bucket"] = pd.cut(
        df["log_moneyness_public"],
        bins=[-float("inf"), -0.15, -0.05, 0.05, 0.15, float("inf")],
        labels=labels,
    )
    records = []
    for label in labels:
        group = df.loc[df["bucket"].astype(str) == label]
        records.append(
            {
                "bucket": label,
                "min_log_moneyness": (
                    round(float(group["log_moneyness_public"].min()), 6)
                    if not group.empty and group["log_moneyness_public"].notna().any()
                    else ""
                ),
                "max_log_moneyness": (
                    round(float(group["log_moneyness_public"].max()), 6)
                    if not group.empty and group["log_moneyness_public"].notna().any()
                    else ""
                ),
                "accepted_quotes": len(group),
                "calls": int(group["is_call_public"].sum()) if not group.empty else 0,
                "puts": int(group["is_put_public"].sum()) if not group.empty else 0,
            }
        )
    return pd.DataFrame(records, columns=columns)


def _theme(name: str) -> dict[str, str]:
    if name == "dark":
        return {
            "bg": "#111827",
            "fg": "#f9fafb",
            "muted": "#9ca3af",
            "edge": "#374151",
            "accent": "#93c5fd",
            "panel": "#1f2937",
        }
    return {
        "bg": "#ffffff",
        "fg": "#111827",
        "muted": "#4b5563",
        "edge": "#d1d5db",
        "accent": "#1d4ed8",
        "panel": "#f3f4f6",
    }


def _svg_escape(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _write_summary_card(
    output_dir: Path, public_summary: Mapping[str, Any], theme: str
) -> None:
    t = _theme(theme)
    lines = [
        ("Real-market provider evidence", 26, "accent"),
        (
            f"Underlying: {public_summary.get('underlying')}  •  As-of: {public_summary.get('asof')}",
            18,
            "fg",
        ),
        (
            f"Provider/feed: {public_summary.get('provider_label')} / {public_summary.get('feed_label')}",
            16,
            "muted",
        ),
        (
            f"Raw → normalized → accepted → selected: {public_summary.get('raw_contract_count')} → {public_summary.get('normalized_contract_count')} → {public_summary.get('accepted_quote_count')} → {public_summary.get('selected_quote_count')}",
            16,
            "fg",
        ),
        (
            f"Rejected quotes: {public_summary.get('rejected_quote_count')}  •  Warnings: {public_summary.get('warning_count')}",
            16,
            "fg",
        ),
        (f"Heston fit status: {public_summary.get('heston_fit_status')}", 18, "accent"),
        (
            "Public bundle is sanitized; raw provider payloads and full quote rows stay local/private.",
            14,
            "muted",
        ),
    ]
    y = 54
    text = []
    for value, size, key in lines:
        text.append(
            f'<text x="42" y="{y}" fill="{t[key]}" font-size="{size}" font-family="Arial, sans-serif">{_svg_escape(value)}</text>'
        )
        y += size + 18
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="980" height="340" viewBox="0 0 980 340">
  <rect width="980" height="340" rx="28" fill="{t['bg']}"/>
  <rect x="24" y="24" width="932" height="292" rx="22" fill="{t['panel']}" stroke="{t['edge']}" stroke-width="2"/>
  {''.join(text)}
</svg>
"""
    (output_dir / f"provider_evidence_summary_card.{theme}.svg").write_text(
        svg, encoding="utf-8"
    )


def _write_pipeline_flow(output_dir: Path, theme: str) -> None:
    t = _theme(theme)
    labels = ["Alpaca/FRED", "Bronze", "Silver", "Gold", "Model-ready", "Heston fit"]
    nodes = []
    arrows = []
    for index, label in enumerate(labels):
        x = 42 + index * 154
        nodes.append(
            f'<rect x="{x}" y="120" width="130" height="72" rx="14" fill="{t["panel"]}" stroke="{t["edge"]}" stroke-width="2"/><text x="{x+65}" y="162" text-anchor="middle" fill="{t["fg"]}" font-size="15" font-family="Arial, sans-serif">{label}</text>'
        )
        if index < len(labels) - 1:
            arrows.append(
                f'<path d="M {x+136} 156 L {x+154} 156" stroke="{t["accent"]}" stroke-width="3" marker-end="url(#arrow)"/>'
            )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="980" height="320" viewBox="0 0 980 320">
  <defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="{t['accent']}"/></marker></defs>
  <rect width="980" height="320" rx="24" fill="{t['bg']}"/>
  <text x="42" y="58" fill="{t['accent']}" font-size="26" font-family="Arial, sans-serif">Provider-backed evidence pipeline</text>
  <text x="42" y="88" fill="{t['muted']}" font-size="15" font-family="Arial, sans-serif">Sanitized summaries are published; raw provider payloads and full quote rows remain private.</text>
  {''.join(arrows)}
  {''.join(nodes)}
</svg>
"""
    (output_dir / f"provider_pipeline_flow.{theme}.svg").write_text(
        svg, encoding="utf-8"
    )


def _style(ax: Any, theme: str) -> dict[str, str]:
    t = _theme(theme)
    ax.figure.patch.set_facecolor(t["bg"])
    ax.set_facecolor(t["bg"])
    ax.tick_params(colors=t["fg"])
    ax.xaxis.label.set_color(t["fg"])
    ax.yaxis.label.set_color(t["fg"])
    ax.title.set_color(t["fg"])
    for spine in ax.spines.values():
        spine.set_color(t["edge"])
    ax.grid(True, alpha=0.25)
    return t


def _plot_bar(
    output_dir: Path,
    filename: str,
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    theme: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    t = _style(ax, theme)
    ax.bar(labels, values, color=t["accent"])
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:g}", ha="center", va="bottom", color=t["fg"], fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename}.{theme}.png", dpi=180)
    plt.close(fig)


def _plot_coverage(output_dir: Path, coverage: pd.DataFrame, theme: str) -> None:
    labels = (
        coverage["expiry"].astype(str).tolist() if not coverage.empty else ["no_data"]
    )
    accepted = (
        coverage["accepted_quotes"].astype(int).tolist() if not coverage.empty else [0]
    )
    calls = coverage["calls"].astype(int).tolist() if not coverage.empty else [0]
    puts = coverage["puts"].astype(int).tolist() if not coverage.empty else [0]
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    t = _style(ax, theme)
    x = list(range(len(labels)))
    ax.bar(x, accepted, label="accepted", color=t["accent"], alpha=0.75)
    ax.plot(x, calls, marker="o", label="calls")
    ax.plot(x, puts, marker="o", label="puts")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_title("Expiry and call/put coverage", loc="left")
    ax.set_ylabel("Quotes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"provider_expiry_strike_coverage.{theme}.png", dpi=180)
    plt.close(fig)


def _compact_policy_name(value: Any, *, default: str = "documented") -> str:
    if isinstance(value, Mapping):
        for key in ("policy", "name", "policy_id", "schema_version", "source"):
            candidate = value.get(key)
            if candidate not in (None, ""):
                return str(candidate)
        return default
    if value in (None, ""):
        return default
    return str(value)


def _compact_rate_details(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for source_key, public_key in (
        ("provider", "rate_provider"),
        ("rate_source", "rate_source"),
        ("series_id", "rate_series_id"),
        ("rate_observation_date", "rate_observation_date"),
        ("selected_rate", "selected_rate"),
        ("flat_rate", "flat_rate"),
        ("rate_compounding", "rate_compounding"),
    ):
        candidate = value.get(source_key)
        if candidate not in (None, "", []):
            result[public_key] = candidate
    return result


def _compact_dividend_details(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for source_key, public_key in (
        ("source", "dividend_source"),
        ("dividend_source", "dividend_source"),
        ("dividend_yield", "dividend_yield"),
        ("dividend_fallback_used", "dividend_fallback_used"),
    ):
        candidate = value.get(source_key)
        if candidate not in (None, "", []):
            result[public_key] = candidate
    return result


def _compact_provider_public_summary(
    *,
    provider_summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    counts: Mapping[str, int],
    warnings_payload: Mapping[str, Any],
    fit: pd.DataFrame,
) -> dict[str, Any]:
    rate_raw = _get_any(
        provider_summary,
        manifest,
        keys=("rate_policy", "rate_policy_name"),
        default="documented",
    )
    dividend_raw = _get_any(
        provider_summary,
        manifest,
        keys=("dividend_policy", "dividend_policy_name"),
        default="documented",
    )
    quality_raw = _get_any(
        provider_summary,
        manifest,
        keys=(
            "quality_policy_name",
            "quote_cleaning_policy",
            "option_cleaning_policy",
            "model_validation_policy",
        ),
        default="QuoteCleaningPolicyV1",
    )
    data_raw = _get_any(
        provider_summary,
        manifest,
        keys=("data_policy_name", "data_policy"),
        default="ProviderPublicEvidencePolicyV1",
    )

    selected_quote_count = _safe_int(
        _fit_value(
            fit,
            manifest,
            ("selected_quote_count", "quote_count", "n_quotes"),
            default=counts.get("selected_calibration_rows", 0),
        )
    )
    best_cost = _safe_float(
        _fit_value(
            fit,
            manifest,
            ("best_cost", "cost", "objective_value"),
            default=float("nan"),
        )
    )
    heston_fit: dict[str, Any] = {
        "status": str(_fit_value(fit, manifest, ("status",), default="unknown")),
        "objective": str(
            _fit_value(
                fit, manifest, ("objective", "objective_type"), default="unknown"
            )
        ),
        "selected_quote_count": selected_quote_count,
    }
    if not (isinstance(best_cost, float) and math.isnan(best_cost)):
        heston_fit["best_cost"] = best_cost

    payload: dict[str, Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "underlying": _get_any(
            provider_summary, manifest, keys=("underlying", "symbol"), default="unknown"
        ),
        "asof": _get_any(
            provider_summary,
            manifest,
            keys=("asof", "as_of", "valuation_timestamp_utc", "valuation_date", "date"),
            default="unknown",
        ),
        "source_run_id": _get_any(
            provider_summary,
            manifest,
            keys=("run_id", "source_run_id", "snapshot_id"),
            default="unknown",
        ),
        "provider_label": _get_any(
            provider_summary,
            manifest,
            keys=("provider_label", "provider"),
            default="provider",
        ),
        "feed_label": _get_any(
            provider_summary,
            manifest,
            keys=("feed_label", "feed"),
            default="marketdata",
        ),
        "rate_policy": _compact_policy_name(rate_raw),
        "dividend_policy": _compact_policy_name(dividend_raw),
        "quality_policy_name": _compact_policy_name(
            quality_raw, default="QuoteCleaningPolicyV1"
        ),
        "data_policy_name": _compact_policy_name(
            data_raw, default="ProviderPublicEvidencePolicyV1"
        ),
        "raw_contract_count": counts.get("raw_contracts", 0),
        "normalized_contract_count": counts.get("normalized_contracts", 0),
        "accepted_quote_count": counts.get("accepted_quotes", 0),
        "rejected_quote_count": counts.get("rejected_quotes", 0),
        "selected_quote_count": counts.get(
            "selected_calibration_rows", selected_quote_count
        ),
        "warning_count": _warning_count(warnings_payload),
        "heston_fit_status": heston_fit["status"],
        "heston_fit": heston_fit,
        "caveats": list(CAVEATS),
    }
    payload.update(_compact_rate_details(rate_raw))
    payload.update(_compact_dividend_details(dividend_raw))
    return payload


def build_provider_evidence(
    *,
    bundle_root: Path,
    provider_summary_json: Path,
    output_dir: Path,
    profile: str,
    existing_fit_summary_csv: Path | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    provider_summary_raw = _read_json(provider_summary_json)
    provider_summary = _sanitize_json_public(provider_summary_raw)
    manifest = _read_json(_first_existing(bundle_root, ("manifest.json",)))
    warnings_payload = _sanitize_json_public(
        _read_json(
            _first_existing(bundle_root, ("warnings.json", "provider_warnings.json"))
        )
    )
    cleaned = _read_frame(bundle_root, "cleaned_quotes")
    rejected = _read_frame(bundle_root, "rejected_quotes")
    heston = _read_frame(bundle_root, "heston_quotes")
    fit = _fit_source(bundle_root, existing_fit_summary_csv)

    quote_summary = _quote_summary(
        provider_summary, manifest, cleaned, rejected, heston, fit
    )
    rejection_counts = _rejection_counts(manifest, rejected)
    expiry_coverage = _expiry_coverage(cleaned)
    strike_coverage = _strike_coverage(cleaned)

    counts = {
        str(row["stage"]): int(row["output_count"])
        for row in quote_summary.to_dict("records")
    }
    public_summary = _compact_provider_public_summary(
        provider_summary=provider_summary,
        manifest=manifest,
        counts=counts,
        warnings_payload=warnings_payload,
        fit=fit,
    )
    _assert_no_forbidden_public_keys(
        public_summary, where="provider_public_summary.json"
    )

    selected_quote_count = _safe_int(
        _fit_value(
            fit,
            manifest,
            ("selected_quote_count", "quote_count", "n_quotes"),
            default=len(heston) or len(cleaned),
        )
    )
    expiry_col = _expiry_column(heston) or _expiry_column(cleaned)
    model_frame = heston if not heston.empty else cleaned
    right_col = _right_column(model_frame)
    right = (
        model_frame[right_col].fillna("").astype(str).str.upper()
        if right_col
        else pd.Series(dtype=str)
    )
    model_ready = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "ready" if selected_quote_count else "not_ready",
        "candidate_heston_rows": (
            int(len(heston)) if not heston.empty else int(len(cleaned))
        ),
        "selected_quotes": selected_quote_count,
        "rejected_quotes": int(len(rejected)),
        "expiry_count": (
            int(model_frame[expiry_col].nunique(dropna=True)) if expiry_col else 0
        ),
        "call_count": (
            int((right.str.startswith("C") | right.eq("CALL")).sum())
            if not right.empty
            else 0
        ),
        "put_count": (
            int((right.str.startswith("P") | right.eq("PUT")).sum())
            if not right.empty
            else 0
        ),
        "preflight_status": _get(
            manifest,
            "preflight_status",
            default="ready" if selected_quote_count else "not_ready",
        ),
        "objective": str(
            _fit_value(
                fit, manifest, ("objective", "objective_type"), default="unknown"
            )
        ),
        "warning_count": _warning_count(warnings_payload),
    }

    heston_fit = pd.DataFrame(
        [
            {
                "status": str(
                    _fit_value(fit, manifest, ("status",), default="unknown")
                ),
                "objective": str(
                    _fit_value(
                        fit,
                        manifest,
                        ("objective", "objective_type"),
                        default="unknown",
                    )
                ),
                "selected_quote_count": selected_quote_count,
                "best_cost": _safe_float(
                    _fit_value(
                        fit,
                        manifest,
                        ("best_cost", "cost", "objective_value"),
                        default=float("nan"),
                    )
                ),
                "kappa": _safe_float(
                    _fit_value(fit, manifest, ("kappa",), default=float("nan"))
                ),
                "vbar": _safe_float(
                    _fit_value(fit, manifest, ("vbar", "theta"), default=float("nan"))
                ),
                "eta": _safe_float(
                    _fit_value(fit, manifest, ("eta", "sigma"), default=float("nan"))
                ),
                "rho": _safe_float(
                    _fit_value(fit, manifest, ("rho",), default=float("nan"))
                ),
                "v": _safe_float(
                    _fit_value(fit, manifest, ("v", "v0"), default=float("nan"))
                ),
                "warning_count": _warning_count(warnings_payload),
            }
        ]
    )
    parameter_summary = pd.DataFrame(
        [
            {"parameter": p, "value": heston_fit[p].iloc[0]}
            for p in ("kappa", "vbar", "eta", "rho", "v")
            if pd.notna(heston_fit[p].iloc[0])
        ]
    )

    json_payloads = {
        "provider_public_summary.json": public_summary,
        "provider_run_metadata.json": {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "underlying": public_summary["underlying"],
            "asof": public_summary["asof"],
            "source_run_id": public_summary["source_run_id"],
            "provider_label": public_summary["provider_label"],
            "feed_label": public_summary["feed_label"],
            "bundle_root": str(bundle_root),
            "provider_summary_json": str(provider_summary_json),
        },
        "provider_quality_policy.json": {
            "name": _get(
                provider_summary, "quality_policy_name", default="QuoteCleaningPolicyV1"
            ),
            "rejection_handling": "Rejected rows stay local/private; public bundle publishes reason counts only.",
        },
        "provider_data_policy.json": {
            "name": _get(
                provider_summary,
                "data_policy_name",
                default="ProviderPublicEvidencePolicyV1",
            ),
            "publish": [
                "labels",
                "counts",
                "reason counts",
                "coverage summaries",
                "policy names",
                "status",
                "warnings",
            ],
            "keep_local_private": [
                "raw provider response bodies",
                "full provider-derived quote rows",
                "credentials",
            ],
        },
        "provider_model_ready_summary.json": model_ready,
        "provider_warnings.json": warnings_payload,
    }
    for filename, payload in json_payloads.items():
        _assert_no_forbidden_public_keys(payload, where=filename)
        _write_json(data_dir / filename, payload)

    csv_payloads = {
        "provider_quote_cleaning_summary.csv": quote_summary,
        "provider_rejection_reason_counts.csv": rejection_counts,
        "provider_expiry_coverage.csv": expiry_coverage,
        "provider_strike_moneyness_coverage.csv": strike_coverage,
        "provider_heston_fit_summary.csv": heston_fit,
        "provider_heston_parameter_summary.csv": parameter_summary,
    }
    for filename, frame in csv_payloads.items():
        forbidden = [
            column
            for column in frame.columns
            if str(column).lower() not in ALLOWED_PUBLIC_COUNT_COLUMNS
            and any(
                fragment in str(column).lower()
                for fragment in FORBIDDEN_PUBLIC_KEY_FRAGMENTS
            )
        ]
        if forbidden:
            raise ValueError(f"Forbidden public columns in {filename}: {forbidden}")
        frame.to_csv(data_dir / filename, index=False)

    for theme in ("light", "dark"):
        _write_summary_card(output_dir, public_summary, theme)
        _write_pipeline_flow(output_dir, theme)
        _plot_bar(
            output_dir,
            "provider_quote_cleaning_waterfall",
            quote_summary["stage"].astype(str).tolist(),
            quote_summary["output_count"].astype(float).tolist(),
            "Provider quote-cleaning funnel",
            "Rows/contracts",
            theme,
        )
        _plot_coverage(output_dir, expiry_coverage, theme)
        param_labels = (
            parameter_summary["parameter"].astype(str).tolist()
            if not parameter_summary.empty
            else ["no_params"]
        )
        param_values = (
            parameter_summary["value"].astype(float).tolist()
            if not parameter_summary.empty
            else [0.0]
        )
        _plot_bar(
            output_dir,
            "provider_heston_fit_summary",
            param_labels,
            param_values,
            "Provider Heston fit parameter summary",
            "Parameter value",
            theme,
        )

    manifest_payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "profile": profile,
        "source_run_id": public_summary["source_run_id"],
        "underlying": public_summary["underlying"],
        "asof": public_summary["asof"],
        "input_references": {
            "bundle_root": str(bundle_root),
            "provider_summary_json": str(provider_summary_json),
            "existing_fit_summary_csv": (
                str(existing_fit_summary_csv) if existing_fit_summary_csv else None
            ),
        },
        "rebuild_command": REBUILD_COMMAND_STUB,
        "caveats": list(CAVEATS),
        "artifacts": [
            *[
                {
                    "filename": filename,
                    "path": f"data/{filename}",
                    "artifact_type": "data",
                }
                for filename in PUBLIC_DATA_FILES
            ],
            *[
                {"filename": filename, "path": filename, "artifact_type": "figure"}
                for filename in PUBLIC_FIGURE_FILES
            ],
        ],
    }
    _assert_no_forbidden_public_keys(
        manifest_payload, where="provider_evidence_manifest.json"
    )
    _write_json(data_dir / "provider_evidence_manifest.json", manifest_payload)
    print(data_dir / "provider_evidence_manifest.json")
    return data_dir / "provider_evidence_manifest.json"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sanitized provider evidence docs artifacts."
    )
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--provider-summary-json", required=True, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--profile", default="release", choices=("smoke", "release"))
    parser.add_argument("--existing-fit-summary-csv", default=None, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    build_provider_evidence(
        bundle_root=args.bundle_root,
        provider_summary_json=args.provider_summary_json,
        output_dir=args.output_dir,
        profile=args.profile,
        existing_fit_summary_csv=args.existing_fit_summary_csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
