"""Calibration quote policy helpers for Heston diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ...models.heston.fourier import (
    HestonIntegralWarning,
    heston_warning_calibration_action,
)

_ACTION_RANK = {"ok": 0, "review": 1, "quarantine": 2, "block": 3}

_WARNING_LABELS: dict[HestonIntegralWarning, str] = {
    HestonIntegralWarning.NONFINITE_TOTAL: "non-finite integral",
    HestonIntegralWarning.NONFINITE_PROBABILITY: "non-finite probability",
    HestonIntegralWarning.PROBABILITY_OUT_OF_RANGE: "probability outside [0, 1]",
    HestonIntegralWarning.LARGE_TAIL_FRACTION: "high tail fraction",
    HestonIntegralWarning.EXCESSIVE_CANCELLATION: "cancellation warning",
    HestonIntegralWarning.TOO_MANY_BAD_PANELS: "panel warning",
    HestonIntegralWarning.QUAD_ERROR_LARGE: "quad error warning",
}


def _warning_reason_labels(flag_value: int) -> list[str]:
    labels: list[str] = []
    for warning, label in _WARNING_LABELS.items():
        if int(flag_value) & int(warning):
            labels.append(label)
    return labels


def _join(labels: list[str]) -> str:
    return "none" if not labels else "; ".join(labels)


def _coerce_quote_diagnostics(
    quote_diagnostics: pd.DataFrame | Mapping[str, Any] | None,
) -> pd.DataFrame:
    if quote_diagnostics is None:
        return pd.DataFrame()
    if isinstance(quote_diagnostics, pd.DataFrame):
        return quote_diagnostics.copy()
    return pd.DataFrame(dict(quote_diagnostics))


def heston_calibration_quote_policy_tables(
    *,
    n_quotes: int,
    quote_diagnostics: pd.DataFrame | Mapping[str, Any] | None = None,
    fit_used_filtered_quotes: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build quote-level block/quarantine/review policy tables.

    ``quote_diagnostics`` may contain ``quote_index``, ``warning_flags``, and
    ``persistent_backend_disagreement`` columns. Missing diagnostics are treated
    as clean, but the returned metadata records that no filtering was applied
    by this helper.
    """
    n = int(n_quotes)
    if n < 0:
        raise ValueError("n_quotes must be nonnegative.")

    diagnostics = _coerce_quote_diagnostics(quote_diagnostics)
    if diagnostics.empty:
        diagnostics = pd.DataFrame(
            {
                "quote_index": np.arange(n, dtype=np.int64),
                "warning_flags": np.zeros(n, dtype=np.uint32),
                "persistent_backend_disagreement": np.zeros(n, dtype=np.bool_),
            }
        )
    else:
        if "quote_index" not in diagnostics.columns:
            diagnostics.insert(0, "quote_index", np.arange(len(diagnostics)))
        diagnostics["quote_index"] = pd.to_numeric(
            diagnostics["quote_index"], errors="raise"
        ).astype(np.int64)
        if "warning_flags" not in diagnostics.columns:
            diagnostics["warning_flags"] = np.uint32(0)
        if "persistent_backend_disagreement" not in diagnostics.columns:
            diagnostics["persistent_backend_disagreement"] = False

    if diagnostics["quote_index"].duplicated().any():
        raise ValueError("quote_diagnostics contains duplicate quote_index values.")
    if (diagnostics["quote_index"] < 0).any() or (
        diagnostics["quote_index"] >= n
    ).any():
        raise ValueError("quote_diagnostics quote_index values are out of range.")

    base = pd.DataFrame({"quote_index": np.arange(n, dtype=np.int64)})
    merged = base.merge(diagnostics, on="quote_index", how="left")
    flags = merged["warning_flags"].fillna(0).to_numpy(dtype=np.uint32)
    persistent_backend = np.asarray(
        [
            False if pd.isna(value) else bool(value)
            for value in merged["persistent_backend_disagreement"]
        ],
        dtype=np.bool_,
    )

    rows: list[dict[str, Any]] = []
    for quote_index, flag, backend_disagreement in zip(
        np.arange(n, dtype=np.int64),
        flags,
        persistent_backend,
        strict=True,
    ):
        action = heston_warning_calibration_action(int(flag))
        reasons = _warning_reason_labels(int(flag))
        if bool(backend_disagreement):
            reasons.append("persistent backend disagreement")
            if _ACTION_RANK[action] < _ACTION_RANK["quarantine"]:
                action = "quarantine"
        rows.append(
            {
                "quote_index": int(quote_index),
                "warning_flags": int(flag),
                "persistent_backend_disagreement": bool(backend_disagreement),
                "calibration_action": action,
                "calibration_action_rank": int(_ACTION_RANK[action]),
                "reason_labels": _join(reasons),
                "fit_used_quote": bool(
                    True
                    if not fit_used_filtered_quotes
                    else action not in {"block", "quarantine"}
                ),
            }
        )

    quote_policy = pd.DataFrame(rows)
    summary = (
        quote_policy.groupby("calibration_action", dropna=False)
        .size()
        .reset_index(name="count")
    )
    for action_name in _ACTION_RANK:
        if action_name not in set(summary["calibration_action"]):
            summary = pd.concat(
                [
                    summary,
                    pd.DataFrame(
                        [{"calibration_action": action_name, "count": 0}],
                    ),
                ],
                ignore_index=True,
            )
    summary["calibration_action_rank"] = summary["calibration_action"].map(_ACTION_RANK)
    summary = summary.sort_values(
        ["calibration_action_rank", "calibration_action"],
        ascending=[False, True],
        ignore_index=True,
    )

    reason_rows: list[dict[str, Any]] = []
    exploded = quote_policy.loc[
        quote_policy["reason_labels"].astype(str) != "none"
    ].copy()
    for reason in sorted(
        {
            reason
            for labels in exploded["reason_labels"].astype(str)
            for reason in labels.split("; ")
            if reason
        }
    ):
        mask = exploded["reason_labels"].astype(str).str.contains(reason, regex=False)
        affected = exploded.loc[mask]
        reason_rows.append(
            {
                "reason_label": reason,
                "count": int(len(affected)),
                "max_calibration_action": str(
                    affected.sort_values("calibration_action_rank", ascending=False)[
                        "calibration_action"
                    ].iloc[0]
                ),
            }
        )
    reason_summary = pd.DataFrame(
        reason_rows,
        columns=["reason_label", "count", "max_calibration_action"],
    )

    policy_summary = pd.concat(
        [
            summary.assign(summary_kind="action"),
            reason_summary.assign(summary_kind="reason"),
        ],
        ignore_index=True,
        sort=False,
    )

    block_count = int((quote_policy["calibration_action"] == "block").sum())
    quarantine_count = int((quote_policy["calibration_action"] == "quarantine").sum())
    review_count = int((quote_policy["calibration_action"] == "review").sum())
    meta = {
        "quote_policy_available": quote_diagnostics is not None,
        "fit_used_filtered_quotes": bool(fit_used_filtered_quotes),
        "blocked_quote_count": block_count,
        "quarantined_quote_count": quarantine_count,
        "review_quote_count": review_count,
        "quote_filtering_policy": (
            "filtered_blocked_and_quarantined"
            if fit_used_filtered_quotes
            else "not_applied_by_this_diagnostic_all_supplied_quotes_evaluated"
        ),
    }

    return quote_policy, policy_summary, meta


__all__ = ["heston_calibration_quote_policy_tables"]
