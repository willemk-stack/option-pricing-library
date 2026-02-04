from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import pandas as pd


def to_frame(items: Sequence[object]) -> pd.DataFrame:
    """Coerce a sequence of dicts or dataclasses into a DataFrame."""

    rows: list[dict[str, Any]] = []
    for it in items:
        if isinstance(it, dict):
            rows.append(dict(it))
        elif is_dataclass(it) and not isinstance(it, type):
            rows.append(asdict(it))
        else:
            raise TypeError(f"Unsupported item type: {type(it)}")
    return pd.DataFrame(rows)


def add_decision_columns(
    df: pd.DataFrame,
    *,
    tol_abs: float = 1e-4,
    tol_rel: float = 1e-4,
    budget_ms: float | None = None,
    abs_err_col: str = "abs_err",
    rel_err_col: str = "rel_err",
    runtime_col: str = "runtime_ms",
) -> pd.DataFrame:
    """Add notebook-friendly decision columns.

    Adds:
      - feasible (abs+rel)
      - within_budget
      - feasible_and_within_budget
      - score (runtime for feasible rows else NaN)
    """

    d = df.copy()

    abs_err = d[abs_err_col].astype(float)
    rel_err = d[rel_err_col].astype(float)

    d["feasible"] = (abs_err <= float(tol_abs)) & (rel_err <= float(tol_rel))

    if budget_ms is None:
        d["within_budget"] = True
    else:
        d["within_budget"] = d[runtime_col].astype(float) <= float(budget_ms)

    d["feasible_and_within_budget"] = d["feasible"] & d["within_budget"]

    d["score"] = np.where(
        d["feasible_and_within_budget"],
        d[runtime_col].astype(float),
        np.nan,
    )
    return d


def best_feasible_row(
    df: pd.DataFrame,
    *,
    runtime_col: str = "runtime_ms",
    feasible_col: str = "feasible_and_within_budget",
) -> int | None:
    """Index of fastest feasible row, else None."""

    if feasible_col not in df.columns:
        return None
    dd = df[df[feasible_col].astype(bool)]
    if dd.empty:
        return None
    return int(dd[runtime_col].astype(float).idxmin())


def pareto_frontier(
    df: pd.DataFrame,
    *,
    x_col: str = "runtime_ms",
    y_col: str = "abs_err",
) -> pd.DataFrame:
    """Nondominated set minimizing (x_col, y_col)."""

    if df.empty:
        return df.copy()

    d = df.copy()
    x = d[x_col].astype(float).to_numpy()
    y = d[y_col].astype(float).to_numpy()

    order = np.argsort(x)
    y_sorted = y[order]

    best_y = np.inf
    keep = np.zeros_like(y_sorted, dtype=bool)
    for i, yi in enumerate(y_sorted):
        if yi <= best_y:
            keep[i] = True
            best_y = yi

    pos = order[keep].tolist()
    return d.take(pos).copy()


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Optional: consistent notebook-friendly column ordering."""

    preferred = [
        "contract",
        "case",
        "kind",
        "payout",
        "tau",
        "spot",
        "strike",
        "sigma",
        "r",
        "q",
        "sweep",
        "coord",
        "method",
        "advection",
        "spacing",
        "Nx",
        "Nt",
        "ic_remedy",
        "runtime_ms",
        # benchmarks
        "bs",
        "analytic",
        "pde",
        "err",
        "abs_err",
        "rel_err",
        "x0",
        # decisioning
        "feasible",
        "within_budget",
        "feasible_and_within_budget",
        "score",
        # failure fields
        "status",
        "error",
    ]
    cols = [c for c in preferred if c in df.columns] + [
        c for c in df.columns if c not in preferred
    ]
    return df[cols]


__all__ = [
    "to_frame",
    "add_decision_columns",
    "best_feasible_row",
    "pareto_frontier",
    "order_columns",
]
