"""High-level orchestration for vol-surface diagnostics.

The low-level building blocks live in :mod:`option_pricing.diagnostics.vol_surface.compute`.
This module provides a notebook-friendly *one-call* runner that returns a
structured report object containing tables + (optional) grids.

Design goals
------------
* **Pure diagnostics**: no pricing logic changes; uses existing no-arb checks.
* **Notebook ergonomics**: returned tables are ready to ``display(df)``.
* **Serialization**: reports can be exported as JSON without dumping huge arrays.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import numpy as np
import pandas as pd

from option_pricing.typing import ScalarFn
from option_pricing.vol.arbitrage import check_surface_noarb
from option_pricing.vol.surface import VolSurface

from .compute import (
    VolSurfaceLike,
    calendar_dW,
    calendar_summary,
    noarb_smile_table,
    noarb_worst_points,
    surface_domain_report,
    surface_points_df,
    svi_fit_table,
)


def _array_summary(a: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a)
    out: dict[str, Any] = {
        "shape": tuple(int(x) for x in a.shape),
        "dtype": str(a.dtype),
    }
    if a.size == 0:
        return out

    if np.issubdtype(a.dtype, np.number):
        aa = a.astype(float, copy=False)
        finite = np.isfinite(aa)
        out["finite_frac"] = float(np.mean(finite))
        if np.any(finite):
            aff = aa[finite]
            out.update(
                {
                    "min": float(np.min(aff)),
                    "max": float(np.max(aff)),
                    "mean": float(np.mean(aff)),
                }
            )
    return out


def _df_to_jsonable(df: pd.DataFrame, *, max_rows: int = 500) -> dict[str, Any]:
    df2 = df.head(int(max_rows))
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "records": df2.replace({np.nan: None}).to_dict(orient="records"),
        "truncated": bool(df.shape[0] > df2.shape[0]),
    }


def _json_default(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _array_summary(o)
    if hasattr(o, "isoformat"):
        try:
            return o.isoformat()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


@dataclass(frozen=True)
class SurfaceDiagnosticsReport:
    """Notebook-friendly bundle of diagnostics outputs."""

    meta: dict[str, Any]
    tables: dict[str, pd.DataFrame]
    arrays: dict[str, np.ndarray]

    def to_dict(
        self,
        *,
        include_arrays: bool = False,
        max_table_rows: int = 500,
        max_array_elems: int = 2_000,
    ) -> dict[str, Any]:
        """Convert report to a JSON-friendly dict.

        By default, arrays are summarized instead of embedded.
        """

        tables = {
            k: _df_to_jsonable(v, max_rows=max_table_rows)
            for k, v in self.tables.items()
        }

        if include_arrays:
            arrays: dict[str, Any] = {}
            for k, a in self.arrays.items():
                a = np.asarray(a)
                if a.size <= int(max_array_elems):
                    arrays[k] = a.tolist()
                else:
                    arrays[k] = _array_summary(a)
            arrays_mode = "embedded"
        else:
            arrays = {k: _array_summary(v) for k, v in self.arrays.items()}
            arrays_mode = "summaries"

        return {
            "meta": dict(self.meta),
            "tables": tables,
            "arrays": arrays,
            "arrays_mode": arrays_mode,
        }

    def to_json(self, **kwargs: Any) -> str:
        payload = self.to_dict(
            include_arrays=bool(kwargs.pop("include_arrays", False)),
            max_table_rows=int(kwargs.pop("max_table_rows", 500)),
            max_array_elems=int(kwargs.pop("max_array_elems", 2_000)),
        )
        return json.dumps(payload, default=_json_default, **kwargs)


def run_surface_diagnostics(
    surface: VolSurfaceLike,
    *,
    forward: ScalarFn,
    df: ScalarFn,
    quotes_df: pd.DataFrame | None = None,
    x_grid: np.ndarray | None = None,
    include_surface_points: bool = False,
    include_svi: bool = True,
    include_domain: bool = True,
    include_calendar_arrays: bool = False,
    top_n: int = 10,
) -> SurfaceDiagnosticsReport:
    """Run a compact suite of vol-surface diagnostics.

    This is intended as the *single entry point* for notebook demos.
    """

    rep = check_surface_noarb(cast(VolSurface, surface), df=df)
    tables: dict[str, pd.DataFrame] = {}
    tables["noarb_smiles"] = noarb_smile_table(rep)

    worst = noarb_worst_points(surface, rep, forward=forward, df=df, top_n=int(top_n))
    tables["noarb_worst_monotonicity"] = worst.monotonicity
    tables["noarb_worst_convexity"] = worst.convexity
    tables["noarb_worst_calendar"] = worst.calendar
    tables["calendar_summary"] = pd.DataFrame([calendar_summary(rep)])

    if include_svi:
        tables["svi_fit"] = svi_fit_table(surface)

    if include_domain:
        tables["domain"] = surface_domain_report(
            surface, quotes_df=quotes_df, forward=forward
        )

    if include_surface_points:
        tables["surface_points"] = surface_points_df(surface, forward=forward)

    arrays: dict[str, np.ndarray] = {}
    if include_calendar_arrays:
        if x_grid is None:
            x_grid = getattr(
                getattr(rep, "calendar_total_variance", None), "x_grid", None
            )
        if x_grid is not None:
            xg = np.asarray(x_grid, dtype=float)
            arrays["calendar_x_grid"] = xg
            arrays["calendar_dW"] = calendar_dW(surface, x_grid=xg)

    meta: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "report_ok": bool(getattr(rep, "ok", True)),
        "report_message": str(getattr(rep, "message", "")),
        "noarb_worst_summary": worst.summary,
        "noarb_suggestions": list(worst.suggestions),
    }

    return SurfaceDiagnosticsReport(meta=meta, tables=tables, arrays=arrays)
