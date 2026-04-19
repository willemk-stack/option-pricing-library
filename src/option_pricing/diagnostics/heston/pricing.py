"""Notebook-facing Step 1 packaging helpers for Heston slice diagnostics.

These entrypoints intentionally stop at packaging and contract-freezing:

- they package existing low-level diagnostics output
- they validate canonical notebook-facing columns
- they do not add new pricing logic, warning heuristics, or plotting objects

Later sprint steps may extend the orchestration that feeds these helpers without
changing the public return artifacts frozen here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ...models.heston.fourier import (
    HestonIntegralBatchDiagnostics,
    HestonIntegralDiagnostics,
)
from .contracts import HESTON_SLICE_TABLE_COLUMNS
from .models import (
    HestonBackendComparisonDiagnostics,
    HestonPriceSliceDiagnostics,
    HestonProbabilitySliceDiagnostics,
)

type ProbabilityDiagnosticsInput = (
    HestonIntegralBatchDiagnostics | HestonIntegralDiagnostics
)


def _coerce_columns(columns: Mapping[str, Any], *, label: str) -> pd.DataFrame:
    if not columns:
        raise ValueError(f"{label} must contain at least one column.")
    return pd.DataFrame(
        {
            str(name): np.atleast_1d(np.asarray(values))
            for name, values in columns.items()
        }
    )


def _coerce_optional_strike(
    strike: Any,
    *,
    size: int,
) -> np.ndarray | None:
    if strike is None:
        return None

    strike_arr = np.asarray(strike, dtype=np.float64).reshape(-1)
    if strike_arr.size == 1:
        strike_arr = np.repeat(strike_arr, size)
    if strike_arr.size != size:
        raise ValueError(
            "strike must be scalar or have the same flattened size as the "
            "probability diagnostics slice."
        )
    return strike_arr


def _warning_count(flags: np.ndarray | None, *, size: int) -> np.ndarray:
    if flags is None:
        return np.zeros(size, dtype=np.int64)

    flags_arr = np.asarray(flags, dtype=np.uint32).reshape(-1)
    return np.asarray(
        [int(int(flag).bit_count()) for flag in flags_arr], dtype=np.int64
    )


def _probability_diag_to_table(
    diagnostics: ProbabilityDiagnosticsInput,
    *,
    strike: Any = None,
) -> pd.DataFrame:
    x = np.asarray(diagnostics.x, dtype=np.float64).reshape(-1)
    size = x.size

    data: dict[str, Any] = {
        "log_moneyness": x,
        "probability": np.asarray(diagnostics.probability, dtype=np.float64).reshape(
            -1
        ),
        "warning_count": _warning_count(
            getattr(diagnostics, "warning_flags", None),
            size=size,
        ),
        "tail_fraction": (
            None
            if diagnostics.tail_abs_fraction is None
            else np.asarray(diagnostics.tail_abs_fraction, dtype=np.float64).reshape(-1)
        ),
        "cancellation_ratio": (
            None
            if diagnostics.cancellation_ratio is None
            else np.asarray(diagnostics.cancellation_ratio, dtype=np.float64).reshape(
                -1
            )
        ),
        "warning_flags": (
            None
            if getattr(diagnostics, "warning_flags", None) is None
            else np.asarray(diagnostics.warning_flags, dtype=np.uint32).reshape(-1)
        ),
        "backend": np.full(size, diagnostics.backend, dtype=object),
        "probability_index": np.full(size, diagnostics.j, dtype=np.int64),
    }

    strike_arr = _coerce_optional_strike(strike, size=size)
    if strike_arr is not None:
        data["strike"] = strike_arr

    quad_error = getattr(diagnostics, "quad_error_estimate", None)
    if quad_error is not None:
        data["quad_error_estimate"] = np.asarray(quad_error, dtype=np.float64).reshape(
            -1
        )

    return pd.DataFrame(data)


def _probability_diag_to_arrays(
    diagnostics: ProbabilityDiagnosticsInput,
) -> dict[str, Any]:
    arrays: dict[str, Any] = {
        "log_moneyness": np.asarray(diagnostics.x, dtype=np.float64),
        "probability": np.asarray(diagnostics.probability, dtype=np.float64),
        "total_integral": np.asarray(diagnostics.total_integral, dtype=np.float64),
    }

    if diagnostics.panel_edges is not None:
        arrays["panel_edges"] = np.asarray(diagnostics.panel_edges, dtype=np.float64)
    if diagnostics.panel_contribs is not None:
        arrays["panel_contribs"] = np.asarray(
            diagnostics.panel_contribs, dtype=np.float64
        )
    if diagnostics.panel_invalid is not None:
        arrays["panel_invalid"] = np.asarray(diagnostics.panel_invalid, dtype=np.bool_)
    if diagnostics.panel_reason is not None:
        arrays["panel_reason"] = np.asarray(diagnostics.panel_reason, dtype=np.uint32)
    if diagnostics.quad_error_estimate is not None:
        arrays["quad_error_estimate"] = np.asarray(
            diagnostics.quad_error_estimate,
            dtype=np.float64,
        )
    if diagnostics.tail_abs_fraction is not None:
        arrays["tail_fraction"] = np.asarray(
            diagnostics.tail_abs_fraction,
            dtype=np.float64,
        )
    if diagnostics.cancellation_ratio is not None:
        arrays["cancellation_ratio"] = np.asarray(
            diagnostics.cancellation_ratio,
            dtype=np.float64,
        )
    if getattr(diagnostics, "warning_flags", None) is not None:
        arrays["warning_flags"] = np.asarray(diagnostics.warning_flags, dtype=np.uint32)

    return arrays


def probability_slice_with_diagnostics(
    *,
    probability_diagnostics: ProbabilityDiagnosticsInput | None = None,
    probability_table: pd.DataFrame | None = None,
    probability_columns: Mapping[str, Any] | None = None,
    strike: Any = None,
    meta: Mapping[str, Any] | None = None,
    arrays: Mapping[str, Any] | None = None,
) -> HestonProbabilitySliceDiagnostics:
    """Package a probability slice into a notebook-facing diagnostics artifact.

    Step 1 scope
    ------------
    This helper freezes the artifact contract without introducing new numerical
    orchestration. Callers may either:

    - provide existing low-level Fourier diagnostics from
      ``option_pricing.models.heston.fourier``, or
    - provide an already-assembled probability table/columns mapping.

    Later sprint steps may add convenience wrappers around existing Heston
    engines, but they must preserve this return-model contract.
    """

    sources = [
        probability_diagnostics is not None,
        probability_table is not None,
        probability_columns is not None,
    ]
    if sum(sources) == 0:
        raise ValueError(
            "Provide probability_diagnostics, probability_table, or "
            "probability_columns."
        )
    if probability_table is not None and probability_columns is not None:
        raise ValueError(
            "Pass either probability_table or probability_columns, not both."
        )

    if probability_table is not None:
        table = probability_table.copy()
    elif probability_columns is not None:
        table = _coerce_columns(
            probability_columns,
            label="probability_columns",
        )
    else:
        assert probability_diagnostics is not None
        table = _probability_diag_to_table(
            probability_diagnostics,
            strike=strike,
        )

    packaged_arrays = {} if arrays is None else dict(arrays)
    packaged_meta = {} if meta is None else dict(meta)

    if probability_diagnostics is not None:
        packaged_arrays = {
            **_probability_diag_to_arrays(probability_diagnostics),
            **packaged_arrays,
        }
        packaged_meta.setdefault("backend", probability_diagnostics.backend)
        packaged_meta.setdefault("probability_index", int(probability_diagnostics.j))
        packaged_meta.setdefault("tau", float(probability_diagnostics.tau))
        packaged_meta.setdefault("source", "option_pricing.models.heston.fourier")

    return HestonProbabilitySliceDiagnostics(
        meta=packaged_meta,
        table=table,
        arrays=packaged_arrays,
    )


def price_slice_with_diagnostics(
    *,
    slice_table: pd.DataFrame | None = None,
    slice_columns: Mapping[str, Any] | None = None,
    meta: Mapping[str, Any] | None = None,
    arrays: Mapping[str, Any] | None = None,
) -> HestonPriceSliceDiagnostics:
    """Freeze the canonical notebook-facing slice-table contract.

    The Step 1 slice artifact is intentionally data-oriented: callers provide
    either an assembled ``DataFrame`` or explicit columns already mapped into
    the canonical verbose names frozen for notebook use.

    Plotting code is expected to consume only the returned tables and arrays,
    and must not recompute diagnostics.
    """

    if slice_table is None and slice_columns is None:
        raise ValueError("Provide slice_table or slice_columns.")
    if slice_table is not None and slice_columns is not None:
        raise ValueError("Pass either slice_table or slice_columns, not both.")

    if slice_table is not None:
        table = slice_table.copy()
    else:
        assert slice_columns is not None
        table = _coerce_columns(slice_columns, label="slice_columns")

    missing = [name for name in HESTON_SLICE_TABLE_COLUMNS if name not in table.columns]
    if missing:
        raise ValueError(
            "slice diagnostics must expose the canonical slice columns: "
            f"{', '.join(missing)}."
        )

    return HestonPriceSliceDiagnostics(
        meta={} if meta is None else dict(meta),
        table=table,
        arrays={} if arrays is None else dict(arrays),
    )


def compare_backend_slice(
    *,
    comparison_table: pd.DataFrame | None = None,
    comparison_columns: Mapping[str, Any] | None = None,
    meta: Mapping[str, Any] | None = None,
    arrays: Mapping[str, Any] | None = None,
) -> HestonBackendComparisonDiagnostics:
    """Package backend-comparison data for a strike slice.

    Step 1 freezes only the public responsibility and return artifact here.
    The comparison table should primarily represent price differences across
    backends. Any direct numerical orchestration remains for later sprint steps.
    """

    if comparison_table is None and comparison_columns is None:
        raise ValueError("Provide comparison_table or comparison_columns.")
    if comparison_table is not None and comparison_columns is not None:
        raise ValueError(
            "Pass either comparison_table or comparison_columns, not both."
        )

    if comparison_table is not None:
        table = comparison_table.copy()
    else:
        assert comparison_columns is not None
        table = _coerce_columns(
            comparison_columns,
            label="comparison_columns",
        )

    return HestonBackendComparisonDiagnostics(
        meta={} if meta is None else dict(meta),
        table=table,
        arrays={} if arrays is None else dict(arrays),
    )
