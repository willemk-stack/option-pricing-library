"""Pure plotting helpers for notebook-facing Heston diagnostics.

These helpers consume already-built Step 1-3 tables and arrays only.
They intentionally do not call pricing or integration routines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from .._mpl import get_plt, pretty_ax, require_columns
from .models import HestonDiagnosticsReport
from .monte_carlo import summarize_bias_vs_timestep, summarize_runtime_vs_error

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

type SliceTableSource = HestonDiagnosticsReport | pd.DataFrame
type MappingSource = HestonDiagnosticsReport | Mapping[str, Any]

_WARNING_SEVERITY_RANK: dict[str, int] = {
    "ok": 0,
    "info": 1,
    "warning": 2,
    "severe": 3,
    "critical": 4,
}


def _ensure_ax(
    ax: Axes | None,
    *,
    figsize: tuple[float, float],
) -> tuple[Figure, Axes]:
    plt = get_plt()
    if ax is not None:
        return cast("Figure", ax.figure), ax
    fig, created_ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    return fig, created_ax


def _severity_rank(value: Any) -> int:
    return _WARNING_SEVERITY_RANK.get(str(value), -1)


def _note_axis(ax: Axes, message: str) -> None:
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
    )


def _status_note(ax: Axes, message: str) -> None:
    ax.text(
        0.02,
        0.98,
        message,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )


def _slice_table(source: SliceTableSource) -> pd.DataFrame:
    if isinstance(source, HestonDiagnosticsReport):
        return source.tables["slice"].copy()
    if isinstance(source, pd.DataFrame):
        return source.copy()
    raise TypeError("source must be a HestonDiagnosticsReport or pandas.DataFrame.")


def _backend_compare_table(source: SliceTableSource) -> pd.DataFrame:
    if isinstance(source, HestonDiagnosticsReport):
        return source.tables["backend_compare"].copy()
    if isinstance(source, pd.DataFrame):
        return source.copy()
    raise TypeError("source must be a HestonDiagnosticsReport or pandas.DataFrame.")


def _config_sweep_payload(
    source: SliceTableSource,
    *,
    prices_by_label: Mapping[str, Any] | None,
    strike: Sequence[float] | np.ndarray | None,
) -> tuple[pd.DataFrame, Mapping[str, Any] | None, np.ndarray | None]:
    if isinstance(source, HestonDiagnosticsReport):
        table = source.tables["config_sweep"].copy()
        extracted_prices = source.arrays.get("config_sweep_prices")
        resolved_prices = (
            extracted_prices
            if isinstance(extracted_prices, Mapping)
            else prices_by_label
        )
        if strike is None:
            slice_table = source.tables["slice"]
            if "strike" in slice_table.columns:
                strike = slice_table["strike"].to_numpy(dtype=np.float64, copy=True)
        resolved_strike = (
            None if strike is None else np.asarray(strike, dtype=np.float64).reshape(-1)
        )
        return table, resolved_prices, resolved_strike

    if isinstance(source, pd.DataFrame):
        resolved_strike = (
            None if strike is None else np.asarray(strike, dtype=np.float64).reshape(-1)
        )
        return source.copy(), prices_by_label, resolved_strike

    raise TypeError("source must be a HestonDiagnosticsReport or pandas.DataFrame.")


def _probability_payload(
    source: MappingSource,
    *,
    probability_key: str,
    slice_table: pd.DataFrame | None,
) -> tuple[Mapping[str, Any], pd.DataFrame | None, pd.DataFrame | None]:
    if isinstance(source, HestonDiagnosticsReport):
        payload = source.arrays.get(probability_key)
        if not isinstance(payload, Mapping):
            payload = {}
        resolved_slice_table = source.tables["slice"].copy()
        worst_table_name = (
            "worst_panels_p0" if probability_key.endswith("p0") else "worst_panels_p1"
        )
        return payload, resolved_slice_table, source.tables.get(worst_table_name)

    if isinstance(source, Mapping):
        return source, None if slice_table is None else slice_table.copy(), None

    raise TypeError(
        "source must be a HestonDiagnosticsReport or a mapping of probability arrays."
    )


def _sort_by_strike(table: pd.DataFrame) -> pd.DataFrame:
    if "strike" not in table.columns or table.empty:
        return table.copy()
    return table.sort_values("strike", kind="stable", ignore_index=True)


def _numeric_series(table: pd.DataFrame, column: str) -> np.ndarray:
    if column not in table.columns:
        return np.full(len(table), np.nan, dtype=np.float64)
    return np.asarray(pd.to_numeric(table[column], errors="coerce"), dtype=np.float64)


def _bool_series(table: pd.DataFrame, column: str) -> np.ndarray:
    if column not in table.columns:
        return np.zeros(len(table), dtype=np.bool_)
    series = table[column]
    if series.dtype == bool or series.dtype == np.bool_:
        return series.to_numpy(dtype=np.bool_, copy=False)
    values = [
        False if pd.isna(value) else bool(value)
        for value in series.to_numpy(copy=False)
    ]
    return np.asarray(values, dtype=np.bool_)


def _mapping_array(
    payload: Mapping[str, Any],
    key: str,
    *,
    ndim: int | None = None,
    dtype: Any = np.float64,
) -> np.ndarray | None:
    value = payload.get(key)
    if value is None:
        return None

    arr = np.asarray(value, dtype=dtype)
    if ndim is not None and arr.ndim != ndim:
        return None
    return arr


def _resolve_point_index(
    point_index: int | None,
    *,
    slice_table: pd.DataFrame | None,
    worst_panels: pd.DataFrame | None,
    point_count: int,
) -> int:
    if point_count <= 0:
        raise ValueError("point_count must be positive.")

    if point_index is not None:
        idx = int(point_index)
    elif worst_panels is not None and not worst_panels.empty:
        idx = int(pd.to_numeric(worst_panels["point_index"], errors="coerce").iloc[0])
    else:
        idx = 0

    if idx < 0 or idx >= point_count:
        raise ValueError(
            f"point_index={idx} is out of bounds for {point_count} plotted points."
        )

    if slice_table is not None and len(slice_table) < point_count:
        raise ValueError("slice_table must have at least as many rows as panel points.")

    return idx


def _panel_title(
    *,
    probability_key: str,
    point_index: int,
    slice_table: pd.DataFrame | None,
    payload: Mapping[str, Any],
) -> str:
    label = probability_key.replace("probability_", "").upper()
    title = f"{label} panel contributions"
    details = [f"point={point_index}"]

    if slice_table is not None and point_index < len(slice_table):
        strike_value = pd.to_numeric(
            pd.Series([slice_table.iloc[point_index].get("strike")]),
            errors="coerce",
        ).iloc[0]
        if pd.notna(strike_value):
            details.append(f"strike={float(strike_value):g}")

    log_moneyness = _mapping_array(payload, "log_moneyness", ndim=1)
    if log_moneyness is not None and point_index < log_moneyness.size:
        details.append(f"x={float(log_moneyness[point_index]):+.3f}")

    return f"{title} ({', '.join(details)})"


def _plot_metric_by_strike(
    source: SliceTableSource,
    *,
    combined_column: str,
    per_p0_column: str | None,
    per_p1_column: str | None,
    ylabel: str,
    title: str,
    ax: Axes | None,
) -> tuple[Figure, Axes]:
    table = _sort_by_strike(_slice_table(source))
    require_columns(table, ["strike", combined_column])

    fig, resolved_ax = _ensure_ax(ax, figsize=(8.0, 4.5))
    resolved_ax.set_title(title)
    resolved_ax.set_xlabel("Strike")
    resolved_ax.set_ylabel(ylabel)

    if table.empty:
        _note_axis(resolved_ax, "No strike diagnostics available.")
        pretty_ax(resolved_ax)
        return fig, resolved_ax

    strike = table["strike"].to_numpy(dtype=np.float64, copy=False)
    combined = _numeric_series(table, combined_column)

    if per_p0_column is not None and per_p0_column in table.columns:
        values = _numeric_series(table, per_p0_column)
        if np.isfinite(values).any():
            resolved_ax.plot(strike, values, marker="o", label=f"{ylabel} P0")

    if per_p1_column is not None and per_p1_column in table.columns:
        values = _numeric_series(table, per_p1_column)
        if np.isfinite(values).any():
            resolved_ax.plot(strike, values, marker="o", label=f"{ylabel} P1")

    if np.isfinite(combined).any():
        resolved_ax.plot(
            strike,
            combined,
            marker="o",
            linewidth=2.0,
            color="black",
            label=f"{ylabel} max(P0, P1)",
        )
    else:
        _note_axis(resolved_ax, f"No finite {ylabel.lower()} values are available.")

    if resolved_ax.lines:
        resolved_ax.legend()
    pretty_ax(resolved_ax)
    return fig, resolved_ax


def plot_panel_contributions(
    source: MappingSource,
    *,
    probability_key: str = "probability_p1",
    point_index: int | None = None,
    slice_table: pd.DataFrame | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot signed panel contributions for one strike point.

    Parameters
    ----------
    source:
        Either a full :class:`HestonDiagnosticsReport` or the extracted mapping
        under ``report.arrays[probability_key]``.
    probability_key:
        One of the packaged probability-array groups, such as ``probability_p0``
        or ``probability_p1``.
    point_index:
        Which strike-row to plot. When omitted and a full report is passed, the
        top-ranked worst panel row is used when available.
    slice_table:
        Optional canonical slice table when ``source`` is an arrays mapping.

    Notes
    -----
    The helper only visualizes packaged arrays. If panel-local detail is
    unavailable for the chosen backend, it draws a note instead of recomputing
    diagnostics.
    """

    payload, resolved_slice_table, worst_panels = _probability_payload(
        source,
        probability_key=probability_key,
        slice_table=slice_table,
    )
    panel_contribs = _mapping_array(payload, "panel_contribs", ndim=2)
    panel_invalid = _mapping_array(payload, "panel_invalid", ndim=2, dtype=np.bool_)
    panel_edges = _mapping_array(payload, "panel_edges", ndim=1)

    fig, resolved_ax = _ensure_ax(ax, figsize=(8.0, 4.5))
    resolved_ax.set_title(title or f"{probability_key} panel contributions")

    if panel_contribs is None or panel_contribs.size == 0:
        resolved_ax.set_xlabel("Integration panel")
        resolved_ax.set_ylabel("Panel contribution")
        message = "Panel contribution detail is unavailable in the packaged arrays."
        if worst_panels is not None and not worst_panels.empty:
            notes = (
                worst_panels["notes"]
                .dropna()
                .astype(str)
                .loc[lambda values: values.str.strip().ne("")]
            )
            if not notes.empty:
                message = notes.iloc[0]
        _note_axis(resolved_ax, message)
        pretty_ax(resolved_ax)
        return fig, resolved_ax

    resolved_point_index = _resolve_point_index(
        point_index,
        slice_table=resolved_slice_table,
        worst_panels=worst_panels,
        point_count=int(panel_contribs.shape[0]),
    )
    contributions = np.asarray(
        panel_contribs[resolved_point_index, :],
        dtype=np.float64,
    )
    invalid_mask = (
        np.zeros(contributions.shape, dtype=np.bool_)
        if panel_invalid is None
        else np.asarray(panel_invalid[resolved_point_index, :], dtype=np.bool_)
    )

    if panel_edges is not None and panel_edges.size == contributions.size + 1:
        x_values = 0.5 * (panel_edges[:-1] + panel_edges[1:])
        widths = np.diff(panel_edges)
        resolved_ax.set_xlabel("Integration variable u")
    else:
        x_values = np.arange(contributions.size, dtype=np.float64)
        widths = np.full(contributions.shape, 0.8, dtype=np.float64)
        resolved_ax.set_xlabel("Integration panel")

    valid_mask = ~invalid_mask
    if np.any(valid_mask):
        resolved_ax.bar(
            x_values[valid_mask],
            contributions[valid_mask],
            width=widths[valid_mask],
            label="panel contribution",
            alpha=0.85,
        )
    if np.any(invalid_mask):
        resolved_ax.bar(
            x_values[invalid_mask],
            contributions[invalid_mask],
            width=widths[invalid_mask],
            label="flagged panel",
            color="tab:red",
            alpha=0.9,
        )

    resolved_ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    resolved_ax.set_ylabel("Panel contribution")
    resolved_ax.set_title(
        title
        or _panel_title(
            probability_key=probability_key,
            point_index=resolved_point_index,
            slice_table=resolved_slice_table,
            payload=payload,
        )
    )
    if np.any(invalid_mask):
        resolved_ax.legend()
    pretty_ax(resolved_ax)
    return fig, resolved_ax


def plot_tail_fraction_by_strike(
    source: SliceTableSource,
    *,
    ax: Axes | None = None,
    title: str = "Tail Fraction by Strike",
) -> tuple[Figure, Axes]:
    """Plot packaged tail-fraction diagnostics across the strike slice.

    The plotted values come directly from the canonical slice table, using the
    combined ``tail_fraction`` column and the optional per-probability
    ``tail_fraction_p0`` / ``tail_fraction_p1`` columns when present.
    """

    return _plot_metric_by_strike(
        source,
        combined_column="tail_fraction",
        per_p0_column="tail_fraction_p0",
        per_p1_column="tail_fraction_p1",
        ylabel="Tail fraction",
        title=title,
        ax=ax,
    )


def plot_cancellation_ratio_by_strike(
    source: SliceTableSource,
    *,
    ax: Axes | None = None,
    title: str = "Cancellation Ratio by Strike",
) -> tuple[Figure, Axes]:
    """Plot packaged cancellation-ratio diagnostics across the strike slice.

    This visualizes the review-oriented cancellation ratios already stored in
    the slice report. It is a reporting helper and does not infer new warning
    policy beyond the packaged table values.
    """

    return _plot_metric_by_strike(
        source,
        combined_column="cancellation_ratio",
        per_p0_column="cancellation_ratio_p0",
        per_p1_column="cancellation_ratio_p1",
        ylabel="Cancellation ratio",
        title=title,
        ax=ax,
    )


def plot_backend_difference_by_strike(
    source: SliceTableSource,
    *,
    ax: Axes | None = None,
    title: str = "Backend Difference by Strike",
) -> tuple[Figure, Axes]:
    """Plot packaged backend pricing differences across the strike slice.

    Prefer passing the full report so the helper can use the dedicated
    ``backend_compare`` table. As a fallback it can consume a slice table that
    already contains ``backend_diff``.
    """

    table = _sort_by_strike(_backend_compare_table(source))
    fig, resolved_ax = _ensure_ax(ax, figsize=(8.0, 4.5))
    resolved_ax.set_title(title)
    resolved_ax.set_xlabel("Strike")
    resolved_ax.set_ylabel("Absolute price difference")

    if "abs_price_diff" in table.columns:
        require_columns(table, ["strike", "abs_price_diff"])
        if table.empty:
            _note_axis(resolved_ax, "No backend comparison diagnostics available.")
            pretty_ax(resolved_ax)
            return fig, resolved_ax

        strike = table["strike"].to_numpy(dtype=np.float64, copy=False)
        abs_diff = _numeric_series(table, "abs_price_diff")
        label = "absolute price difference"
        if {"backend_a", "backend_b"} <= set(table.columns):
            backends = table[["backend_a", "backend_b"]].drop_duplicates()
            if len(backends) == 1:
                row = backends.iloc[0]
                label = f"|{row['backend_a']} - {row['backend_b']}|"

        if np.isfinite(abs_diff).any():
            resolved_ax.plot(strike, abs_diff, marker="o", linewidth=2.0, label=label)
            resolved_ax.legend()
        else:
            _note_axis(
                resolved_ax,
                "Backend comparison exists but does not contain finite differences.",
            )

        pretty_ax(resolved_ax)
        return fig, resolved_ax

    slice_table = _slice_table(source)
    require_columns(slice_table, ["strike", "backend_diff"])
    slice_table = _sort_by_strike(slice_table)
    if slice_table.empty:
        _note_axis(resolved_ax, "No backend comparison diagnostics available.")
        pretty_ax(resolved_ax)
        return fig, resolved_ax

    strike = slice_table["strike"].to_numpy(dtype=np.float64, copy=False)
    abs_diff = np.abs(_numeric_series(slice_table, "backend_diff"))
    if np.isfinite(abs_diff).any():
        resolved_ax.plot(strike, abs_diff, marker="o", linewidth=2.0)
    else:
        _note_axis(
            resolved_ax,
            "Slice diagnostics exist but do not contain finite backend differences.",
        )

    pretty_ax(resolved_ax)
    return fig, resolved_ax


def plot_config_sweep(
    source: SliceTableSource,
    *,
    prices_by_label: Mapping[str, Any] | None = None,
    strike: Sequence[float] | np.ndarray | None = None,
    ax: Axes | None = None,
    title: str = "Config / Convergence Sweep",
) -> tuple[Figure, Axes]:
    """Plot the packaged config sweep.

    When per-strike ``config_sweep_prices`` arrays are available, the plot shows
    absolute price differences versus the baseline configuration across strikes.
    Otherwise it falls back to the summary table.

    The helper never recomputes additional sweep cases; it only consumes the
    packaged report artifacts.
    """

    table, resolved_prices, resolved_strike = _config_sweep_payload(
        source,
        prices_by_label=prices_by_label,
        strike=strike,
    )
    fig, resolved_ax = _ensure_ax(ax, figsize=(8.0, 4.5))
    resolved_ax.set_title(title)

    if (
        isinstance(resolved_prices, Mapping)
        and resolved_strike is not None
        and resolved_strike.size > 0
    ):
        ordered_labels: list[str]
        if "config_label" in table.columns:
            ordered_labels = [
                str(label) for label in table["config_label"].astype(str).tolist()
            ]
        else:
            ordered_labels = [str(label) for label in resolved_prices]

        baseline_label: str | None = None
        baseline_values: np.ndarray | None = None
        plotted = 0
        for label in ordered_labels:
            values = resolved_prices.get(label)
            if values is None:
                continue
            arr = np.asarray(values, dtype=np.float64).reshape(-1)
            if arr.size != resolved_strike.size:
                continue
            if baseline_values is None:
                baseline_label = label
                baseline_values = arr
                continue
            resolved_ax.plot(
                resolved_strike,
                np.abs(arr - baseline_values),
                marker="o",
                label=label,
            )
            plotted += 1

        if baseline_values is not None and plotted > 0:
            resolved_ax.axhline(
                0.0,
                color="black",
                linestyle="--",
                linewidth=1.0,
                label=f"baseline: {baseline_label}",
            )
            resolved_ax.set_xlabel("Strike")
            resolved_ax.set_ylabel("|price - baseline|")
            resolved_ax.legend()
            pretty_ax(resolved_ax)
            return fig, resolved_ax

    require_columns(table, ["config_label", "max_abs_price_diff_vs_baseline"])
    resolved_ax.set_xlabel("Configuration")
    resolved_ax.set_ylabel("Max |price - baseline|")

    if table.empty:
        _note_axis(resolved_ax, "No config sweep diagnostics available.")
        pretty_ax(resolved_ax)
        return fig, resolved_ax

    config_labels = table["config_label"].astype(str).tolist()
    max_diff = _numeric_series(table, "max_abs_price_diff_vs_baseline")
    x_positions = np.arange(len(config_labels), dtype=np.float64)

    resolved_ax.bar(x_positions, max_diff)
    resolved_ax.set_xticks(x_positions, config_labels, rotation=30, ha="right")
    if not (
        isinstance(resolved_prices, Mapping)
        and resolved_strike is not None
        and resolved_strike.size > 0
    ):
        _status_note(
            resolved_ax,
            "Per-strike sweep arrays unavailable; showing summary table only.",
        )
    pretty_ax(resolved_ax)
    return fig, resolved_ax


def plot_smile_with_warning_overlay(
    source: SliceTableSource,
    *,
    ax: Axes | None = None,
    title: str = "Smile with Warning / Discontinuity Overlay",
) -> tuple[Figure, Axes]:
    """Plot the implied-vol smile and overlay warning or continuity flags.

    Warning markers come from the packaged strike-slice table, combining the
    stored warning counts, severities, smoothness flags, and discontinuity
    flags. The overlay is meant to focus review attention, not to prove the
    smile is economically valid.
    """

    table = _sort_by_strike(_slice_table(source))
    require_columns(
        table,
        [
            "strike",
            "implied_vol",
            "warning_count",
            "severity",
            "smoothness_flag",
            "discontinuity_flag",
        ],
    )

    fig, resolved_ax = _ensure_ax(ax, figsize=(8.0, 4.5))
    resolved_ax.set_title(title)
    resolved_ax.set_xlabel("Strike")
    resolved_ax.set_ylabel("Implied vol")

    if table.empty:
        _note_axis(resolved_ax, "No strike diagnostics available.")
        pretty_ax(resolved_ax)
        return fig, resolved_ax

    strike = table["strike"].to_numpy(dtype=np.float64, copy=False)
    implied_vol = _numeric_series(table, "implied_vol")
    if np.isfinite(implied_vol).any():
        resolved_ax.plot(
            strike,
            implied_vol,
            marker="o",
            linewidth=2.0,
            color="black",
            label="implied vol",
        )
    else:
        _note_axis(resolved_ax, "No finite implied-vol values are available.")
        pretty_ax(resolved_ax)
        return fig, resolved_ax

    warning_count = _numeric_series(table, "warning_count")
    severity_rank = np.asarray(
        [_severity_rank(value) for value in table["severity"]],
        dtype=np.int64,
    )
    warning_mask = (warning_count > 0) | (severity_rank >= _severity_rank("warning"))
    continuity_mask = _bool_series(table, "smoothness_flag") | _bool_series(
        table,
        "discontinuity_flag",
    )

    if np.any(warning_mask):
        resolved_ax.scatter(
            strike[warning_mask],
            implied_vol[warning_mask],
            s=70,
            color="tab:orange",
            label="warning severity / count",
            zorder=3,
        )
    if np.any(continuity_mask):
        resolved_ax.scatter(
            strike[continuity_mask],
            implied_vol[continuity_mask],
            s=80,
            color="tab:red",
            marker="x",
            label="smoothness / discontinuity flag",
            zorder=4,
        )

    if not np.any(warning_mask) and not np.any(continuity_mask):
        _status_note(
            resolved_ax,
            "No warning, smoothness, or discontinuity flags are active.",
        )

    if resolved_ax.lines or resolved_ax.collections:
        resolved_ax.legend()
    pretty_ax(resolved_ax)
    return fig, resolved_ax


def _heston_mc_metric_table(
    source: pd.DataFrame,
    *,
    metric: str,
) -> pd.DataFrame:
    require_columns(source, ["scheme", "n_steps", "dt"])
    if metric in source.columns:
        return source.copy()

    if {
        "price",
        "signed_error",
        "abs_error",
        "stderr",
        "ci_half_width",
        "covered_reference",
    } <= set(source.columns):
        bias_summary = summarize_bias_vs_timestep(source)
        if metric in bias_summary.columns:
            return bias_summary

    if {
        "runtime_seconds",
        "runtime_per_path",
        "abs_error",
        "signed_error",
        "stderr",
    } <= set(source.columns):
        runtime_summary = summarize_runtime_vs_error(source)
        if metric in runtime_summary.columns:
            return runtime_summary

    raise ValueError(
        "source must be a prepared Heston MC summary table or a raw sweep with the "
        f"columns required to derive {metric!r}."
    )


def plot_heston_mc_bias_vs_timestep(
    summary: pd.DataFrame,
    *,
    x: str = "dt",
    include_abs_error: bool = False,
    ax: Axes | None = None,
    title: str = "Heston MC Bias vs Timestep",
) -> tuple[Figure, Axes]:
    if x not in {"dt", "n_steps"}:
        raise ValueError("x must be either 'dt' or 'n_steps'.")

    require_columns(summary, ["scheme", x, "mean_signed_error"])
    if include_abs_error:
        require_columns(summary, ["mean_abs_error"])

    fig, resolved_ax = _ensure_ax(ax, figsize=(8.0, 4.5))
    resolved_ax.set_title(title)
    resolved_ax.set_xlabel("dt" if x == "dt" else "n_steps")
    resolved_ax.set_ylabel("Bias (MC - reference)")

    if summary.empty:
        _note_axis(resolved_ax, "No Heston MC bias summary rows are available.")
        pretty_ax(resolved_ax)
        return fig, resolved_ax

    for scheme, group in summary.groupby("scheme", sort=True):
        ordered = group.sort_values(x, kind="stable")
        x_values = pd.to_numeric(ordered[x], errors="coerce").to_numpy(dtype=np.float64)
        bias = pd.to_numeric(ordered["mean_signed_error"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        resolved_ax.plot(x_values, bias, marker="o", linewidth=2.0, label=str(scheme))
        if include_abs_error and "mean_abs_error" in ordered.columns:
            abs_error = pd.to_numeric(
                ordered["mean_abs_error"],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            resolved_ax.plot(
                x_values,
                abs_error,
                marker="x",
                linestyle="--",
                alpha=0.75,
                label=f"{scheme} |bias|",
            )

    resolved_ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    if resolved_ax.lines:
        resolved_ax.legend()
    pretty_ax(resolved_ax)
    return fig, resolved_ax


def plot_heston_mc_runtime_vs_error(
    summary: pd.DataFrame,
    *,
    error_metric: str = "mean_abs_error",
    ax: Axes | None = None,
    title: str = "Heston MC Runtime vs Error",
) -> tuple[Figure, Axes]:
    require_columns(summary, ["scheme", "mean_runtime_seconds", error_metric])

    fig, resolved_ax = _ensure_ax(ax, figsize=(8.0, 4.5))
    resolved_ax.set_title(title)
    resolved_ax.set_xlabel("Mean runtime (seconds)")
    resolved_ax.set_ylabel(error_metric.replace("_", " "))

    if summary.empty:
        _note_axis(resolved_ax, "No Heston MC runtime summary rows are available.")
        pretty_ax(resolved_ax)
        return fig, resolved_ax

    for scheme, group in summary.groupby("scheme", sort=True):
        ordered = group.sort_values("mean_runtime_seconds", kind="stable")
        runtime = pd.to_numeric(
            ordered["mean_runtime_seconds"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        error = pd.to_numeric(ordered[error_metric], errors="coerce").to_numpy(
            dtype=np.float64
        )
        resolved_ax.plot(runtime, error, marker="o", linewidth=2.0, label=str(scheme))

    if resolved_ax.lines:
        resolved_ax.legend()
    pretty_ax(resolved_ax)
    return fig, resolved_ax


def plot_heston_mc_scheme_comparison(
    source: pd.DataFrame,
    *,
    metric: str = "mean_abs_error",
    x: str = "n_steps",
    ax: Axes | None = None,
    title: str = "Heston MC Scheme Comparison",
) -> tuple[Figure, Axes]:
    if x not in {"dt", "n_steps"}:
        raise ValueError("x must be either 'dt' or 'n_steps'.")

    table = _heston_mc_metric_table(source, metric=metric)
    require_columns(table, ["scheme", x, metric])

    fig, resolved_ax = _ensure_ax(ax, figsize=(8.0, 4.5))
    resolved_ax.set_title(title)
    resolved_ax.set_xlabel("dt" if x == "dt" else "n_steps")
    resolved_ax.set_ylabel(metric.replace("_", " "))

    if table.empty:
        _note_axis(resolved_ax, "No Heston MC comparison rows are available.")
        pretty_ax(resolved_ax)
        return fig, resolved_ax

    for scheme, group in table.groupby("scheme", sort=True):
        ordered = group.sort_values(x, kind="stable")
        x_values = pd.to_numeric(ordered[x], errors="coerce").to_numpy(dtype=np.float64)
        y_values = pd.to_numeric(ordered[metric], errors="coerce").to_numpy(
            dtype=np.float64
        )
        resolved_ax.plot(
            x_values, y_values, marker="o", linewidth=2.0, label=str(scheme)
        )

    if resolved_ax.lines:
        resolved_ax.legend()
    pretty_ax(resolved_ax)
    return fig, resolved_ax


__all__ = [
    "plot_backend_difference_by_strike",
    "plot_cancellation_ratio_by_strike",
    "plot_config_sweep",
    "plot_heston_mc_bias_vs_timestep",
    "plot_heston_mc_runtime_vs_error",
    "plot_heston_mc_scheme_comparison",
    "plot_panel_contributions",
    "plot_smile_with_warning_overlay",
    "plot_tail_fraction_by_strike",
]
