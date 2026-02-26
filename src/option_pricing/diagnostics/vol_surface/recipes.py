# ruff: noqa: E402
from __future__ import annotations

"""Small orchestration helpers for vol-surface diagnostics.

These functions are intentionally placed under ``option_pricing.diagnostics``:
they encode *workflow glue* that is convenient for notebooks and experiments
(e.g., trying multiple calibration/repair configurations) but should not be
surprising core-library behavior.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from option_pricing.data_generators.Synthetic_Surface import generate_bad_svi_smile_case
from option_pricing.vol.surface import VolSurface
from option_pricing.vol.svi.diagnostics import (
    GJExample51RepairResult,
    check_butterfly_arbitrage,
)
from option_pricing.vol.svi.math import gatheral_g_vec
from option_pricing.vol.svi.repair import repair_butterfly_with_fallback


def default_svi_repair_candidates(
    *,
    robust_data_only: bool = True,
    include_robust_all_candidate: bool | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Return an ordered list of stable SVI calibration+repair candidates.

    The defaults mirror the candidate set used in demo 7.

    Parameters
    ----------
    robust_data_only:
        Base setting used for the main line-search / project candidates.
    include_robust_all_candidate:
        Whether to append an additional all-data robust-loss candidate.
        When omitted, it defaults to ``not robust_data_only``.
    """
    if include_robust_all_candidate is None:
        include_robust_all_candidate = not robust_data_only
    base_repair = dict(
        robust_data_only=bool(robust_data_only),
        repair_butterfly=True,
        repair_method="line_search",
        repair_n_scan=101,
        repair_n_bisect=50,
        refit_after_repair=True,
    )

    candidates: list[tuple[str, dict[str, Any]]] = [
        ("line_search_no_refit", dict(base_repair, refit_after_repair=False)),
        ("line_search_refit", dict(base_repair, refit_after_repair=True)),
        (
            "project_no_refit",
            dict(base_repair, repair_method="project", refit_after_repair=False),
        ),
    ]

    # Optionally include an all-data robust-loss candidate that often helps when
    # quotes include outliers.
    if include_robust_all_candidate:
        candidates.append(
            (
                "robust_all_line_search",
                dict(base_repair, robust_data_only=False, loss="huber", f_scale=0.1),
            )
        )

    return candidates


def build_svi_surface_with_fallback(
    rows: list[tuple[float, float, float]],
    *,
    forward,
    candidates: Sequence[tuple[str, dict[str, Any]]],
    fallback_surface: VolSurface | None = None,
) -> tuple[VolSurface, str, pd.DataFrame]:
    """Try building an SVI surface with multiple calibration+repair candidates.

    Parameters
    ----------
    rows:
        Iterable of (T, K, iv) points.
    forward:
        forward(T) -> float.
    candidates:
        Sequence of (label, calibrate_kwargs) tried in order.
    fallback_surface:
        If all candidates fail, optionally return this surface (label="FALLBACK").

    Returns
    -------
    surface, mode_label, attempts_df
    """
    attempts: list[dict[str, Any]] = []
    last_exc: Exception | None = None

    for label, ck in candidates:
        try:
            s = VolSurface.from_svi(rows, forward=forward, calibrate_kwargs=dict(ck))
            attempts.append({"label": str(label), "ok": True, "error": ""})
            return s, str(label), pd.DataFrame(attempts)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            attempts.append(
                {"label": str(label), "ok": False, "error": f"{type(e).__name__}: {e}"}
            )

    attempts_df = pd.DataFrame(attempts)

    if fallback_surface is not None:
        attempts_df = pd.concat(
            [
                attempts_df,
                pd.DataFrame(
                    [{"label": "FALLBACK", "ok": True, "error": ""}],
                ),
            ],
            ignore_index=True,
        )
        return fallback_surface, "FALLBACK", attempts_df

    raise RuntimeError(
        "All SVI build attempts failed",
        attempts_df,
        last_exc,
    )


__all__ = [
    "ExplicitSVIRepairDemoResult",
    "default_svi_repair_candidates",
    "build_svi_surface_with_fallback",
    "run_explicit_svi_repair_demo",
    "gj_example51_comparison_table",
    "gj_example51_check_table",
]


@dataclass(frozen=True, slots=True)
class ExplicitSVIRepairDemoResult:
    """Notebook-friendly bundle for the explicit one-slice SVI repair demo."""

    metadata: pd.DataFrame
    summary: pd.DataFrame
    repair_attempts: pd.DataFrame
    smile_pre_plot: SimpleNamespace
    smile_post_plot: SimpleNamespace


def _min_g_on_plot_grid(
    params: Any,
    y_min: float,
    y_max: float,
    *,
    n: int = 801,
    w_floor: float = 0.0,
) -> tuple[float, float]:
    y = np.linspace(float(y_min), float(y_max), int(n), dtype=float)
    g = np.asarray(gatheral_g_vec(y, params, w_floor=w_floor), dtype=float)
    idx = int(np.nanargmin(g)) if np.any(np.isfinite(g)) else 0
    return float(g[idx]), float(y[idx])


def run_explicit_svi_repair_demo(
    *,
    focus_T: float,
    repaired_surface: VolSurface | None = None,
    y_domain_fallback: tuple[float, float] = (-0.35, 0.35),
    w_floor: float = 1e-12,
    g_floor: float = 0.0,
    plot_grid_size: int = 801,
    y_obj_size: int = 101,
    y_penalty_size: int = 301,
    repair_scan_size: int = 101,
    repair_bisect_steps: int = 50,
) -> ExplicitSVIRepairDemoResult:
    """Run the explicit one-slice butterfly-repair showcase used in demo 7.

    The goal is to keep notebook code focused on *displaying* the before/after
    narrative, while this helper owns the deterministic smile generation, repair
    fallback sequence, and tabular summaries.
    """
    from .compute import get_smile_at_T

    base_params = None
    base_source = "canonical"
    smile_fx = None

    if repaired_surface is not None:
        try:
            smile_fx = get_smile_at_T(repaired_surface, float(focus_T))
            base_params = getattr(smile_fx, "params", None)
            if base_params is not None:
                base_source = "focus_slice_params"
        except Exception:  # noqa: BLE001
            smile_fx = None

    y_domain = y_domain_fallback
    if (
        smile_fx is not None
        and hasattr(smile_fx, "y_min")
        and hasattr(smile_fx, "y_max")
    ):
        y_domain = (float(smile_fx.y_min), float(smile_fx.y_max))

    bad_case = generate_bad_svi_smile_case(
        T=float(focus_T),
        y_domain=y_domain,
        base_params=base_params,
    )

    T_case = float(bad_case.T)
    y_min, y_max = float(bad_case.y_min), float(bad_case.y_max)
    params_pre = bad_case.params_bad

    check_pre = check_butterfly_arbitrage(
        params_pre,
        y_domain_hint=(y_min, y_max),
        w_floor=w_floor,
        g_floor=g_floor,
    )
    min_g_pre_plot, argmin_pre_plot = _min_g_on_plot_grid(
        params_pre,
        y_min,
        y_max,
        n=plot_grid_size,
        w_floor=w_floor,
    )

    params_post, check_post, repair_attempt_log = repair_butterfly_with_fallback(
        params_pre,
        T=T_case,
        y_domain_hint=(y_min, y_max),
        w_floor=w_floor,
        try_jw_optimal=True,
        raw_methods=("line_search", "project"),
        jw_kwargs=dict(
            y_obj=np.linspace(y_min, y_max, int(y_obj_size)),
            y_penalty=np.linspace(y_min, y_max, int(y_penalty_size)),
            init_method="line_search",
            init_n_scan=int(repair_scan_size),
            init_n_bisect=int(repair_bisect_steps),
        ),
        raw_kwargs=dict(
            n_scan=int(repair_scan_size),
            n_bisect=int(repair_bisect_steps),
        ),
    )

    min_g_post_plot, argmin_post_plot = _min_g_on_plot_grid(
        params_post,
        y_min,
        y_max,
        n=plot_grid_size,
        w_floor=w_floor,
    )

    metadata = pd.DataFrame(
        [
            {
                "base_source": base_source,
                "generator_source": bad_case.source,
                **bad_case.metadata,
                "y_min": y_min,
                "y_max": y_max,
                "T": T_case,
            }
        ]
    )

    summary = pd.DataFrame(
        [
            {
                "stage": "before_repair",
                "ok": bool(check_pre.ok),
                "min_g": float(check_pre.min_g),
                "min_g_plot": float(min_g_pre_plot),
                "argmin_plot": float(argmin_pre_plot),
                "lee_ok": bool(check_pre.lee_ok),
                "params": repr(params_pre),
                "failure_reason": str(check_pre.failure_reason),
            },
            {
                "stage": "after_repair",
                "ok": bool(check_post.ok),
                "min_g": float(check_post.min_g),
                "min_g_plot": float(min_g_post_plot),
                "argmin_plot": float(argmin_post_plot),
                "lee_ok": bool(check_post.lee_ok),
                "params": repr(params_post),
                "failure_reason": str(check_post.failure_reason),
            },
        ]
    )

    repair_attempts = pd.DataFrame(repair_attempt_log)
    smile_pre_plot = SimpleNamespace(
        params=params_pre, y_min=y_min, y_max=y_max, T=T_case
    )
    smile_post_plot = SimpleNamespace(
        params=params_post,
        y_min=y_min,
        y_max=y_max,
        T=T_case,
    )

    return ExplicitSVIRepairDemoResult(
        metadata=metadata,
        summary=summary,
        repair_attempts=repair_attempts,
        smile_pre_plot=smile_pre_plot,
        smile_post_plot=smile_post_plot,
    )


def gj_example51_comparison_table(result: GJExample51RepairResult) -> pd.DataFrame:
    """Return the compact computed-vs-paper table used in demo 7."""
    return pd.DataFrame(
        [
            {
                "quantity": "JW.v",
                "computed": float(result.jw_raw.v),
                "paper": result.paper_jw["v"],
            },
            {
                "quantity": "JW.psi",
                "computed": float(result.jw_raw.psi),
                "paper": result.paper_jw["psi"],
            },
            {
                "quantity": "JW.p",
                "computed": float(result.jw_raw.p),
                "paper": result.paper_jw["p"],
            },
            {
                "quantity": "JW.c",
                "computed": float(result.jw_raw.c),
                "paper": result.paper_jw["c"],
            },
            {
                "quantity": "JW.v_tilde",
                "computed": float(result.jw_raw.v_tilde),
                "paper": result.paper_jw["v_tilde"],
            },
            {
                "quantity": "Section 5.1 target c0",
                "computed": float(result.c0_target),
                "paper": result.paper_proj["c0"],
            },
            {
                "quantity": "Section 5.1 target vtilde0",
                "computed": float(result.vtilde0_target),
                "paper": result.paper_proj["vtilde0"],
            },
            {
                "quantity": "Projected c0 (actual)",
                "computed": float(result.jw_projected.c),
                "paper": result.paper_proj["c0"],
            },
            {
                "quantity": "Projected vtilde0 (actual)",
                "computed": float(result.jw_projected.v_tilde),
                "paper": result.paper_proj["vtilde0"],
            },
            {
                "quantity": "Optimal c*",
                "computed": float(result.jw_optimal.c),
                "paper": result.paper_opt["c_star"],
            },
            {
                "quantity": "Optimal vtilde*",
                "computed": float(result.jw_optimal.v_tilde),
                "paper": result.paper_opt["vtilde_star"],
            },
        ]
    )


def gj_example51_check_table(result: GJExample51RepairResult) -> pd.DataFrame:
    """Return the compact butterfly-check summary used in demo 7."""
    return pd.DataFrame(
        [
            {
                "stage": "orig",
                "ok": bool(result.check_raw.ok),
                "min_g": float(result.check_raw.min_g),
                "failure_reason": str(result.check_raw.failure_reason),
            },
            {
                "stage": "projected",
                "ok": bool(result.check_projected.ok),
                "min_g": float(result.check_projected.min_g),
                "failure_reason": str(result.check_projected.failure_reason),
            },
            {
                "stage": "optimal",
                "ok": bool(result.check_optimal.ok),
                "min_g": float(result.check_optimal.min_g),
                "failure_reason": str(result.check_optimal.failure_reason),
            },
        ]
    )
