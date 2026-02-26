# ruff: noqa: E402
from __future__ import annotations

"""Small orchestration helpers for vol-surface diagnostics.

These functions are intentionally placed under ``option_pricing.diagnostics``:
they encode *workflow glue* that is convenient for notebooks and experiments
(e.g., trying multiple calibration/repair configurations) but should not be
surprising core-library behavior.
"""

from collections.abc import Sequence
from typing import Any

import pandas as pd

from option_pricing.vol.surface import VolSurface


def default_svi_repair_candidates(
    *, robust_data_only: bool = True
) -> list[tuple[str, dict[str, Any]]]:
    """Return an ordered list of stable SVI calibration+repair candidates.

    The defaults mirror the candidate set used in demo 7.
    """
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

    # If we allow all data, include a robust-loss candidate that often helps when
    # quotes include outliers.
    if not robust_data_only:
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
    "default_svi_repair_candidates",
    "build_svi_surface_with_fallback",
]
