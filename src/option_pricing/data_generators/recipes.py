# ruff: noqa: E402
from __future__ import annotations

"""Opinionated synthetic-surface recipes.

The core generator :func:`option_pricing.data_generators.synthetic_surface.generate_synthetic_surface`
is intentionally simple and exposes many knobs. For demos and tests we often
want *reproducible* synthetic quotes whose **latent truth** is arbitrage-free,
while still allowing the **observed quotes** to contain noise/outliers.

This module provides small orchestration helpers to keep notebooks clean.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from option_pricing.data_generators.synthetic_surface import (
    SyntheticSurface,
    generate_synthetic_surface,
)
from option_pricing.vol.arbitrage import SurfaceNoArbReport, check_surface_noarb
from option_pricing.vol.surface_core import VolSurface


@dataclass(frozen=True)
class LatentNoArbSyntheticResult:
    """Bundle returned by :func:`generate_synthetic_surface_latent_noarb`."""

    synthetic: SyntheticSurface
    surface_true: VolSurface
    noarb_true: SurfaceNoArbReport
    rows_true: list[tuple[float, float, float]]
    cfg_used: dict[str, Any]
    tuning_log: pd.DataFrame


def _truth_cfg_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Clone cfg but force *latent truth* generation (no noise / no missing)."""
    truth = dict(cfg)
    # Preserve all latent-shape parameters, grid, expiries, seed, etc.
    truth.update(
        dict(
            noise_mode="none",
            noise_level=0.0,
            outlier_prob=0.0,
            outlier_scale=0.0,
            missing_prob=0.0,
            noise_smooth_window=1,
        )
    )
    return truth


def _rows_from_synth_true(syn: SyntheticSurface) -> list[tuple[float, float, float]]:
    # SyntheticSurface already contains the realized (T, K) grid; for truth
    # generation we call generate_synthetic_surface with missing_prob=0.
    T = np.asarray(syn.T, dtype=float).reshape(-1)
    K = np.asarray(syn.K, dtype=float).reshape(-1)
    iv = np.asarray(syn.iv_true, dtype=float).reshape(-1)
    return [(float(t), float(k), float(v)) for t, k, v in zip(T, K, iv, strict=True)]


def _safe_poly_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a conservative poly-model config likely to pass no-arb checks."""
    out = dict(cfg)
    out["model"] = "poly"

    # Keep term structure mild and monotone in total variance.
    out.setdefault("base_level", 0.20)
    out["term_slope"] = max(0.0, float(out.get("term_slope", 0.03)))
    out.setdefault("term_ref", 0.50)

    # Keep smile convex and not too steep in wings.
    out["skew"] = float(out.get("skew", -0.02))
    out["curvature"] = float(out.get("curvature", 0.06))
    out["twist"] = float(out.get("twist", 0.0))
    return out


def _tune_latent_cfg(
    cfg: dict[str, Any],
    rep: SurfaceNoArbReport,
    *,
    tune_strategy: Literal["svi_safe", "poly_safe"] = "svi_safe",
) -> dict[str, Any]:
    """Heuristic tuning step to nudge the latent surface toward no-arbitrage."""
    out = dict(cfg)
    model = str(out.get("model", "poly")).lower().strip()

    # Ensure time monotonicity controls are not negative.
    if model == "poly":
        out["term_slope"] = max(0.0, float(out.get("term_slope", 0.0)))
        # Reduce wing curvature and higher-order twist; damp skew slightly.
        out["curvature"] = float(out.get("curvature", 0.10)) * 0.85
        out["twist"] = float(out.get("twist", 0.0)) * 0.70
        out["skew"] = float(out.get("skew", -0.02)) * 0.90
        # Slightly raise overall level to avoid near-zero w issues.
        out["base_level"] = max(float(out.get("base_level", 0.18)) * 1.02, 0.05)
        return out

    if model == "svi":
        # Clamp term slopes to avoid calendar variance violations.
        out["atm_term_slope"] = max(0.0, float(out.get("atm_term_slope", 0.0)))
        out["b_term_slope"] = max(0.0, float(out.get("b_term_slope", 0.0)))
        out["sigma_term_slope"] = max(0.0, float(out.get("sigma_term_slope", 0.0)))

        # Make wings less aggressive: shrink b and |rho|, increase sigma.
        out["b_level"] = float(out.get("b_level", 0.20)) * 0.80
        out["rho"] = float(out.get("rho", -0.30)) * 0.90
        out["sigma_level"] = float(out.get("sigma_level", 0.20)) * 1.05
        # Reduce m offset magnitude to avoid asymmetric extremes.
        out["m_level"] = float(out.get("m_level", 0.0)) * 0.70
        # Keep atm level sensible.
        out["atm_level"] = max(float(out.get("atm_level", 0.18)) * 1.01, 0.05)
        return out

    # Unknown model -> fallback to conservative poly.
    return _safe_poly_defaults(out)


def _log_row(cfg: dict[str, Any], rep: SurfaceNoArbReport, *, round_i: int) -> dict:
    model = str(cfg.get("model", "poly")).lower().strip()
    row: dict[str, Any] = {
        "round": int(round_i),
        "model": model,
        "ok": bool(rep.ok),
        "message": str(rep.message),
    }
    # Keep log compact and readable.
    if model == "poly":
        for k in ("base_level", "term_slope", "skew", "curvature", "twist"):
            if k in cfg:
                row[k] = float(cfg[k])
    if model == "svi":
        for k in (
            "atm_level",
            "atm_term_slope",
            "b_level",
            "b_term_slope",
            "rho",
            "m_level",
            "sigma_level",
            "sigma_term_slope",
        ):
            if k in cfg:
                row[k] = float(cfg[k])
    return row


def generate_synthetic_surface_latent_noarb(
    *,
    max_rounds: int = 8,
    enforce: bool = True,
    tune_strategy: Literal["svi_safe", "poly_safe"] = "svi_safe",
    fallback_poly: bool = True,
    **synth_cfg: Any,
) -> LatentNoArbSyntheticResult:
    """Generate synthetic quotes with an arbitrage-free latent truth.

    Parameters
    ----------
    max_rounds:
        Maximum tuning iterations.
    enforce:
        If False, no tuning is performed; the first generated latent truth is
        returned alongside its no-arbitrage report.
    tune_strategy:
        Heuristic tuning mode.
    fallback_poly:
        If the model is SVI-like and cannot be tuned to pass no-arbitrage checks,
        optionally switch to a conservative polynomial latent truth.
    **synth_cfg:
        Forwarded to :func:`generate_synthetic_surface`.

    Returns
    -------
    LatentNoArbSyntheticResult
        Contains observed synthetic quotes, latent truth surface, no-arb report,
        and a tuning log.

    Raises
    ------
    RuntimeError
        If ``enforce=True`` and no arbitrage-free latent truth could be produced.
    """
    if max_rounds <= 0:
        raise ValueError("max_rounds must be >= 1")

    cfg = dict(synth_cfg)
    log_rows: list[dict[str, Any]] = []

    last_rep: SurfaceNoArbReport | None = None

    for i in range(int(max_rounds)):
        syn_obs = generate_synthetic_surface(**cfg)

        syn_true = generate_synthetic_surface(**_truth_cfg_from_cfg(cfg))
        rows_true = _rows_from_synth_true(syn_true)
        surface_true = VolSurface.from_grid(rows_true, forward=syn_true.forward)
        rep = check_surface_noarb(surface_true, df=syn_true.df)

        log_rows.append(_log_row(cfg, rep, round_i=i))

        last_rep = rep

        if (not enforce) or bool(rep.ok):
            return LatentNoArbSyntheticResult(
                synthetic=syn_obs,
                surface_true=surface_true,
                noarb_true=rep,
                rows_true=rows_true,
                cfg_used=dict(cfg),
                tuning_log=pd.DataFrame(log_rows),
            )

        cfg = _tune_latent_cfg(cfg, rep, tune_strategy=tune_strategy)

    # If still not OK, optionally fall back to a conservative poly truth.
    if enforce and fallback_poly and str(cfg.get("model", "")).lower() != "poly":
        cfg_poly = _safe_poly_defaults(cfg)
        syn_obs = generate_synthetic_surface(**cfg_poly)
        syn_true = generate_synthetic_surface(**_truth_cfg_from_cfg(cfg_poly))
        rows_true = _rows_from_synth_true(syn_true)
        surface_true = VolSurface.from_grid(rows_true, forward=syn_true.forward)
        rep = check_surface_noarb(surface_true, df=syn_true.df)
        log_rows.append(_log_row(cfg_poly, rep, round_i=len(log_rows)))

        if bool(rep.ok):
            return LatentNoArbSyntheticResult(
                synthetic=syn_obs,
                surface_true=surface_true,
                noarb_true=rep,
                rows_true=rows_true,
                cfg_used=dict(cfg_poly),
                tuning_log=pd.DataFrame(log_rows),
            )

        # record last report and move on; surface/rows variables not needed
        last_rep = rep
        cfg = cfg_poly

    tuning_log = pd.DataFrame(log_rows)
    msg = (
        "Failed to generate an arbitrage-free latent truth surface "
        f"after {len(log_rows)} attempts. "
        f"Last status: ok={bool(getattr(last_rep, 'ok', False))}, message={getattr(last_rep, 'message', '')!s}"
    )
    raise RuntimeError(msg, tuning_log)


__all__ = [
    "LatentNoArbSyntheticResult",
    "generate_synthetic_surface_latent_noarb",
]
