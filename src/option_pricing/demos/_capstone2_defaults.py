"""Defaults for the Capstone 2 demo recipe."""

from __future__ import annotations

from typing import Any

import numpy as np

from option_pricing.diagnostics.vol_surface.recipes import default_svi_repair_candidates
from option_pricing.numerics.grids import SpacingPolicy
from option_pricing.numerics.pde.domain import Coord
from option_pricing.numerics.pde.operators import AdvectionScheme
from option_pricing.pricers.pde.domain import BSDomainConfig, BSDomainPolicy


def get_capstone2_defaults(seed: int) -> dict[str, Any]:
    """Return default configs for the Capstone 2 demo.

    Parameters
    ----------
    seed:
        Random seed for the synthetic surface generator.
    """
    synth_cfg = dict(
        spot=100.0,
        r=0.02,
        q=0.00,
        expiries=(0.10, 0.15, 0.20, 0.25, 0.33, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00),
        x_grid=np.linspace(-0.30, 0.30, 31),
        model="svi",
        atm_level=0.20,
        atm_term_slope=0.045,
        atm_term_ref=0.50,
        b_level=0.030,
        b_term_slope=0.006,
        rho=-0.35,
        m_level=-0.015,
        m_decay=1.0,
        sigma_level=0.24,
        sigma_term_slope=0.02,
        noise_mode="absolute",
        noise_level=0.006,
        noise_dist="normal",
        noise_smooth_window=1,
        outlier_prob=0.0,
        outlier_scale=4.0,
        missing_prob=0.0,
        seed=int(seed),
    )

    svi_calib_no_repair = dict(
        robust_data_only=True,
        repair_butterfly=False,
        refit_after_repair=False,
    )

    svi_calib_repair = dict(
        robust_data_only=True,
        repair_butterfly=True,
        repair_method="line_search",
        repair_n_scan=101,
        repair_n_bisect=50,
        refit_after_repair=True,
        butterfly_min_g_tol=None,
        butterfly_min_g_tol_scale=1.0,
    )

    svi_calib_repair_candidates = default_svi_repair_candidates(
        robust_data_only=True,
        include_robust_all_candidate=True,
    )
    # Demo stability: make the repair scan denser and try robust-all earlier.
    for _, ck in svi_calib_repair_candidates:
        if "repair_n_scan" in ck:
            ck["repair_n_scan"] = 201
        ck.setdefault("butterfly_min_g_tol", None)
        ck.setdefault("butterfly_min_g_tol_scale", 1.0)
    for i, (label, _) in enumerate(svi_calib_repair_candidates):
        if label == "robust_all_line_search":
            svi_calib_repair_candidates.insert(0, svi_calib_repair_candidates.pop(i))
            break

    lv_diag_expiries = np.linspace(0.12, 1.95, 10)
    lv_diag_ygrid = np.linspace(-0.30, 0.30, 61)

    shared_strikes = np.linspace(75.0, 130.0, 31)
    shared_expiries = np.array(
        [0.10, 0.15, 0.20, 0.25, 0.33, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00],
        dtype=float,
    )

    lv_domain_cfg = BSDomainConfig(
        policy=BSDomainPolicy.LOG_NSIGMA,
        n_sigma=5.5,
        center="strike",
        spacing=SpacingPolicy.CLUSTERED,
        cluster_strength=2.5,
    )

    lv_pde_solver_cfg = dict(
        coord=Coord.LOG_S,
        domain_cfg=lv_domain_cfg,
        method="rannacher",
        advection=AdvectionScheme.CENTRAL,
    )

    lv_sweep_grids = [(101, 201), (151, 301), (201, 401), (251, 501)]

    return {
        "ENFORCE_ARB_FREE_LATENT_TRUTH": True,
        "SYNTH_MAX_ROUNDS": 8,
        "SYNTH_CFG": synth_cfg,
        "SVI_CALIB_NO_REPAIR": svi_calib_no_repair,
        "SVI_CALIB_REPAIR": svi_calib_repair,
        "SVI_CALIB_REPAIR_CANDIDATES": svi_calib_repair_candidates,
        "LV_DIAG_EXPIRIES": lv_diag_expiries,
        "LV_DIAG_YGRID": lv_diag_ygrid,
        "SHARED_STRIKES": shared_strikes,
        "SHARED_EXPIRIES": shared_expiries,
        "LV_DOMAIN_CFG": lv_domain_cfg,
        "LV_PDE_SOLVER_CFG": lv_pde_solver_cfg,
        "LV_SWEEP_GRIDS": lv_sweep_grids,
    }
