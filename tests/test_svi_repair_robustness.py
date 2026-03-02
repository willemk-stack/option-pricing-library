import numpy as np

from option_pricing.data_generators.synthetic_surface import generate_synthetic_surface
from option_pricing.diagnostics.vol_surface.recipes import (
    build_svi_surface_with_fallback,
    default_svi_repair_candidates,
)


def _with_auto_tol(candidates, *, scale: float):
    out = []
    for label, ck in candidates:
        cfg = dict(ck)
        cfg.setdefault("butterfly_min_g_tol", None)
        cfg.setdefault("butterfly_min_g_tol_scale", float(scale))
        out.append((label, cfg))
    return out


def test_svi_repair_robustness_stress():
    expiries = (0.10, 0.15, 0.20, 0.25, 0.33, 0.50, 0.75, 1.00, 1.25)
    x_grid = np.linspace(-0.30, 0.30, 31)
    seeds = (3, 5, 7, 11, 13)
    noise_levels = (0.0005, 0.0010, 0.0020)

    candidates = default_svi_repair_candidates(
        robust_data_only=True,
        include_robust_all_candidate=True,
    )
    candidates = _with_auto_tol(candidates, scale=1.0)

    for seed in seeds:
        for nl in noise_levels:
            syn = generate_synthetic_surface(
                model="svi",
                expiries=expiries,
                x_grid=x_grid,
                noise_mode="absolute",
                noise_level=float(nl),
                noise_dist="normal",
                noise_smooth_window=1,
                outlier_prob=0.0,
                missing_prob=0.0,
                seed=int(seed),
            )

            surface, mode, attempts = build_svi_surface_with_fallback(
                syn.rows_obs,
                forward=syn.forward,
                candidates=candidates,
                fallback_surface=None,
            )

            assert mode != "FALLBACK"
            assert attempts["ok"].astype(bool).any()
            assert len(surface.smiles) == len(np.unique(syn.T))
