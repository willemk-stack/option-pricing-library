import numpy as np
import pandas as pd
import pytest

from option_pricing.data_generators import recipes as R
from option_pricing.data_generators.synthetic_surface import generate_synthetic_surface
from option_pricing.diagnostics.vol_surface.recipes import (
    build_svi_surface_with_fallback,
    default_svi_repair_candidates,
)


class _DummyRep:
    def __init__(self, ok: bool, message: str = "x"):
        self.ok = ok
        self.message = message


def test_recipes_fallback_poly_path(monkeypatch):
    # 1st attempt fails => fallback_poly path => succeeds
    calls = iter([False, True])

    def fake_check_surface_noarb(*args, **kwargs):
        return _DummyRep(ok=next(calls), message="patched")

    monkeypatch.setattr(R, "check_surface_noarb", fake_check_surface_noarb)

    res = R.generate_synthetic_surface_latent_noarb(
        enforce=True,
        max_rounds=1,
        fallback_poly=True,
        model="svi",  # ensures fallback section runs
        expiries=(0.25,),
        x_grid=np.linspace(-0.1, 0.1, 5),
        seed=0,
    )

    assert bool(res.noarb_true.ok) is True
    assert res.cfg_used["model"] == "poly"
    assert isinstance(res.tuning_log, pd.DataFrame)
    assert len(res.tuning_log) >= 2


def test_recipes_raises_runtimeerror_when_enforce_and_no_fallback(monkeypatch):
    def fake_check_surface_noarb(*args, **kwargs):
        return _DummyRep(ok=False, message="patched-fail")

    monkeypatch.setattr(R, "check_surface_noarb", fake_check_surface_noarb)

    with pytest.raises(RuntimeError) as e:
        R.generate_synthetic_surface_latent_noarb(
            enforce=True,
            max_rounds=1,
            fallback_poly=False,
            model="poly",
            expiries=(0.25,),
            x_grid=(-0.1, 0.0, 0.1),
            seed=0,
        )

    # RuntimeError(msg, tuning_log) => args[1] is the DataFrame
    assert len(e.value.args) >= 2


def test_build_svi_surface_with_fallback_noiseless():
    syn = generate_synthetic_surface(
        model="svi",
        expiries=(0.5, 1.0, 1.5),
        x_grid=np.linspace(-0.25, 0.25, 11),
        noise_mode="none",
        noise_level=0.0,
        outlier_prob=0.0,
        noise_smooth_window=1,
        seed=7,
    )

    candidates = default_svi_repair_candidates(
        robust_data_only=True,
        include_robust_all_candidate=False,
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


def test_build_svi_surface_with_fallback_mild_noise():
    syn = generate_synthetic_surface(
        model="svi",
        expiries=(0.5, 1.0, 1.5),
        x_grid=np.linspace(-0.25, 0.25, 11),
        noise_mode="absolute",
        noise_level=0.0005,
        noise_dist="normal",
        noise_smooth_window=1,
        outlier_prob=0.0,
        seed=7,
    )

    candidates = default_svi_repair_candidates(
        robust_data_only=True,
        include_robust_all_candidate=True,
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
