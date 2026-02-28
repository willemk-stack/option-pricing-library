import numpy as np
import pandas as pd
import pytest

from option_pricing.data_generators import recipes as R


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
