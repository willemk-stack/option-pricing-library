import numpy as np
import pytest

# NOTE: This import assumes you renamed Synthetic_Surface.py -> synthetic_surface.py
from option_pricing.data_generators import generate_synthetic_surface
from option_pricing.data_generators.synthetic_surface import generate_bad_svi_smile_case


def test_data_generators_package_importable():
    # Importing from package hits option_pricing/data_generators/__init__.py
    from option_pricing import data_generators as dg

    assert hasattr(dg, "generate_synthetic_surface")
    assert callable(dg.generate_synthetic_surface)


def test_generate_synthetic_surface_deterministic_seed():
    cfg = dict(
        spot=100.0,
        r=0.02,
        q=0.01,
        expiries=(0.25, 0.5),
        x_grid=np.linspace(-0.2, 0.2, 9),
        model="poly",
        seed=123,
    )
    a = generate_synthetic_surface(**cfg)
    b = generate_synthetic_surface(**cfg)

    np.testing.assert_allclose(a.T, b.T)
    np.testing.assert_allclose(a.K, b.K)
    np.testing.assert_allclose(a.iv_true, b.iv_true)
    np.testing.assert_allclose(a.iv_obs, b.iv_obs)
    assert a.rows_obs == b.rows_obs


def test_generate_synthetic_surface_noise_none_equals_true():
    syn = generate_synthetic_surface(
        expiries=(0.25, 0.5),
        x_grid=np.linspace(-0.3, 0.3, 11),
        model="poly",
        noise_mode="none",
        noise_level=0.25,  # should be ignored in "none" mode
        missing_prob=0.0,
        seed=7,
    )

    np.testing.assert_allclose(syn.iv_obs, syn.iv_true, rtol=0, atol=0)
    assert len(syn.rows_obs) == syn.iv_obs.size
    assert np.all(np.isfinite(syn.iv_obs))
    # sanity check: curves are callable
    assert syn.forward(0.5) > 0.0
    assert 0.0 < syn.df(0.5) <= 1.0


def test_generate_synthetic_surface_missing_prob_reduces_points():
    base = generate_synthetic_surface(
        expiries=(0.25, 0.5),
        x_grid=np.linspace(-0.2, 0.2, 21),
        model="poly",
        missing_prob=0.0,
        seed=11,
    )
    miss = generate_synthetic_surface(
        expiries=(0.25, 0.5),
        x_grid=np.linspace(-0.2, 0.2, 21),
        model="poly",
        missing_prob=0.5,
        seed=11,
    )

    assert miss.iv_obs.size < base.iv_obs.size
    assert len(miss.rows_obs) == miss.iv_obs.size


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(spot=0.0),
        dict(expiries=()),
        dict(expiries=(0.0, 0.5)),
        dict(x_grid=()),
        dict(x_grid=(np.nan, 0.0, 0.1)),
    ],
)
def test_generate_synthetic_surface_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        generate_synthetic_surface(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(noise_mode="absolute", noise_level=0.1, noise_dist="bogus"),
        dict(model="bogus_model"),
        dict(noise_mode="bogus_noise_mode", noise_level=0.1),
    ],
)
def test_generate_synthetic_surface_unknown_settings_raise(kwargs):
    with pytest.raises(ValueError):
        generate_synthetic_surface(expiries=(0.25,), x_grid=(-0.1, 0.0, 0.1), **kwargs)


def test_generate_bad_svi_smile_case_smoke():
    case = generate_bad_svi_smile_case(T=1.0, y_domain=(-0.2, 0.2))
    assert case.T == 1.0
    assert case.y_min == -0.2
    assert case.y_max == 0.2
    # "bad" should differ from "good"
    assert case.params_bad != case.params_good
