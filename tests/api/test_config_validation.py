import pytest

import option_pricing.config as config
from option_pricing.config import ImpliedVolConfig, NumericsConfig
from option_pricing.monte_carlo import MCConfig


def test_numerics_config_validation():
    with pytest.raises(ValueError):
        NumericsConfig(abs_tol=0.0)
    with pytest.raises(ValueError):
        NumericsConfig(max_iter=0)


def test_implied_vol_config_validation():
    with pytest.raises(ValueError):
        ImpliedVolConfig(sigma_lo=1.0, sigma_hi=0.5)
    with pytest.raises(ValueError):
        ImpliedVolConfig(bounds_eps=-1e-6)


def test_mc_config_validation():
    with pytest.raises(ValueError):
        MCConfig(n_paths=0)


def test_mc_config_is_not_exported_from_legacy_config_module():
    assert not hasattr(config, "MCConfig")
    assert not hasattr(config, "RandomConfig")
