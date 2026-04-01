import numpy as np
import pytest

from option_pricing.data_generators.recipes import (
    generate_synthetic_surface_latent_noarb,
)


def test_generate_synthetic_surface_latent_noarb_enforce_false_smoke():
    # Keep the grid tiny so this stays fast
    res = generate_synthetic_surface_latent_noarb(
        enforce=False,
        max_rounds=2,
        model="poly",
        expiries=(0.25, 0.5),
        x_grid=np.linspace(-0.15, 0.15, 7),
        noise_mode="absolute",
        noise_level=0.01,
        missing_prob=0.25,
        seed=123,
    )

    # enforce=False => should return after first attempt
    assert len(res.tuning_log) == 1

    # Truth generation forces missing_prob=0 => rows_true should be full grid
    assert len(res.rows_true) == (2 * 7)  # len(expiries) * len(x_grid)

    # Basic shape sanity on observed synthetic
    assert res.synthetic.iv_obs.size == len(res.synthetic.rows_obs)


def test_generate_synthetic_surface_latent_noarb_max_rounds_validation():
    with pytest.raises(ValueError):
        generate_synthetic_surface_latent_noarb(max_rounds=0)
