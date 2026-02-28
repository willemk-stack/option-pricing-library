import numpy as np
import pytest

from option_pricing.data_generators.synthetic_surface import generate_bad_svi_smile_case
from option_pricing.vol.svi.calibrate import calibrate_svi
from option_pricing.vol.svi.math import svi_total_variance


def test_calibrate_svi_smoke_linear_success():
    case = generate_bad_svi_smile_case(T=1.0, y_domain=(-0.25, 0.25))
    y = np.linspace(case.y_min, case.y_max, 7)
    w_obs = svi_total_variance(y, case.params_good)

    fit = calibrate_svi(
        y=y,
        w_obs=w_obs,
        loss="linear",
        robust_data_only=True,
        slice_T=case.T,
    )

    params = fit.params
    vals = np.array([params.a, params.b, params.rho, params.m, params.sigma])
    assert np.all(np.isfinite(vals))


def test_calibrate_svi_shape_mismatch_raises():
    y = np.array([0.0, 0.1])
    w_obs = np.array([0.01])
    with pytest.raises(ValueError):
        calibrate_svi(y=y, w_obs=w_obs)
