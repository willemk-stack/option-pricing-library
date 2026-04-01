import numpy as np
import pytest


def test_calibrate_svi_robustify_everything_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers calibrate_svi Branch A: robust_data_only=False (SciPy robust loss).

    We monkeypatch least_squares so the test is fast and deterministic, while
    still exercising the post-solve robust-weight computation.
    """
    import option_pricing.vol.svi.calibrate as cal_mod
    from option_pricing.vol.svi import diagnostics as diag_mod
    from option_pricing.vol.svi.calibrate import calibrate_svi
    from option_pricing.vol.svi.diagnostics import ButterflyCheck
    from option_pricing.vol.svi.math import svi_total_variance
    from option_pricing.vol.svi.models import SVIParams

    # Small synthetic dataset.
    y = np.linspace(-0.5, 0.5, 11, dtype=np.float64)
    p_true = SVIParams(a=0.02, b=0.60, rho=-0.15, m=0.05, sigma=0.30)
    w_obs = svi_total_variance(y, p_true)

    # Start from a noticeably different x0 so robust weights are not all 1.
    x0 = SVIParams(a=0.01, b=0.20, rho=0.40, m=-0.10, sigma=0.60)

    # Make diagnostics fast (build_svi_diagnostics calls this with large n_scan).
    def _fast_bfly(*_args, **_kwargs) -> ButterflyCheck:
        return ButterflyCheck(
            ok=True,
            y_domain=(-1.0, 1.0),
            min_g=1.0,
            argmin_y=0.0,
            n_stationary=0,
            g_left_inf=1.0,
            g_right_inf=1.0,
            stationary_points=(),
            failure_reason=None,
        )

    monkeypatch.setattr(diag_mod, "check_butterfly_arbitrage", _fast_bfly)

    # Patch least_squares to instantly "converge" to the initial guess.
    class _Res:
        def __init__(self, x: np.ndarray):
            self.success = True
            self.x = np.asarray(x, dtype=np.float64)
            self.message = "ok"
            self.nfev = 1
            self.cost = 0.0
            self.optimality = 0.0

    def _fake_least_squares(*, x0, **_kwargs):
        return _Res(x0)

    monkeypatch.setattr(cal_mod, "least_squares", _fake_least_squares)

    fit = calibrate_svi(
        y=y,
        w_obs=w_obs,
        x0=x0,
        robust_data_only=False,  # <-- Branch A
        loss="huber",
        f_scale=1e-4,  # amplify residuals so rhoprime < 1 is likely
        slice_T=1.0,
    )

    # Robust weights were computed (and are nonnegative / finite).
    rw_min = float(fit.diag.checks.robust_weights_min)
    rw_max = float(fit.diag.checks.robust_weights_max)
    assert np.isfinite(rw_min) and np.isfinite(rw_max)
    assert 0.0 <= rw_min <= rw_max <= 1.0


def test_calibrate_svi_unknown_refit_after_repair_mode_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Covers the ValueError branch for unknown refit_after_repair_mode."""
    import option_pricing.vol.svi as svi_pkg
    import option_pricing.vol.svi.calibrate as cal_mod
    from option_pricing.vol.svi.calibrate import calibrate_svi
    from option_pricing.vol.svi.diagnostics import ButterflyCheck
    from option_pricing.vol.svi.math import svi_total_variance
    from option_pricing.vol.svi.models import SVIParams

    y = np.linspace(-0.25, 0.25, 7, dtype=np.float64)
    p = SVIParams(a=0.02, b=0.40, rho=0.0, m=0.0, sigma=0.25)
    w_obs = svi_total_variance(y, p)

    # Fast solver stub.
    class _Res:
        def __init__(self, x: np.ndarray):
            self.success = True
            self.x = np.asarray(x, dtype=np.float64)
            self.message = "ok"
            self.nfev = 1
            self.cost = 0.0
            self.optimality = 0.0

    def _fake_least_squares(*, x0, **_kwargs):
        return _Res(x0)

    monkeypatch.setattr(cal_mod, "least_squares", _fake_least_squares)

    # Make repair + checks no-ops so we can reach the mode validation quickly.
    def _bfly_ok(*_a, **_k) -> ButterflyCheck:
        return ButterflyCheck(
            ok=True,
            y_domain=(-1.0, 1.0),
            min_g=1.0,
            argmin_y=0.0,
            n_stationary=0,
            g_left_inf=1.0,
            g_right_inf=1.0,
            stationary_points=(),
            failure_reason=None,
        )

    monkeypatch.setattr(svi_pkg, "check_butterfly_arbitrage", _bfly_ok)
    monkeypatch.setattr(svi_pkg, "repair_butterfly_raw", lambda p, **_k: p)
    monkeypatch.setattr(cal_mod, "repair_butterfly_raw", lambda p, **_k: p)

    with pytest.raises(ValueError, match="Unknown refit_after_repair_mode"):
        calibrate_svi(
            y=y,
            w_obs=w_obs,
            loss="linear",
            robust_data_only=True,
            repair_butterfly=True,
            refit_after_repair=True,
            refit_after_repair_mode="definitely_not_a_mode",  # type: ignore[arg-type]
            slice_T=1.0,
        )
