import numpy as np
import pytest

from option_pricing.data_generators.synthetic_surface import generate_bad_svi_smile_case
from option_pricing.vol.svi import repair as repair_mod
from option_pricing.vol.svi.diagnostics import ButterflyCheck
from option_pricing.vol.svi.models import SVIParams
from option_pricing.vol.svi.repair import (
    repair_butterfly_jw_optimal,
    repair_butterfly_raw,
)


def test_repair_jw_optimal_short_circuit_on_good_params():
    case = generate_bad_svi_smile_case(T=1.0, y_domain=(-0.25, 0.25))
    y_obj = np.linspace(case.y_min, case.y_max, 9)
    y_pen = np.linspace(case.y_min, case.y_max, 21)

    repaired = repair_butterfly_jw_optimal(
        case.params_good,
        T=case.T,
        y_obj=y_obj,
        y_penalty=y_pen,
        y_domain_hint=(case.y_min, case.y_max),
        w_floor=1e-12,
        max_nfev=50,
    )

    assert repaired == case.params_good


def test_repair_raw_fixes_bad_params(monkeypatch: pytest.MonkeyPatch):
    case = generate_bad_svi_smile_case(T=1.0, y_domain=(-0.25, 0.25))
    base = case.params_good
    bad = SVIParams(
        a=base.a,
        b=base.b * 1.2,
        rho=base.rho,
        m=base.m,
        sigma=base.sigma,
    )

    def _check_false(*_args, **_kwargs) -> ButterflyCheck:
        return ButterflyCheck(
            ok=False,
            y_domain=(case.y_min, case.y_max),
            min_g=-0.01,
            argmin_y=0.0,
            n_stationary=0,
            g_left_inf=-0.01,
            g_right_inf=0.01,
            stationary_points=(),
            failure_reason="fake",
            lee_ok=False,
            lee_left_ok=False,
            lee_right_ok=False,
            lee_left_slope=-1.0,
            lee_right_slope=1.0,
            lee_slope_cap=2.0,
        )

    def _fake_find(*_args, **_kwargs):
        candidate = SVIParams(
            a=base.a,
            b=base.b,
            rho=base.rho,
            m=base.m,
            sigma=base.sigma,
        )
        ok = ButterflyCheck(
            ok=True,
            y_domain=(case.y_min, case.y_max),
            min_g=0.01,
            argmin_y=0.0,
            n_stationary=0,
            g_left_inf=0.01,
            g_right_inf=0.01,
            stationary_points=(),
            failure_reason=None,
            lee_ok=True,
            lee_left_ok=True,
            lee_right_ok=True,
            lee_left_slope=-0.5,
            lee_right_slope=0.5,
            lee_slope_cap=2.0,
        )
        return 0.5, candidate, ok

    monkeypatch.setattr(repair_mod, "check_butterfly_arbitrage", _check_false)
    monkeypatch.setattr(repair_mod, "_find_min_feasible_lambda", _fake_find)

    repaired = repair_butterfly_raw(
        bad,
        T=case.T,
        y_domain_hint=(case.y_min, case.y_max),
        w_floor=1e-12,
        method="line_search",
    )

    assert repaired != bad
