import numpy as np
import pytest


def test_gj_section51_targets_clamps_and_raises():
    from option_pricing.vol.svi.models import JWParams
    from option_pricing.vol.svi.repair import _gj_section51_targets

    # tiny negative c* gets clamped to 0
    p = 1e-12
    eps = 5e-15
    psi = -(p + eps) / 2.0  # makes c* = -eps (>-1e-14)
    jw = JWParams(v=0.2, psi=psi, p=p, c=0.1, v_tilde=0.1)
    c_ast, vt_ast = _gj_section51_targets(jw)
    assert c_ast == 0.0
    assert vt_ast == 0.0

    # truly negative c* raises
    jw_bad = JWParams(v=0.2, psi=-0.2, p=0.1, c=0.1, v_tilde=0.1)
    with pytest.raises(ValueError, match="Invalid c_ast"):
        _gj_section51_targets(jw_bad)

    # denom <= 0 raises
    jw_denom = JWParams(v=0.2, psi=0.5, p=-1.0, c=0.1, v_tilde=0.1)  # c*=0, denom=-1
    with pytest.raises(ValueError, match="p \\+ c_ast"):
        _gj_section51_targets(jw_denom)


def test_normalized_black_call_intrinsic_branch():
    from option_pricing.vol.svi.repair import _normalized_black_call_from_total_variance

    y = np.array([-0.25, 0.25], dtype=np.float64)
    w = np.array([0.0, 0.0], dtype=np.float64)
    got = _normalized_black_call_from_total_variance(y, w)
    intrinsic = np.maximum(1.0 - np.exp(y), 0.0)
    np.testing.assert_allclose(got, intrinsic, rtol=0.0, atol=0.0)


def test_repair_with_fallback_logs_then_succeeds(monkeypatch):
    from option_pricing.vol.svi import repair as repair_mod
    from option_pricing.vol.svi.diagnostics import ButterflyCheck
    from option_pricing.vol.svi.models import SVIParams

    p_bad = SVIParams(a=0.1, b=1.3, rho=0.7, m=0.1, sigma=0.2)
    p_good = SVIParams(a=0.05, b=0.3, rho=-0.2, m=0.0, sigma=0.25)

    def fake_check(p, *_, **kwargs):
        ok = p == p_good
        ydom = kwargs.get("y_domain_hint", (-1.0, 1.0))
        return ButterflyCheck(
            ok=ok,
            y_domain=ydom,
            min_g=0.01 if ok else -0.01,
            argmin_y=0.0,
            n_stationary=0,
            g_left_inf=0.0,
            g_right_inf=0.0,
            stationary_points=(),
            failure_reason=None if ok else "fake",
        )

    monkeypatch.setattr(repair_mod, "check_butterfly_arbitrage", fake_check)
    monkeypatch.setattr(
        repair_mod,
        "repair_butterfly_jw_optimal",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("jw failed")),
    )
    monkeypatch.setattr(repair_mod, "repair_butterfly_raw", lambda *_a, **_k: p_good)

    out, post, log = repair_mod.repair_butterfly_with_fallback(
        p_bad,
        T=1.0,
        y_domain_hint=(-0.5, 0.5),
        w_floor=0.0,
        try_jw_optimal=True,
        raw_methods=("line_search",),
    )
    assert out == p_good
    assert post.ok is True
    assert [row["method"] for row in log] == ["jw_optimal", "line_search"]
