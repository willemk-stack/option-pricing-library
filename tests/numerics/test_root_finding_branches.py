import pytest

from option_pricing.exceptions import (
    DerivativeTooSmallError,
    NoBracketError,
    NoConvergenceError,
    NotBracketedError,
)
from option_pricing.numerics.root_finding import (
    bisection_method,
    bracketed_newton,
    ensure_bracket,
    newton_method,
)


def test_bisection_endpoints_and_not_bracketed():
    def fn(x):
        return x

    res_lo = bisection_method(fn, 0.0, 1.0)
    assert res_lo.converged
    assert res_lo.root == 0.0

    def fn_hi(x):
        return x - 1.0

    res_hi = bisection_method(fn_hi, 0.0, 1.0)
    assert res_hi.converged
    assert res_hi.root == 1.0

    with pytest.raises(NotBracketedError):
        bisection_method(lambda x: x**2 + 1.0, -1.0, 1.0)


def test_bisection_interval_tolerance_path():
    res = bisection_method(
        lambda x: x - 0.25,
        0.0,
        1.0,
        tol_f=1e-12,
        tol_x=0.6,
        max_iter=2,
    )
    assert res.converged
    assert 0.0 < res.root < 1.0


def test_bisection_domain_clamp():
    res = bisection_method(
        lambda x: x - 0.25,
        -1.0,
        2.0,
        domain=(0.0, 1.0),
    )
    assert res.converged
    assert 0.0 <= res.root <= 1.0


def test_newton_derivative_too_small():
    with pytest.raises(DerivativeTooSmallError):
        newton_method(
            lambda x: x + 1.0,
            -1.0,
            1.0,
            x0=0.25,
            dFn=lambda x: 0.0,
        )


def test_newton_step_tolerance_converges():
    res = newton_method(
        lambda x: x - 1.0,
        0.0,
        2.0,
        x0=1.0 + 1e-10,
        dFn=lambda x: 1.0,
        tol_x=1e-6,
    )
    assert res.converged
    assert abs(res.root - 1.0) < 1e-8


def test_newton_uses_numerical_derivative():
    res = newton_method(
        lambda x: x - 2.0,
        0.0,
        4.0,
        x0=0.5,
        dFn=None,
    )
    assert res.converged
    assert abs(res.root - 2.0) < 1e-6


def test_newton_no_convergence():
    with pytest.raises(NoConvergenceError):
        newton_method(
            lambda x: x - 10.0,
            0.0,
            1.0,
            x0=0.0,
            dFn=lambda x: 1.0,
            tol_f=0.0,
            tol_x=0.0,
            max_iter=1,
        )


def test_ensure_bracket_errors_and_success():
    with pytest.raises(ValueError):
        ensure_bracket(lambda x: x - 1.0, 1.0, 1.0)

    with pytest.raises(ValueError):
        ensure_bracket(lambda x: x - 1.0, 0.0, 1.0, grow=1.0)

    with pytest.raises(ValueError):
        ensure_bracket(lambda x: x - 1.0, 0.0, 2.0, domain=(1.0, 1.0))

    with pytest.raises(NoBracketError):
        ensure_bracket(lambda x: 1.0, 1.0, 2.0, hi_max=3.0, max_steps=2)

    with pytest.raises(NoBracketError):
        ensure_bracket(lambda x: -1.0, 1.0, 2.0, hi_max=3.0, max_steps=2)

    lo, hi = ensure_bracket(lambda x: x - 2.0, 0.0, 1.0, hi_max=4.0)
    assert lo < hi
    assert hi >= 2.0

    lo2, hi2 = ensure_bracket(lambda x: x - 2.0, 0.0, 4.0, hi_max=1.0)
    assert lo2 < hi2


def test_bracketed_newton_autobracket_and_fallback():
    res = bracketed_newton(
        lambda x: x - 2.0,
        0.0,
        1.0,
        dFn=lambda x: 1.0,
        hi_max=4.0,
    )
    assert res.converged
    assert abs(res.root - 2.0) < 1e-8

    res_bisect = bracketed_newton(
        lambda x: x - 1.0,
        0.0,
        2.0,
        dFn=lambda x: 0.0,
    )
    assert res_bisect.converged
    assert abs(res_bisect.root - 1.0) < 1e-6


def test_bracketed_newton_no_convergence():
    with pytest.raises(NoConvergenceError):
        bracketed_newton(
            lambda x: x - 2.0,
            0.0,
            4.0,
            x0=1.0,
            dFn=lambda x: 1.0,
            tol_f=0.0,
            tol_x=0.0,
            max_iter=1,
        )
