from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from option_pricing.models.local_vol.pde import _coord_key, local_vol_pde_coeffs


def test_coord_key_accepts_enum_like_names_and_rejects_unknowns() -> None:
    assert _coord_key("s") == "S"
    assert _coord_key("ln_s") == "LOG_S"
    assert _coord_key(SimpleNamespace(name="LOG")) == "LOG_S"

    with pytest.raises(ValueError, match="Unsupported coord"):
        _coord_key("strike")


def test_log_s_coefficients_apply_tau_floor_sigma_floor_and_vector_cache() -> None:
    calls: list[tuple[tuple[float, ...], float]] = []

    def local_var(S: np.ndarray, tau: float) -> np.ndarray:
        calls.append((tuple(np.asarray(S, dtype=float)), tau))
        return np.array([[np.nan], [-1.0], [0.04]])

    coeffs = local_vol_pde_coeffs(
        coord="LOG_S",
        local_var=local_var,
        r=0.03,
        q=0.01,
        tau_floor=0.25,
        sigma2_floor=0.02,
        sigma2_cap=0.03,
    )
    x = np.log(np.array([90.0, 100.0, 110.0]))

    a = coeffs.a(x, tau=0.0)
    b = coeffs.b(list(x), tau=0.0)

    np.testing.assert_allclose(a, np.array([0.01, 0.01, 0.015]))
    np.testing.assert_allclose(b, np.array([0.01, 0.01, 0.005]))
    assert len(calls) == 1  # b(list(x)) reuses equal-valued vector cache
    assert calls[0][1] == 0.25
    np.testing.assert_allclose(coeffs.c(x, 0.0), np.full(3, -0.03))


def test_s_coefficients_handle_scalar_cache_and_sigma_cap() -> None:
    calls: list[tuple[float, float]] = []

    def local_var(S: np.ndarray, tau: float) -> float:
        calls.append((float(np.asarray(S)), tau))
        return 4.0

    coeffs = local_vol_pde_coeffs(
        coord="S",
        local_var=local_var,
        r=0.05,
        q=0.02,
        sigma2_floor=0.01,
        sigma2_cap=0.25,
    )

    first = coeffs.a(np.array(2.0), 1.0)
    second = coeffs.a(np.array(2.0), 1.0)

    assert float(first) == pytest.approx(0.5)
    assert float(second) == pytest.approx(0.5)
    assert len(calls) == 1
    assert float(coeffs.b(np.array(2.0), 1.0)) == pytest.approx(0.06)
    assert float(coeffs.c(np.array(2.0), 1.0)) == pytest.approx(-0.05)


def test_vector_local_var_scalar_return_broadcasts_to_input_shape() -> None:
    coeffs = local_vol_pde_coeffs(
        coord="S",
        local_var=lambda S, tau: np.array(0.04),
        r=0.0,
        q=0.0,
    )

    out = coeffs.a(np.array([1.0, 2.0, 3.0]), 1.0)

    np.testing.assert_allclose(out, 0.5 * 0.04 * np.array([1.0, 4.0, 9.0]))
