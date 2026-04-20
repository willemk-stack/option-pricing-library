from __future__ import annotations

import numpy as np
import pytest

from option_pricing.models.heston.params import HestonParams


def test_heston_params_as_array_preserves_field_order() -> None:
    params = HestonParams(
        kappa=2.5,
        vbar=0.04,
        eta=0.60,
        rho=-0.75,
        v=0.05,
    )

    np.testing.assert_allclose(
        params.as_array(),
        np.array([2.5, 0.04, 0.60, -0.75, 0.05], dtype=np.float64),
        atol=0.0,
        rtol=0.0,
    )


def test_heston_params_roundtrip_transforms_preserve_core_fields() -> None:
    params = HestonParams(
        kappa=1.7,
        vbar=0.032,
        eta=0.48,
        rho=-0.42,
        v=0.027,
    )

    raw = params.TransformToUnconstrained()
    restored = HestonParams.TransformToConstrained(raw)

    np.testing.assert_allclose(
        restored.as_array(),
        params.as_array(),
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_heston_params_alias_methods_match_pascal_case_methods() -> None:
    params = HestonParams(
        kappa=2.0,
        vbar=0.04,
        eta=0.55,
        rho=-0.70,
        v=0.05,
    )

    raw_pascal = params.TransformToUnconstrained()
    raw_snake = params.transform_to_unconstrained()
    restored_pascal = HestonParams.TransformToConstrained(raw_pascal)
    restored_snake = HestonParams.transform_to_constrained(raw_snake)

    np.testing.assert_allclose(raw_pascal, raw_snake, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        restored_pascal.as_array(),
        restored_snake.as_array(),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kappa": 0.0, "vbar": 0.04, "eta": 0.55, "rho": 0.0, "v": 0.05}, "kappa"),
        ({"kappa": 2.0, "vbar": -1.0e-6, "eta": 0.55, "rho": 0.0, "v": 0.05}, "bar"),
        ({"kappa": 2.0, "vbar": 0.04, "eta": -1.0e-6, "rho": 0.0, "v": 0.05}, "eta"),
        ({"kappa": 2.0, "vbar": 0.04, "eta": 0.55, "rho": 1.1, "v": 0.05}, "rho"),
        ({"kappa": 2.0, "vbar": 0.04, "eta": 0.55, "rho": 0.0, "v": -1.0e-6}, r"\$v\$"),
        (
            {"kappa": np.nan, "vbar": 0.04, "eta": 0.55, "rho": 0.0, "v": 0.05},
            "finite",
        ),
    ],
)
def test_heston_params_validation_rejects_invalid_inputs(
    kwargs: dict[str, float],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        HestonParams(**kwargs)


def test_heston_params_transform_to_constrained_rejects_bad_raw_vectors() -> None:
    with pytest.raises(ValueError, match="Expected 5 unconstrained parameters"):
        HestonParams.TransformToConstrained([0.0, 1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="must be finite"):
        HestonParams.TransformToConstrained([0.0, 1.0, np.nan, 3.0, 4.0])
