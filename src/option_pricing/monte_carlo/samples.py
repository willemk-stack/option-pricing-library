"""Sample validation and effective-sample helpers for Monte Carlo pricing."""

from __future__ import annotations

import numpy as np

from ..typing import FloatArray, FloatDType
from .estimators import ControlVariate, apply_control_variate, pair_antithetic


def as_1d_samples(
    values: object,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> FloatArray:
    samples = np.asarray(values, dtype=FloatDType)
    if samples.shape != expected_shape:
        raise ValueError(
            f"{name} samples must have shape {expected_shape}; got {samples.shape}"
        )
    return samples


def effective_samples_from_payoffs(
    payoff_values: FloatArray,
    *,
    antithetic: bool,
) -> FloatArray:
    if not antithetic:
        return payoff_values

    if payoff_values.shape[0] % 2 != 0:
        raise ValueError("antithetic=True requires an even number of payoff samples")

    n_pairs = payoff_values.shape[0] // 2
    return pair_antithetic(payoff_values[:n_pairs], payoff_values[n_pairs:])


def effective_payoff_samples(
    *,
    terminal: FloatArray,
    payoff_values: FloatArray,
    antithetic: bool,
    control: ControlVariate | None,
) -> FloatArray:
    if not antithetic:
        samples = payoff_values
        if control is not None:
            control_values = as_1d_samples(
                control.values(terminal),
                name="control",
                expected_shape=terminal.shape,
            )
            samples = apply_control_variate(samples, control_values, control.mean)
        return samples

    samples = effective_samples_from_payoffs(
        payoff_values,
        antithetic=antithetic,
    )

    if control is not None:
        n_pairs = terminal.shape[0] // 2
        control_values = as_1d_samples(
            control.values(terminal),
            name="control",
            expected_shape=terminal.shape,
        )
        control_pairs = pair_antithetic(
            control_values[:n_pairs],
            control_values[n_pairs:],
        )
        samples = apply_control_variate(samples, control_pairs, control.mean)

    return samples
