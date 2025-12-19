from __future__ import annotations

import math

from option_pricing.types import PricingInputs  # adjust name to your actual module


def forward_discounted(p: PricingInputs) -> float:
    """S*e^{-q tau} - K*e^{-r tau} (the RHS of put-call parity)."""
    return p.S * math.exp(-p.q * p.tau) - p.K * math.exp(-p.r * p.tau)


def put_call_parity_residual(*, call: float, put: float, p: PricingInputs) -> float:
    """
    Residual = (C - P) - (S e^{-q tau} - K e^{-r tau}).
    Should be ~0 for European options under consistent inputs.
    """
    return (call - put) - forward_discounted(p)
