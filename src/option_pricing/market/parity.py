from __future__ import annotations

from option_pricing.types import PricingInputs


def forward_discounted(p: PricingInputs) -> float:
    """Prepaid-forward minus discounted strike: ``Fp - K*df``.

    Under continuous carry, put-call parity is::

        C - P = S e^{-q tau} - K e^{-r tau}
              = Fp(tau) - K df(tau)

    With a curves-first setup we compute:

        - ``df(tau)`` from the discount curve
        - ``Fp(tau) = df(tau) * F(tau)`` from (discount, forward)

    This avoids recomputing exponentials in multiple places.
    """
    tau = p.tau
    df = p.df
    fp = p.ctx.prepaid_forward(tau)
    return fp - p.K * df


def put_call_parity_residual(*, call: float, put: float, p: PricingInputs) -> float:
    """Residual = (C - P) - (Fp - K*df). Should be ~0 under consistent inputs."""
    return (float(call) - float(put)) - forward_discounted(p)
