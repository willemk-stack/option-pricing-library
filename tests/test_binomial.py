import math

from option_pricing.models.binomial_crr import (
    binom_call_from_inputs,
    binom_put_from_inputs,
)
from option_pricing.models.bs import bs_call_from_inputs


def test_binomial_converges_toward_bs_as_steps_increase(make_inputs):
    """CRR binomial call should approach BS call as steps increase."""
    p = make_inputs(S=100.0, K=105.0, r=0.04, sigma=0.22, T=1.0)

    bs = float(bs_call_from_inputs(p))

    steps = [25, 50, 100, 200, 400]
    errs = [abs(binom_call_from_inputs(p, n_steps=n) - bs) for n in steps]

    # not strictly monotone, but high-N should be better than low-N
    assert errs[-1] <= errs[0]
    # keep this loose; tighten once you know typical accuracy
    assert errs[-1] <= 2e-2


def test_binomial_put_call_parity_approximately(make_inputs):
    """Binomial should satisfy parity approximately for sufficiently many steps."""
    p = make_inputs(S=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)
    n_steps = 400

    C = float(binom_call_from_inputs(p, n_steps=n_steps))
    P = float(binom_put_from_inputs(p, n_steps=n_steps))

    df = math.exp(-p.r * (p.T - p.t))
    rhs = p.S - p.K * df

    assert abs((C - P) - rhs) <= 5e-3
