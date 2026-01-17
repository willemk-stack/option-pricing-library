"""Small conversion helpers.

The library historically used :class:`~option_pricing.types.OptionSpec` and
:class:`~option_pricing.types.PricingInputs` as the primary user inputs.

As the codebase grows, it is useful to split "what is being priced" (an
instrument) from "how it is priced" (a pricer). The functions here provide a
low-friction bridge so you can migrate pricers progressively without breaking
existing notebooks.
"""

from __future__ import annotations

from ..types import OptionSpec, PricingInputs
from .base import ExerciseStyle
from .vanilla import VanillaOption


def from_option_spec(
    spec: OptionSpec,
    *,
    tau: float | None = None,
    exercise: ExerciseStyle = ExerciseStyle.EUROPEAN,
) -> VanillaOption:
    """Convert an :class:`~option_pricing.types.OptionSpec` to a :class:`VanillaOption`.

    Parameters
    ----------
    spec
        Vanilla option specification (kind, strike, expiry).
    tau
        Optional time to expiry. If not provided, ``spec.expiry`` is assumed to
        already be time-to-expiry.
    exercise
        Exercise style (defaults to European).

    Notes
    -----
    ``OptionSpec.expiry`` is historically an absolute expiry time ``T`` used with
    a valuation time ``t`` in :class:`PricingInputs`. The instruments package, by
    contrast, uses a single number interpreted as **time to expiry**.
    """
    expiry = float(spec.expiry if tau is None else tau)
    return VanillaOption(
        expiry=expiry,
        strike=float(spec.strike),
        kind=spec.kind,
        exercise=exercise,
    )


def from_pricing_inputs(
    p: PricingInputs, *, exercise: ExerciseStyle = ExerciseStyle.EUROPEAN
) -> VanillaOption:
    """Convert :class:`~option_pricing.types.PricingInputs` to a :class:`VanillaOption`.

    Uses ``p.tau`` for time-to-expiry.
    """
    return from_option_spec(p.spec, tau=p.tau, exercise=exercise)
