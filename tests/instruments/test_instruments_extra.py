from typing import Any, cast

import numpy as np
import pytest

from option_pricing.instruments.digital import DigitalPayoff, digital_put_payoff
from option_pricing.instruments.vanilla import VanillaOption, VanillaPayoff
from option_pricing.types import OptionType


def test_digital_put_payoff_scalar_and_array():
    assert digital_put_payoff(90.0, K=100.0, payout=2.0) == 2.0
    arr = digital_put_payoff(np.array([90.0, 110.0]), K=100.0, payout=2.0)
    np.testing.assert_allclose(arr, np.array([2.0, 0.0]))


def test_digital_and_vanilla_unsupported_kind_raises():
    dp = DigitalPayoff(kind=cast(Any, "weird"), strike=100.0, payout=1.0)
    with pytest.raises(ValueError):
        dp(100.0)

    vp = VanillaPayoff(kind=cast(Any, "weird"), strike=100.0)
    with pytest.raises(ValueError):
        vp(100.0)


def test_vanilla_intrinsic_value_calls_payoff():
    inst = VanillaOption(expiry=1.0, strike=100.0, kind=OptionType.CALL)
    assert inst.intrinsic_value(105.0) == inst.payoff(105.0)
