from typing import Any, cast

import pytest

from option_pricing.exceptions import NoBracketError, NotBracketedError
from option_pricing.numerics.root_finding import (
    RootMethod,
    bisection_method,
    ensure_bracket,
    get_root_method,
)


def test_bisection_raises_when_not_bracketed():
    with pytest.raises(NotBracketedError):
        bisection_method(lambda x: x * x + 1.0, -1.0, 2.0)


def test_ensure_bracket_expands_and_clamps_domain():
    # domain clamp hits the internal _clamp branches too
    lo, hi = ensure_bracket(
        lambda x: x - 0.5,
        lo=-1.0,
        hi=0.1,
        hi_max=2.0,
        domain=(0.0, 2.0),
        max_steps=50,
    )
    assert lo == 0.0
    assert hi > 0.5


def test_ensure_bracket_raises_no_bracket():
    with pytest.raises(NoBracketError):
        ensure_bracket(lambda x: 1.0, lo=0.0, hi=0.1, hi_max=1.0, max_steps=5)


def test_get_root_method_registry_and_invalid():
    assert get_root_method(RootMethod.BISECTION) is bisection_method
    with pytest.raises(ValueError):
        get_root_method(cast(Any, "nope"))
