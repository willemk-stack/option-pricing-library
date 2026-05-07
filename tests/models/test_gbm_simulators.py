import numpy as np

from option_pricing.market.curves import (
    FlatCarryForwardCurve,
    FlatDiscountCurve,
    PricingContext,
)
from option_pricing.models.gbm import GBMPathSimulator, GBMTerminalSimulator
from option_pricing.monte_carlo import MCConfig, RandomConfig


def _pricing_context() -> PricingContext:
    return PricingContext(
        spot=100.0,
        discount=FlatDiscountCurve(0.03),
        forward=FlatCarryForwardCurve(spot=100.0, r=0.03, q=0.01),
    )


def test_gbm_terminal_simulator_returns_terminal_shape() -> None:
    simulator = GBMTerminalSimulator(sigma=0.2)
    cfg = MCConfig(n_paths=7, random=RandomConfig(seed=5))

    terminal = simulator.simulate_terminal(ctx=_pricing_context(), tau=1.0, cfg=cfg)

    assert terminal.shape == (7,)
    assert np.all(terminal > 0.0)


def test_gbm_path_simulator_returns_path_shape() -> None:
    simulator = GBMPathSimulator(sigma=0.2, n_steps=6)
    cfg = MCConfig(n_paths=5, random=RandomConfig(seed=7))

    paths = simulator.simulate_paths(ctx=_pricing_context(), tau=1.0, cfg=cfg)

    assert paths.shape == (5, 7)
    assert np.all(paths[:, 0] == 100.0)
