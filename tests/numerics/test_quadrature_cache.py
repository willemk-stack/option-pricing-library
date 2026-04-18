from __future__ import annotations

import numpy as np
import pytest

from option_pricing.numerics import (
    CompositeRule,
    PanelSpacing,
    QuadratureConfig,
    build_gauss_legendre_rule,
)
from option_pricing.numerics.quadrature import (
    composite_fixed_rule,
    composite_gauss_legendre,
    gauss_legendre_nodes_weights,
)


def test_build_gauss_legendre_rule_returns_composite_rule_and_reuses_cache() -> None:
    build_gauss_legendre_rule.cache_clear()
    cfg = QuadratureConfig(
        u_max=12.0,
        n_panels=3,
        nodes_per_panel=4,
        panel_spacing=PanelSpacing.UNIFORM,
    )

    first = build_gauss_legendre_rule(cfg)
    second = build_gauss_legendre_rule(cfg)
    cache_info = build_gauss_legendre_rule.cache_info()

    assert isinstance(first, CompositeRule)
    assert second is first
    assert cache_info.misses == 1
    assert cache_info.hits == 1


def test_composite_gauss_legendre_matches_manual_fixed_rule() -> None:
    cfg = QuadratureConfig(u_max=2.0, n_panels=4, nodes_per_panel=6)

    def _eval_fn(u):
        u_arr = np.asarray(u, dtype=np.float64)
        return np.asarray(np.exp(-u_arr) * (1.0 + u_arr), dtype=np.float64)

    nodes, weights = gauss_legendre_nodes_weights(cfg.nodes_per_panel)
    manual = composite_fixed_rule(_eval_fn, cfg, nodes, weights)
    cached = composite_gauss_legendre(_eval_fn, cfg)

    assert cached == pytest.approx(manual, rel=0.0, abs=1e-14)
