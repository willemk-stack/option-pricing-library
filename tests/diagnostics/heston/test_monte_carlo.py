from __future__ import annotations

import matplotlib
import numpy as np

import option_pricing.diagnostics.mc_vs_bs.tables as mc_vs_bs_tables
from option_pricing.diagnostics.mc_vs_bs.tables import convergence_table
from option_pricing.models.heston import HestonParams
from option_pricing.pricers.heston import heston_price_call_from_ctx
from option_pricing.pricers.heston_mc import heston_mc_price_call
from option_pricing.types import OptionType

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _params() -> HestonParams:
    return HestonParams(kappa=2.0, vbar=0.04, eta=0.55, rho=-0.70, v=0.05)


def _install_heston_mc_diagnostics(monkeypatch) -> None:
    params = _params()

    def _bs_like_price(p):
        return float(
            heston_price_call_from_ctx(
                strike=p.K,
                ctx=p.ctx,
                tau=p.tau,
                params=params,
            )
        )

    def _mc_price(p, *, cfg):
        return heston_mc_price_call(p, params=params, n_steps=64, cfg=cfg)

    monkeypatch.setattr(mc_vs_bs_tables, "_default_bs_price", lambda: _bs_like_price)
    monkeypatch.setattr(mc_vs_bs_tables, "_default_mc_price", lambda: _mc_price)


def test_heston_mc_convergence_table_has_expected_shape_and_columns(
    make_inputs,
    monkeypatch,
) -> None:
    _install_heston_mc_diagnostics(monkeypatch)
    p = make_inputs(
        S=100.0,
        K=100.0,
        r=0.02,
        q=0.0,
        sigma=0.2,
        T=1.0,
        kind=OptionType.CALL,
    )

    table = convergence_table(p, n_paths_list=[512, 1_024, 2_048], seed=5)

    assert table.shape[0] == 3
    assert table["n_paths"].to_list() == [512, 1_024, 2_048]
    assert {"n_paths", "mc", "se", "bs", "err", "MC", "SE", "BS", "MC-BS"} <= set(
        table.columns
    )
    assert np.all(np.isfinite(table[["mc", "se", "bs", "err"]].to_numpy(dtype=float)))


def test_heston_mc_convergence_table_has_no_plotting_side_effects(
    make_inputs,
    monkeypatch,
) -> None:
    _install_heston_mc_diagnostics(monkeypatch)
    p = make_inputs(
        S=100.0,
        K=100.0,
        r=0.02,
        q=0.0,
        sigma=0.2,
        T=1.0,
        kind=OptionType.CALL,
    )

    before = tuple(plt.get_fignums())
    convergence_table(p, n_paths_list=[256, 512], seed=7)
    after = tuple(plt.get_fignums())

    assert after == before
