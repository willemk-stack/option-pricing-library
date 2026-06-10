from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from option_pricing.viz.core import DistSpec, hist_with_curve, plot_specs  # noqa: E402


def test_hist_with_curve_handles_empty_samples() -> None:
    fig, ax = plt.subplots()
    out = hist_with_curve(ax, np.array([np.nan, np.inf]), title="empty")

    assert out is ax
    assert "empty" in ax.get_title()
    assert not ax.axison
    plt.close(fig)


def test_hist_with_curve_scales_curve_when_density_false_and_calls_post_ax() -> None:
    called: dict[str, object] = {}

    def curve(xs: np.ndarray) -> np.ndarray:
        return np.ones_like(xs)

    def post_ax(ax, x: np.ndarray) -> None:
        called["ax"] = ax
        called["n"] = x.size

    fig, ax = plt.subplots()
    hist_with_curve(
        ax,
        np.array([1.0, 2.0, 3.0, np.nan]),
        curve,
        bins=2,
        density=False,
        n_grid=5,
        xlim_quantiles=(0.0, 1.0),
        title="panel",
        xlabel="x",
        ylabel="count",
        show_mean=True,
        show_median=True,
        post_ax=post_ax,
        hist_kwargs={"alpha": 0.5},
        curve_kwargs={"linewidth": 1.0},
    )

    assert called == {"ax": ax, "n": 3}
    assert ax.get_title() == "panel"
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "count"
    assert len(ax.lines) >= 3  # curve + mean + median
    plt.close(fig)


def test_plot_specs_rejects_nonpositive_ncols() -> None:
    with pytest.raises(ValueError, match="ncols"):
        plot_specs([], ncols=0)


def test_plot_specs_hides_unused_axes_and_shares_limits() -> None:
    specs = [
        DistSpec(name="left", samples=np.array([1.0, 2.0, 3.0]), title="L"),
        DistSpec(name="right", samples=np.array([10.0, 11.0, 12.0]), title="R"),
        DistSpec(name="third", samples=np.array([100.0, 101.0, 102.0]), title="Third"),
        DistSpec(name="last", samples=np.array([200.0, 201.0, 202.0]), title="Last"),
    ]

    fig, axes = plot_specs(
        specs,
        ncols=2,
        share_xlim=True,
        share_ylim=True,
        suptitle="all panels",
    )

    assert axes.shape == (4,)
    assert all(ax.axison for ax in axes)
    assert fig._suptitle is not None
    active = axes[:3]
    assert len({tuple(ax.get_xlim()) for ax in active}) == 1
    assert len({tuple(ax.get_ylim()) for ax in active}) == 1
    plt.close(fig)


def test_plot_specs_reports_missing_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    real_modules = dict(sys.modules)

    def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "matplotlib.pyplot":
            raise ModuleNotFoundError("matplotlib")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)
    sys.modules.pop("matplotlib.pyplot", None)

    try:
        with pytest.raises(ModuleNotFoundError, match="plot_specs requires matplotlib"):
            plot_specs([DistSpec(name="x", samples=np.array([1.0]))])
    finally:
        sys.modules.clear()
        sys.modules.update(real_modules)
