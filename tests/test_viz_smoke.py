import numpy as np
import pytest


def test_plot_specs_smoke():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg", force=True)

    from option_pricing.viz.core import DistSpec, plot_specs

    rng = np.random.default_rng(0)
    specs = [
        DistSpec(name="A", samples=rng.normal(size=500)),
        DistSpec(name="B", samples=rng.normal(loc=0.5, scale=1.2, size=500)),
    ]

    fig, axs = plot_specs(specs, ncols=2, suptitle="Smoke")
    assert fig is not None
    assert axs is not None

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_sweep_smoke_and_save(tmp_path, make_inputs):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg", force=True)

    from option_pricing.diagnostics.greeks.sweep import SweepResult
    from option_pricing.viz.plot_sweeps import plot_sweep

    x = np.linspace(80.0, 120.0, 9)
    res = SweepResult(
        x=x,
        price=np.maximum(x - 100.0, 0.0),
        delta=(x > 100.0).astype(float),
        gamma=np.zeros_like(x),
        vega=np.ones_like(x),
        theta=-0.5 * np.ones_like(x),
    )

    base = make_inputs(S=100.0, K=100.0, r=0.02, q=0.0, sigma=0.2, T=1.0, t=0.0)
    out = tmp_path / "sweep.png"

    fig, axs = plot_sweep(res, base=base, show=False, savepath=out)
    assert out.exists()
    assert axs.shape == (3, 2)

    import matplotlib.pyplot as plt

    plt.close(fig)
