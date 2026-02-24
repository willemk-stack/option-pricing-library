from __future__ import annotations

from typing import Any

import numpy as np

from .compute import (
    LocalVolCompareReport,
    LocalVolGridReport,
    calendar_dW,
    calendar_dW_from_report,
    call_prices_from_smile,
    first_failing_smile,
    get_smile_at_T,
    surface_domain_report,
    surface_slices,
    svi_fit_table,
    svi_residuals_df,
)


def _get_plt():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Plotting requires matplotlib. Install it with: pip install matplotlib"
        ) from e
    return plt


def plot_smile_slices(
    surface: Any,
    *,
    forward,
    title: str = "Surface — smile slices",
    figsize=(9, 4),
):
    """Plot all smiles as strike vs implied vol curves."""
    plt = _get_plt()
    plt.figure(figsize=figsize)

    for sl in surface_slices(surface, forward=forward):
        plt.plot(sl.K, sl.iv, marker="o", linewidth=1, label=f"T={sl.T:g}y")

    plt.xlabel("Strike K")
    plt.ylabel("Implied vol")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_queried_smile(
    *,
    K: np.ndarray,
    iv: np.ndarray,
    T: float,
    title: str | None = None,
    figsize=(9, 4),
):
    """Plot a queried smile curve (K vs iv)."""
    plt = _get_plt()
    plt.figure(figsize=figsize)
    plt.plot(K, iv, linewidth=2)
    plt.xlabel("Strike K")
    plt.ylabel("Implied vol")
    plt.title(title or f"Queried smile at T={float(T):g}y")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_call_monotonicity_diagnostics(
    surface: Any,
    *,
    T: float,
    forward,
    df,
    bad_indices: np.ndarray,
    bs_model: Any | None = None,
    figsize=(9, 4),
):
    """Plot call prices vs strike and highlight violating intervals."""
    plt = _get_plt()
    K, C, _iv = call_prices_from_smile(
        surface, T=T, forward=forward, df=df, bs_model=bs_model
    )

    plt.figure(figsize=figsize)
    plt.plot(K, C, marker="o", linewidth=1.5, label="Call price")

    bad_i = np.asarray(bad_indices, dtype=int)
    if bad_i.size:
        plt.scatter(K[bad_i], C[bad_i], s=60, label="Violation start")
        plt.scatter(K[bad_i + 1], C[bad_i + 1], s=60, label="Violation end")

    plt.xlabel("Strike K")
    plt.ylabel("Call price (discounted)")
    plt.title(
        f"Call monotonicity check at T={float(T):g}y — bad intervals={bad_i.size}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_calendar_heatmap(
    surface: Any,
    *,
    x_grid: np.ndarray,
    title: str = "Calendar check — negative cells indicate violations",
    figsize=(10, 4),
):
    """Plot Δw heatmap over expiry steps and x_grid."""
    plt = _get_plt()
    xg = np.asarray(x_grid, dtype=float)
    dW = calendar_dW(surface, x_grid=xg)

    plt.figure(figsize=figsize)
    plt.imshow(dW, aspect="auto", origin="lower")
    plt.colorbar(label=r"Δw = w(T_{i+1},x) - w(T_i,x)")
    plt.yticks(
        np.arange(len(surface.smiles) - 1),
        [
            f"{float(surface.smiles[i].T):g}→{float(surface.smiles[i + 1].T):g}"
            for i in range(len(surface.smiles) - 1)
        ],
    )
    plt.xticks(
        np.linspace(0, len(xg) - 1, 5),
        np.round(np.linspace(xg[0], xg[-1], 5), 3),
    )
    plt.xlabel("log-moneyness x grid (approx)")
    plt.ylabel("Expiry step")
    plt.title(title)
    plt.show()


def plot_first_strike_monotonicity_violation(
    surface: Any,
    report: Any,
    *,
    forward,
    df,
    bs_model: Any | None = None,
    figsize=(9, 4),
):
    """Find the first strike-monotonicity violation and plot call prices highlighting bad intervals.

    If there are no violations, prints a short message and returns None.
    """
    found = first_failing_smile(report)
    if found is None:
        print("No strike monotonicity violations found.")
        return None

    T_fail, rep_fail = found
    bad_i = np.asarray(rep_fail.bad_indices, dtype=int)
    return plot_call_monotonicity_diagnostics(
        surface,
        T=T_fail,
        forward=forward,
        df=df,
        bad_indices=bad_i,
        bs_model=bs_model,
        figsize=figsize,
    )


def plot_calendar_heatmap_from_report(
    surface: Any,
    report: Any,
    *,
    title: str = "Calendar check — negative cells indicate violations",
    figsize=(10, 4),
):
    """If calendar check failed, plot Δw heatmap. Otherwise prints status and returns None."""
    maybe = calendar_dW_from_report(surface, report)
    if maybe is None:
        cal = getattr(report, "calendar_total_variance", None)
        if cal is None:
            print("Calendar check not available.")
        else:
            print("Calendar performed:", bool(getattr(cal, "performed", False)))
            print("Calendar OK:", bool(getattr(cal, "ok", True)))
            print("Calendar message:", str(getattr(cal, "message", "")))
        return None
    xg, _dW = maybe
    return plot_calendar_heatmap(surface, x_grid=xg, title=title, figsize=figsize)


# -----------------------------------------------------------------------------
# SVI-specific diagnostics plots (duck-typed via compute.svi_* helpers)
# -----------------------------------------------------------------------------


def plot_svi_fit_slice(
    surface: Any,
    *,
    T: float,
    forward,
    quotes_df=None,
    kind: str = "w",
    n_curve: int = 401,
    title: str | None = None,
    figsize=(9, 4),
):
    """Plot SVI smile fit at expiry ``T``.

    Parameters
    ----------
    surface:
        A surface whose smile at ``T`` is typically an ``SVISmile``.
    T:
        Expiry.
    forward:
        Forward curve callable ``forward(T)``.
    quotes_df:
        Optional quotes DataFrame (single-expiry or multi-expiry). If provided,
        market points are overlaid.
    kind:
        "w" plots total variance vs log-moneyness. "iv" plots implied vol.
    n_curve:
        Number of points for the fitted curve.
    """

    plt = _get_plt()
    kind = str(kind).lower().strip()
    if kind not in {"w", "iv"}:
        raise ValueError("kind must be 'w' or 'iv'")

    s = get_smile_at_T(surface, float(T))
    y_curve = np.linspace(float(s.y_min), float(s.y_max), int(n_curve), dtype=float)

    if kind == "w":
        fit_curve = np.asarray(s.w_at(y_curve), dtype=float)
        ylab = "Total variance w(y)"
    else:
        fit_curve = np.asarray(s.iv_at(y_curve), dtype=float)
        ylab = "Implied vol"

    plt.figure(figsize=figsize)
    plt.plot(y_curve, fit_curve, linewidth=2.0, label="fit")

    if quotes_df is not None:
        df_res = svi_residuals_df(surface, quotes_df, T=float(T), forward=forward)
        if not df_res.empty:
            if kind == "w":
                plt.scatter(df_res["y"], df_res["w_mkt"], s=30, label="market")
            else:
                plt.scatter(df_res["y"], df_res["iv_mkt"], s=30, label="market")

    plt.xlabel("log-moneyness y = ln(K/F)")
    plt.ylabel(ylab)
    plt.title(title or f"SVI fit slice at T={float(T):g}y ({kind})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_svi_residuals(
    surface: Any,
    *,
    T: float,
    quotes_df,
    forward,
    kind: str = "w",
    title: str | None = None,
    figsize=(9, 4),
):
    """Plot SVI fit residuals at expiry ``T`` (fit - market)."""
    plt = _get_plt()
    kind = str(kind).lower().strip()
    if kind not in {"w", "iv"}:
        raise ValueError("kind must be 'w' or 'iv'")

    df_res = svi_residuals_df(surface, quotes_df, T=float(T), forward=forward)
    if df_res.empty:
        print(f"No quote points found for residuals at T={float(T):g}y")
        return None

    if kind == "w":
        resid = np.asarray(df_res["resid_w"], dtype=float)
        ylab = "Residual (w_fit - w_mkt)"
    else:
        resid = np.asarray(df_res["resid_iv"], dtype=float)
        ylab = "Residual (iv_fit - iv_mkt)"

    y = np.asarray(df_res["y"], dtype=float)

    plt.figure(figsize=figsize)
    plt.axhline(0.0, linewidth=1.0)
    plt.plot(y, resid, marker="o", linewidth=1.5)
    plt.xlabel("log-moneyness y = ln(K/F)")
    plt.ylabel(ylab)
    plt.title(title or f"SVI residuals at T={float(T):g}y ({kind})")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_butterfly_proxy(
    surface_or_smile: Any,
    *,
    T: float | None = None,
    n: int = 801,
    w_floor: float = 0.0,
    g_floor: float = 0.0,
    title: str | None = None,
    figsize=(9, 4),
):
    """Plot Gatheral-Jacquier butterfly proxy g(y) for an SVI smile.

    Accepts either:
      * a surface + ``T`` (will locate the smile at that expiry), or
      * a smile object directly (``T`` ignored).

    Negative values (below ``g_floor``) indicate likely butterfly arbitrage.
    """
    plt = _get_plt()

    if T is not None:
        smile = get_smile_at_T(surface_or_smile, float(T))
        T_plot = float(T)
    else:
        smile = surface_or_smile
        T_plot = float(getattr(smile, "T", np.nan))

    params = getattr(smile, "params", None)
    if params is None:
        raise TypeError("plot_butterfly_proxy requires an SVI-like smile with .params")

    # Lazy import keeps plotting optional and avoids importing SVI unless needed.
    from option_pricing.vol.svi import gatheral_g_vec  # type: ignore

    y = np.linspace(float(smile.y_min), float(smile.y_max), int(n), dtype=float)
    g = np.asarray(
        gatheral_g_vec(y.astype(np.float64, copy=False), params, w_floor=w_floor)
    )

    plt.figure(figsize=figsize)
    plt.axhline(
        float(g_floor), linewidth=1.0, linestyle="--", label=f"g_floor={g_floor:g}"
    )
    plt.plot(y, g, linewidth=2.0, label="g(y)")

    if np.any(np.isfinite(g)):
        i = int(np.nanargmin(g))
        plt.scatter([y[i]], [g[i]], s=50, label=f"min g={float(g[i]):.3g}")

    plt.xlabel("log-moneyness y = ln(K/F)")
    plt.ylabel("Butterfly proxy g(y)")
    plt.title(title or f"Butterfly proxy g(y) at T={T_plot:g}y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_svi_rmse_by_expiry(
    surface: Any,
    *,
    title: str = "SVI fit quality — RMSE in total variance",
    figsize=(9, 4),
):
    """Plot per-expiry RMSE (total variance space) if SVI diagnostics are present."""
    plt = _get_plt()
    df = svi_fit_table(surface)
    if df.empty or ("rmse_w" not in df.columns) or (df["has_diagnostics"].sum() == 0):
        print("No SVI diagnostics available on this surface.")
        return None

    d = df.loc[df["has_diagnostics"].astype(bool)].copy()
    plt.figure(figsize=figsize)
    plt.plot(d["T"].astype(float), d["rmse_w"].astype(float), marker="o", linewidth=1.5)
    plt.xlabel("Expiry T")
    plt.ylabel("RMSE (w)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_surface_domain_report(
    surface: Any,
    *,
    quotes_df=None,
    forward=None,
    title: str = "Domain of trust — model vs data log-moneyness coverage",
    figsize=(9, 4),
):
    """Visualize per-expiry model domain and (optional) quote coverage."""
    plt = _get_plt()
    df = surface_domain_report(surface, quotes_df=quotes_df, forward=forward)
    if df.empty:
        print("Empty domain report.")
        return None

    Tvals = df["T"].astype(float).to_numpy()
    y0m = df["y_model_min"].astype(float).to_numpy()
    y1m = df["y_model_max"].astype(float).to_numpy()
    y0d = df["y_data_min"].astype(float).to_numpy()
    y1d = df["y_data_max"].astype(float).to_numpy()

    plt.figure(figsize=figsize)
    for i in range(len(Tvals)):
        plt.plot(
            [y0m[i], y1m[i]],
            [Tvals[i], Tvals[i]],
            linewidth=4,
            alpha=0.6,
            label="model" if i == 0 else None,
        )
        if np.isfinite(y0d[i]) and np.isfinite(y1d[i]):
            plt.plot(
                [y0d[i], y1d[i]],
                [Tvals[i], Tvals[i]],
                linewidth=2,
                label="data" if i == 0 else None,
            )

    plt.xlabel("log-moneyness y")
    plt.ylabel("Expiry T")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


# ----------------------------
# Local-vol plotting helpers
# ----------------------------


def _imshow_grid(
    data: np.ndarray, *, x: np.ndarray, y: np.ndarray, title: str, cbar: str, figsize
):
    plt = _get_plt()
    plt.figure(figsize=figsize)
    plt.imshow(data, aspect="auto", origin="lower")
    plt.colorbar(label=cbar)

    plt.yticks(np.arange(len(y)), [f"{float(t):g}" for t in y])
    xt = np.linspace(0, len(x) - 1, 5)
    plt.xticks(xt, np.round(np.linspace(float(x[0]), float(x[-1]), 5), 3))
    plt.xlabel("log-moneyness y")
    plt.ylabel("Expiry T")
    plt.title(title)
    plt.show()


def plot_localvol_heatmap(
    rep: LocalVolGridReport,
    *,
    kind: str = "sigma",
    mask_invalid: bool = True,
    title: str | None = None,
    figsize=(10, 4),
):
    """Heatmap of local-vol diagnostics over (T, y).

    kind:
      - "sigma": local volatility
      - "local_var": local variance
      - "denom": Gatheral denominator
    """
    kind = str(kind)
    if kind == "sigma":
        data = np.asarray(rep.sigma, dtype=float)
        cbar = r"$\sigma_{loc}$"
        default_title = "Local volatility heatmap"
    elif kind in {"var", "local_var"}:
        data = np.asarray(rep.local_var, dtype=float)
        cbar = r"$\sigma_{loc}^2$"
        default_title = "Local variance heatmap"
    elif kind == "denom":
        data = np.asarray(rep.denom, dtype=float)
        cbar = "denom"
        default_title = "Gatheral denominator heatmap"
    else:
        raise ValueError("kind must be 'sigma', 'local_var', or 'denom'")

    if mask_invalid:
        data = np.where(rep.invalid, np.nan, data)

    _imshow_grid(
        data,
        x=rep.y,
        y=rep.expiries,
        title=title or default_title,
        cbar=cbar,
        figsize=figsize,
    )


def plot_localvol_invalid_mask(
    rep: LocalVolGridReport,
    *,
    reason_bit: int | None = None,
    title: str | None = None,
    figsize=(10, 4),
):
    """Plot invalid mask; optionally filter to a single reason bit."""
    if reason_bit is None:
        mask = np.asarray(rep.invalid, dtype=float)
        t = title or "Local-vol invalid mask (any reason)"
    else:
        rb = np.uint32(int(reason_bit))
        mask = np.asarray((rep.reason & rb) != 0, dtype=float)
        t = title or f"Local-vol invalid mask (reason bit={int(reason_bit)})"

    _imshow_grid(
        mask, x=rep.y, y=rep.expiries, title=t, cbar="invalid", figsize=figsize
    )


def plot_localvol_histogram(
    rep: LocalVolGridReport,
    *,
    bins: int = 50,
    title: str = "Local vol histogram (valid points)",
    figsize=(8, 4),
):
    """Histogram of valid local-vol values."""
    plt = _get_plt()
    sig = np.asarray(rep.sigma, dtype=float)
    mask = np.asarray(rep.invalid, dtype=bool) | (~np.isfinite(sig))
    vals = sig[~mask]

    plt.figure(figsize=figsize)
    plt.hist(vals, bins=int(bins))
    plt.xlabel(r"$\sigma_{loc}$")
    plt.ylabel("count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_localvol_method_diff(
    rep: LocalVolCompareReport,
    *,
    kind: str = "sigma",
    mask_invalid: bool = True,
    title: str | None = None,
    figsize=(10, 4),
):
    """Heatmap of (Dupire - Gatheral) on the shared (T, K) grid."""

    kind = str(kind)
    if kind == "sigma":
        data = np.asarray(rep.diff_sigma, dtype=float)
        cbar = r"$\sigma_{Dupire} - \sigma_{Gatheral}$"
        default_title = "Local vol difference (Dupire - Gatheral)"
    elif kind in {"var", "local_var"}:
        data = np.asarray(rep.diff_local_var, dtype=float)
        cbar = r"$\sigma^2_{Dupire} - \sigma^2_{Gatheral}$"
        default_title = "Local variance difference (Dupire - Gatheral)"
    else:
        raise ValueError("kind must be 'sigma' or 'local_var'")

    if mask_invalid:
        data = np.where(np.asarray(rep.invalid_union, dtype=bool), np.nan, data)

    plt = _get_plt()
    plt.figure(figsize=figsize)
    plt.imshow(data, aspect="auto", origin="lower")
    plt.colorbar(label=cbar)

    # Y axis: maturities
    plt.yticks(np.arange(len(rep.expiries)), [f"{float(t):g}" for t in rep.expiries])
    plt.ylabel("Expiry T")

    # X axis: strikes
    x = np.asarray(rep.strikes, dtype=float)
    xt = np.linspace(0, len(x) - 1, 5)
    plt.xticks(xt, np.round(np.linspace(float(x[0]), float(x[-1]), 5), 3))
    plt.xlabel("Strike K")

    plt.title(title or default_title)
    plt.show()
