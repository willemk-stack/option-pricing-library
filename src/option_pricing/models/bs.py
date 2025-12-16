import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from ..types import PricingInputs


def _validate_inputs(t, x, K, r, sigma, T):
    assert 0 <= t < T, "Must have 0 <= t < T"
    assert K > 0 and sigma > 0, "K and sigma must be positive"

    x_arr = np.asarray(x, dtype=float)
    assert np.all(x_arr > 0), "x must be positive"

    return x_arr


def _d_plus_minus(t, x, K, r, sigma, T):
    """Return d_plus (d1), d_minus (d2), tau for given parameters."""
    tau = T - t
    sqrt_tau = np.sqrt(tau)
    d_plus = (np.log(x / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d_minus = d_plus - sigma * sqrt_tau
    return d_plus, d_minus, tau


def bs_call(t, x, K, r, sigma, T):
    """Black-Scholes price of a European call (no dividends)."""
    x = _validate_inputs(t, x, K, r, sigma, T)
    d_plus, d_minus, tau = _d_plus_minus(t, x, K, r, sigma, T)
    return x * norm.cdf(d_plus) - K * np.exp(-r * tau) * norm.cdf(d_minus)


def bs_call_from_inputs(p: PricingInputs) -> float:
    return bs_call(t=p.t, x=p.S, K=p.K, r=p.r, sigma=p.sigma, T=p.T)


def bs_put(t, x, K, r, sigma, T):
    """Black-Scholes price of a European put (no dividends)."""
    x = _validate_inputs(t, x, K, r, sigma, T)
    d_plus, d_minus, tau = _d_plus_minus(t, x, K, r, sigma, T)
    return K * np.exp(-r * tau) * norm.cdf(-d_minus) - x * norm.cdf(-d_plus)


def bs_put_from_inputs(p: PricingInputs) -> float:
    return bs_put(t=p.t, x=p.S, K=p.K, r=p.r, sigma=p.sigma, T=p.T)


# ----------------------------
# Analytic Greeks (NEW)
# ----------------------------


def bs_call_greeks_analytic(t, x, K, r, sigma, T):
    """
    Analytic Greeks for the Black-Scholes European CALL (no dividends).

    Returns a dict with price, delta, gamma, vega, theta
    where theta = ∂C/∂t (calendar time, holding T fixed), per year.
    """
    x = _validate_inputs(t, x, K, r, sigma, T)
    d1, d2, tau = _d_plus_minus(t, x, K, r, sigma, T)
    sqrt_tau = np.sqrt(tau)

    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    phi_d1 = norm.pdf(d1)
    disc = np.exp(-r * tau)

    price = x * Nd1 - K * disc * Nd2
    delta = Nd1
    gamma = phi_d1 / (x * sigma * sqrt_tau)
    vega = x * phi_d1 * sqrt_tau
    theta = -(x * phi_d1 * sigma) / (2.0 * sqrt_tau) - r * K * disc * Nd2

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
    }


def bs_put_greeks_analytic(t, x, K, r, sigma, T):
    """
    Analytic Greeks for the Black-Scholes European PUT (no dividends).

    Returns a dict with price, delta, gamma, vega, theta
    where theta = ∂P/∂t (calendar time, holding T fixed), per year.
    """
    x = _validate_inputs(t, x, K, r, sigma, T)
    d1, d2, tau = _d_plus_minus(t, x, K, r, sigma, T)
    sqrt_tau = np.sqrt(tau)

    Nmd1 = norm.cdf(-d1)
    Nmd2 = norm.cdf(-d2)
    phi_d1 = norm.pdf(d1)
    disc = np.exp(-r * tau)

    price = K * disc * Nmd2 - x * Nmd1
    delta = norm.cdf(d1) - 1.0
    gamma = phi_d1 / (x * sigma * sqrt_tau)
    vega = x * phi_d1 * sqrt_tau
    theta = -(x * phi_d1 * sigma) / (2.0 * sqrt_tau) + r * K * disc * Nmd2

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
    }


def bs_call_greeks_analytic_from_inputs(p: PricingInputs):
    return bs_call_greeks_analytic(t=p.t, x=p.S, K=p.K, r=p.r, sigma=p.sigma, T=p.T)


def bs_put_greeks_analytic_from_inputs(p: PricingInputs):
    return bs_put_greeks_analytic(t=p.t, x=p.S, K=p.K, r=p.r, sigma=p.sigma, T=p.T)


# ----------------------------
# Finite-difference Greeks (your existing code)
# ----------------------------


def finite_diff_greeks(t, x, K, r, sigma, T, h_x=None, h_sigma=None, h_t=None):
    """
    Finite-difference Greeks for the Black-Scholes call with signature
        bs_call(t, x, K, r, sigma, T).

    Returns a dict with price, delta, gamma, vega, theta,
    where theta ≈ ∂C/∂t (calendar time, holding T fixed).
    """
    _ = _validate_inputs(t, x, K, r, sigma, T)

    h_x = h_x or (0.01 * x)  # 1% of spot
    h_sigma = h_sigma or (0.01 * sigma)  # 1% of vol
    h_t = h_t or (1.0 / 365.0)  # 1 day in years

    h_sigma = min(h_sigma, 0.5 * sigma)

    V = bs_call(t, x, K, r, sigma, T)

    V_up_x = bs_call(t, x + h_x, K, r, sigma, T)
    V_down_x = bs_call(t, x - h_x, K, r, sigma, T)

    delta = (V_up_x - V_down_x) / (2.0 * h_x)
    gamma = (V_up_x - 2.0 * V + V_down_x) / (h_x**2)

    V_up_sigma = bs_call(t, x, K, r, sigma + h_sigma, T)
    V_down_sigma = bs_call(t, x, K, r, sigma - h_sigma, T)

    vega = (V_up_sigma - V_down_sigma) / (2.0 * h_sigma)

    theta = None
    h_t = min(h_t, 0.5 * (T - t)) if T > t else h_t

    if t - h_t >= 0.0 and t + h_t < T:
        V_up_t = bs_call(t + h_t, x, K, r, sigma, T)
        V_down_t = bs_call(t - h_t, x, K, r, sigma, T)
        theta = (V_up_t - V_down_t) / (2.0 * h_t)
    elif t + h_t < T:
        V_up_t = bs_call(t + h_t, x, K, r, sigma, T)
        theta = (V_up_t - V) / h_t
    elif t - h_t >= 0.0:
        V_down_t = bs_call(t - h_t, x, K, r, sigma, T)
        theta = (V - V_down_t) / h_t
    else:
        raise ValueError("Cannot compute Theta: time steps violate 0 <= t < T.")

    return {"price": V, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


# ----------------------------
# Plot sweep (optional: choose analytic vs FD)
# ----------------------------


def sweep_x(
    t=0.0,
    K=100,
    r=0.01,
    sigma=0.2,
    T=1.0,
    x_min=None,
    x_max=None,
    n=100,
    method="analytic",
):
    """
    Sweep underlying price x and plot price + Greeks.

    method: "analytic" (default) or "fd"
    """
    if x_min is None:
        x_min = 0.5 * K
    if x_max is None:
        x_max = 1.5 * K

    x_grid = np.linspace(x_min, x_max, n)

    prices, deltas, gammas, vegas, thetas = [], [], [], [], []

    for x in x_grid:
        if method == "analytic":
            g = bs_call_greeks_analytic(t, x, K, r, sigma, T)
        elif method == "fd":
            g = finite_diff_greeks(t, x, K, r, sigma, T)
        else:
            raise ValueError("method must be 'analytic' or 'fd'")

        prices.append(g["price"])
        deltas.append(g["delta"])
        gammas.append(g["gamma"])
        vegas.append(g["vega"])
        thetas.append(g["theta"])

    prices = np.array(prices)
    deltas = np.array(deltas)
    gammas = np.array(gammas)
    vegas = np.array(vegas)
    thetas = np.array(thetas)

    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(8, 12))

    axs[0].plot(x_grid, prices)
    axs[0].set_ylabel("Price")
    axs[1].plot(x_grid, deltas)
    axs[1].set_ylabel("Delta")
    axs[2].plot(x_grid, gammas)
    axs[2].set_ylabel("Gamma")
    axs[3].plot(x_grid, vegas)
    axs[3].set_ylabel("Vega")
    axs[4].plot(x_grid, thetas)
    axs[4].set_ylabel("Theta")
    axs[4].set_xlabel("Underlying x")

    plt.tight_layout()
    plt.show()
