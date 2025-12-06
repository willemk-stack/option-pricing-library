import numpy as np
from scipy.stats import norm

def _d_plus_minus(t, x, K, r, sigma, T):
    """Return d_plus, d_minus for given parameters."""
    tau = T - t
    sqrt_tau = np.sqrt(tau)
    d_plus = (np.log(x / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d_minus = d_plus - sigma * sqrt_tau
    return d_plus, d_minus, tau

def bs_call(t, x, K, r, sigma, T):
    """Price of a European call option under the Black-Scholes model.

    This function computes the time-t price of a European call option with
    maturity T and strike K, assuming the underlying follows a geometric
    Brownian motion under the risk-neutral measure.

    The pricing formula is
        C(t, x) = x * N(d_+) - K * exp(-r * (T - t)) * N(d_-),

    where
        d_+ = [ln(x / K) + (r + 0.5 * sigma^2) * (T - t)] / (sigma * sqrt(T - t)),
        d_- = d_+ - sigma * sqrt(T - t),
    and N(·) is the standard normal cumulative distribution function.

    Args:
        t (float): Valuation time, satisfying 0 <= t < T.
        x (float or np.ndarray): Underlying asset price at time t (must be > 0).
        K (float): Option strike price (must be > 0).
        r (float): Continuously compounded risk-free interest rate.
        sigma (float): Volatility of the underlying asset (must be > 0).
        T (float): Option maturity time (must satisfy T > t).

    Returns:
        float or np.ndarray: The Black-Scholes value of the European call option
        at time t. If `x` is an array, the result has the same shape as `x`.

    Raises:
        AssertionError: If input parameters violate the assumptions
            (e.g., T <= t, x <= 0, K <= 0, or sigma <= 0), if such checks are added.

    Notes:
        - This implementation assumes no dividends.
        - The function is vectorized with respect to `x` when used with NumPy arrays.
    """
    assert 0 <= t < T, "Must have 0 <= t < T"
    assert x > 0 and K > 0 and sigma > 0, "x, K, sigma must be positive"

    d_plus, d_minus, tau = _d_plus_minus(t, x, K, r, sigma, T)
    N_d_plus = norm.cdf(d_plus)
    N_d_minus = norm.cdf(d_minus)
    return x * N_d_plus - K * np.exp(-r * tau) * N_d_minus

def bs_put(t, x, K, r, sigma, T):
    """Price of a European put option under the Black-Scholes model.

    This function computes the time-t price of a European put option with
    maturity T and strike K, assuming the underlying follows a geometric
    Brownian motion under the risk-neutral measure.

    The pricing formula is
        P(t, x) = K * exp(-r * (T - t)) * N(-d_-) - x * N(-d_+),

    where
        d_+ = [ln(x / K) + (r + 0.5 * sigma^2) * (T - t)] / (sigma * sqrt(T - t)),
        d_- = d_+ - sigma * sqrt(T - t),
    and N(·) is the standard normal cumulative distribution function.

    Args:
        t (float): Valuation time, satisfying 0 <= t < T.
        x (float or np.ndarray): Underlying asset price at time t (must be > 0).
        K (float): Option strike price (must be > 0).
        r (float): Continuously compounded risk-free interest rate.
        sigma (float): Volatility of the underlying asset (must be > 0).
        T (float): Option maturity time (must satisfy T > t).

    Returns:
        float or np.ndarray: The Black-Scholes value of the European put option
        at time t. If `x` is an array, the result has the same shape as `x`.

    Raises:
        AssertionError: If input parameters violate the assumptions
            (e.g., T <= t, x <= 0, K <= 0, or sigma <= 0), if such checks are added.

    Notes:
        - This implementation assumes no dividends.
        - The function is vectorized with respect to `x` when used with NumPy arrays.
        - Put-call parity holds:
              c(t, x, K, r, sigma, T) - p(t, x, K, r, sigma, T)
              = x - K * exp(-r * (T - t)).
    """
    assert 0 <= t < T, "Must have 0 <= t < T"
    assert x > 0 and K > 0 and sigma > 0, "x, K, sigma must be positive"

    d_plus, d_minus, tau = _d_plus_minus(t, x, K, r, sigma, T)
    N_minus_d_plus = norm.cdf(-d_plus)
    N_minus_d_minus = norm.cdf(-d_minus)
    return K * np.exp(-r * tau) * N_minus_d_minus - x * N_minus_d_plus

def finite_diff_greeks(t, x, K, r, sigma, T,
                       h_x=None, h_sigma=None, h_t=None):
    """
    Finite-difference Greeks for the Black-Scholes call with signature
        bs_call(t, x, K, r, sigma, T).

    Returns a dict with price, delta, gamma, vega, theta,
    where theta ≈ ∂C/∂t (calendar time, holding T fixed).
    """
    # Sensible step sizes if not provided
    h_x     = h_x     or (0.01 * x)        # 1% of spot
    h_sigma = h_sigma or (0.01 * sigma)    # 1% of vol
    h_t     = h_t     or (1.0 / 365.0)     # 1 day in years

    # Make sure we don't violate bs_call assertions
    # For sigma: keep sigma ± h_sigma > 0
    h_sigma = min(h_sigma, 0.5 * sigma)

    # Base price
    V = bs_call(t, x, K, r, sigma, T)

    # ----- Delta & Gamma in x -----
    V_up_x   = bs_call(t, x + h_x, K, r, sigma, T)
    V_down_x = bs_call(t, x - h_x, K, r, sigma, T)

    delta = (V_up_x - V_down_x) / (2.0 * h_x)
    gamma = (V_up_x - 2.0 * V + V_down_x) / (h_x ** 2)

    # ----- Vega in sigma -----
    V_up_sigma   = bs_call(t, x, K, r, sigma + h_sigma, T)
    V_down_sigma = bs_call(t, x, K, r, sigma - h_sigma, T)

    vega = (V_up_sigma - V_down_sigma) / (2.0 * h_sigma)

    # ----- Theta in t (calendar time) -----
    # We need t ± h_t to satisfy 0 <= t ± h_t < T.
    # Use central difference if possible; otherwise fall back to one-sided.
    theta = None

    # Enforce that h_t doesn't push us past maturity
    h_t = min(h_t, 0.5 * (T - t)) if T > t else h_t

    if t - h_t >= 0.0 and t + h_t < T:
        # Central difference: theta = ∂V/∂t
        V_up_t   = bs_call(t + h_t, x, K, r, sigma, T)
        V_down_t = bs_call(t - h_t, x, K, r, sigma, T)
        theta = (V_up_t - V_down_t) / (2.0 * h_t)
    elif t + h_t < T:
        # Forward difference at/near t = 0
        V_up_t = bs_call(t + h_t, x, K, r, sigma, T)
        theta = (V_up_t - V) / h_t
    elif t - h_t >= 0.0:
        # Backward difference near maturity
        V_down_t = bs_call(t - h_t, x, K, r, sigma, T)
        theta = (V - V_down_t) / h_t
    else:
        raise ValueError("Cannot compute Theta: time steps violate 0 <= t < T.")

    return {
        "price": V,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta
    }

import numpy as np
import matplotlib.pyplot as plt

def sweep_x(t=0.0, K=100, r=0.01, sigma=0.2, T=1.0,
            x_min=None, x_max=None, n=100):
    """
    Sweep the underlying price x and plot price + Greeks.

    Parameters
    ----------
    t : float
        Valuation time (0 <= t < T).
    K : float
        Strike.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Maturity.
    x_min, x_max : float, optional
        Range of underlying prices to sweep. If None, use [0.5*K, 1.5*K].
    n : int
        Number of grid points.
    """
    if x_min is None:
        x_min = 0.5 * K
    if x_max is None:
        x_max = 1.5 * K

    x_grid = np.linspace(x_min, x_max, n)

    prices, deltas, gammas, vegas, thetas = [], [], [], [], []

    for x in x_grid:
        g = finite_diff_greeks(t, x, K, r, sigma, T)
        prices.append(g["price"])
        deltas.append(g["delta"])
        gammas.append(g["gamma"])
        vegas.append(g["vega"])
        thetas.append(g["theta"])

    prices = np.array(prices)
    deltas = np.array(deltas)
    gammas = np.array(gammas)
    vegas  = np.array(vegas)
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
