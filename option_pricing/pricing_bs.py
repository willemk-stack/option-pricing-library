import numpy as np
from scipy.stats import norm

def _d_plus_minus(t, x, K, r, sigma, T):
    """Return d_plus, d_minus for given parameters."""
    tau = T - t
    sqrt_tau = np.sqrt(tau)
    d_plus = (np.log(x / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d_minus = d_plus - sigma * sqrt_tau
    return d_plus, d_minus, tau

def c(t, x, K, r, sigma, T):
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
    d_plus, d_minus, tau = _d_plus_minus(t, x, K, r, sigma, T)
    N_d_plus = norm.cdf(d_plus)
    N_d_minus = norm.cdf(d_minus)
    return x * N_d_plus - K * np.exp(-r * tau) * N_d_minus

def p(t, x, K, r, sigma, T):
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
    d_plus, d_minus, tau = _d_plus_minus(t, x, K, r, sigma, T)
    N_minus_d_plus = norm.cdf(-d_plus)
    N_minus_d_minus = norm.cdf(-d_minus)
    return K * np.exp(-r * tau) * N_minus_d_minus - x * N_minus_d_plus
