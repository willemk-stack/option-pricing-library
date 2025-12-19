from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import numpy as np
from numpy.typing import NDArray

from ..models.stochastic_processes import sim_gbm_terminal
from ..types import PricingInputs
from ..vanilla import call_payoff, make_vanilla_payoff, put_payoff


def _apply_control_variate(X: np.ndarray, Y: np.ndarray, EY: float) -> np.ndarray:
    # Guard against degenerate controls
    var_y = float(np.var(Y, ddof=1)) if Y.size > 1 else 0.0
    if var_y <= 0.0:
        return X

    cov = float(np.cov(X, Y, ddof=1)[0, 1])
    b = cov / var_y
    return X - b * (Y - float(EY))


@dataclass(frozen=True, slots=True)
class ControlVariate:
    """
    Control variate specification for variance reduction.

    A control variate is a random variable Y that is correlated with the target
    payoff X and has a known (or analytically computable) expectation E[Y]. The
    Monte Carlo estimator can be adjusted using Y to reduce variance.

    Attributes
    ----------
    values
        Function mapping terminal prices ``ST`` (shape ``(n_paths,)``) to control
        variate samples ``Y(ST)`` (same shape).
    mean
        The known expectation of the control variate under the pricing measure,
        i.e. ``E[Y]``.
    """

    values: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    mean: float


@dataclass(frozen=True, slots=True)
class McGBMModel:
    """
    Monte Carlo pricer for European payoffs under a GBM (Black-Scholes-Merton) model.

    The underlying follows geometric Brownian motion under the risk-neutral measure:

        dS_t = (r - q) S_t dt + sigma S_t dW_t

    where ``r`` is the continuously-compounded risk-free rate, ``q`` is the
    continuous dividend yield (or convenience yield), and ``sigma`` is volatility.

    Parameters
    ----------
    S0
        Spot price at time 0. Must be positive.
    r
        Continuously-compounded risk-free rate.
    q
        Continuous dividend yield.
    sigma
        Volatility (annualized). Must be positive.
    tau
        Time to maturity in years. Must be positive.
    n_paths
        Number of Monte Carlo paths (terminal samples). Must be positive.
        If ``antithetic=True``, must be even.
    antithetic
        If ``True``, use antithetic variates by pairing ``Z`` and ``-Z``.
    rng
        NumPy random number generator used for sampling.

    Notes
    -----
    - This model simulates *terminal* prices only (suitable for European payoffs).
    - Pricing returns both the discounted estimate and an estimated standard error.
    """

    S0: float
    r: float
    q: float
    sigma: float
    tau: float
    n_paths: int
    antithetic: bool = False
    rng: np.random.Generator = field(
        default_factory=np.random.default_rng,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.S0 <= 0.0:
            raise ValueError("S0 must be positive")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if self.antithetic and (self.n_paths % 2 != 0):
            raise ValueError(
                "antithetic=True requires an even n_paths (paired samples)."
            )

    def simulate_terminal(self) -> np.ndarray:
        """
        Simulate terminal underlying prices S_T under risk-neutral GBM.

        Returns
        -------
        np.ndarray
            Array of terminal prices with shape ``(n_paths,)`` and dtype float64.

        Notes
        -----
        - Uses drift ``mu = r - q`` under the risk-neutral measure.
        - If ``antithetic=False``, draws ``n_paths`` independent standard normals.
        - If ``antithetic=True``, draws ``n_paths/2`` normals ``Z`` and returns
          paired samples generated from ``Z`` and ``-Z``.
        """
        mu = self.r - self.q  # risk-neutral drift

        if not self.antithetic:
            return sim_gbm_terminal(
                n_paths=self.n_paths,
                T=self.tau,
                mu=mu,
                sigma=self.sigma,
                S0=self.S0,
                rng=self.rng,
            )

        # Antithetic: use Z and -Z pairs
        n_pairs = self.n_paths // 2
        Z = self.rng.standard_normal(n_pairs)

        drift = (mu - 0.5 * self.sigma**2) * self.tau
        vol = self.sigma * np.sqrt(self.tau)

        ST_pos = self.S0 * np.exp(drift + vol * Z)
        ST_neg = self.S0 * np.exp(drift - vol * Z)

        out = np.concatenate([ST_pos, ST_neg])
        return np.asarray(out, dtype=np.float64)

    def price_european(
        self,
        payoff: Callable[[np.ndarray], np.ndarray],
        *,
        control: ControlVariate | None = None,
    ) -> tuple[float, float]:
        """
        Price a European payoff via Monte Carlo, with optional variance reduction.

        Parameters
        ----------
        payoff
            Vectorized payoff function mapping terminal prices ``ST`` (shape
            ``(n_paths,)``) to payoff samples (same shape).
        control
            Optional control variate definition. If provided, the payoff samples
            are adjusted using the control variate samples and the known mean
            (via ``_apply_control_variate``).

        Returns
        -------
        (price, stderr) : tuple[float, float]
            ``price`` is the discounted Monte Carlo estimate of E[payoff(S_T)].
            ``stderr`` is the estimated standard error of the *discounted*
            estimator.

        Notes
        -----
        - Discounting uses ``exp(-r * tau)``.
        - With ``antithetic=False``, the standard error scales as ``1/sqrt(n_paths)``.
        - With ``antithetic=True``, estimates are formed from pair-averaged samples,
          and the standard error uses ``n_pairs = n_paths/2`` effective observations.
        - Sample standard deviation uses ``ddof=1``; if the effective sample size is
          1, the returned standard error is 0.0.
        """
        ST = self.simulate_terminal()
        payoff_vals = payoff(ST)

        disc = float(np.exp(-self.r * self.tau))

        # ---------- plain MC ----------
        if not self.antithetic:
            X_eff = payoff_vals
            if control is not None:
                Y_vals = control.values(ST)
                X_eff = _apply_control_variate(X_eff, Y_vals, control.mean)

            mean = float(X_eff.mean())
            std = float(X_eff.std(ddof=1)) if self.n_paths > 1 else 0.0

            price = disc * mean
            std_err = disc * std / float(np.sqrt(self.n_paths))
            return price, std_err

        # ---------- antithetic MC ----------
        n_pairs = self.n_paths // 2
        Xp = 0.5 * (payoff_vals[:n_pairs] + payoff_vals[n_pairs:])

        if control is not None:
            Y_vals = control.values(ST)
            Yp = 0.5 * (Y_vals[:n_pairs] + Y_vals[n_pairs:])
            Xp = _apply_control_variate(Xp, Yp, control.mean)

        mean = float(Xp.mean())
        std = float(Xp.std(ddof=1)) if n_pairs > 1 else 0.0

        price = disc * mean
        std_err = disc * std / float(np.sqrt(n_pairs))
        return price, std_err


def _make_rng(seed: int | None, rng: np.random.Generator | None) -> np.random.Generator:
    if rng is not None:
        return rng
    if seed is not None:
        return np.random.default_rng(int(seed))
    return np.random.default_rng()


def mc_price(
    p: PricingInputs,
    *,
    n_paths: int,
    antithetic: bool = False,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Price a European vanilla option using Monte Carlo simulation under a GBM model.

    This function builds a :class:`McGBMModel` from the market/model inputs in ``p``,
    simulates terminal prices, evaluates the corresponding vanilla payoff, and
    returns the Monte Carlo estimator along with its sampling uncertainty.

    Parameters
    ----------
    p : PricingInputs
        Pricing inputs containing (at least) the spot ``S``, risk-free rate ``r``,
        dividend yield ``q``, volatility ``sigma``, time to maturity ``tau``,
        strike ``K``, and option specification ``spec.kind`` (e.g. call/put).
    n_paths : int
        Number of Monte Carlo paths to simulate.
    antithetic : bool, default False
        If ``True``, use antithetic variates to reduce variance.
    seed : int | None, default None
        Seed used to initialize a new RNG when ``rng`` is not provided.
        Ignored if ``rng`` is provided.
    rng : np.random.Generator | None, default None
        Optional NumPy random number generator to use. If provided, it takes
        precedence over ``seed``.

    Returns
    -------
    (price, stderr) : tuple[float, float]
        ``price`` is the Monte Carlo estimate of the discounted option value.
        ``stderr`` is the estimated standard error of the Monte Carlo estimator
        (i.e., the standard deviation of the estimate).

    Notes
    -----
    - The option is European and depends only on the terminal underlying price.
    - The simulation is performed under the risk-neutral measure with continuous
      dividend yield ``q``.
    - Reproducibility: pass ``seed`` or an explicit ``rng``. If both are given,
      ``rng`` is used.

    See Also
    --------
    McGBMModel.price_european : Prices a European payoff via Monte Carlo.
    make_vanilla_payoff : Constructs the call/put payoff function.

    Examples
    --------
    >>> price, err = mc_price(p, n_paths=200_000, antithetic=True, seed=123)
    >>> price, err
    (10.42, 0.03)
    """
    model = McGBMModel(
        S0=p.S,
        r=p.r,
        q=p.q,
        sigma=p.sigma,
        tau=p.tau,
        n_paths=int(n_paths),
        antithetic=bool(antithetic),
        rng=_make_rng(seed, rng),
    )

    payoff = make_vanilla_payoff(p.spec.kind, K=p.K)
    return model.price_european(payoff)


def mc_price_call(
    p: PricingInputs,
    *,
    n_paths: int,
    antithetic: bool = False,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    model = McGBMModel(
        S0=p.S,
        r=p.r,
        q=p.q,
        sigma=p.sigma,
        tau=p.tau,
        n_paths=int(n_paths),
        antithetic=bool(antithetic),
        rng=_make_rng(seed, rng),
    )
    payoff = partial(call_payoff, K=p.K)
    return model.price_european(payoff)


def mc_price_put(
    p: PricingInputs,
    *,
    n_paths: int,
    antithetic: bool = False,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    model = McGBMModel(
        S0=p.S,
        r=p.r,
        q=p.q,
        sigma=p.sigma,
        tau=p.tau,
        n_paths=int(n_paths),
        antithetic=bool(antithetic),
        rng=_make_rng(seed, rng),
    )
    payoff = partial(put_payoff, K=p.K)
    return model.price_european(payoff)
