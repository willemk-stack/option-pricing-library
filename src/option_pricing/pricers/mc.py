from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from ..config import MCConfig, RandomConfig
from ..instruments.base import ExerciseStyle, TerminalInstrument
from ..instruments.factory import from_pricing_inputs
from ..instruments.payoffs import call_payoff, make_vanilla_payoff, put_payoff
from ..market.curves import PricingContext, avg_carry_from_forward, avg_rate_from_df
from ..models.stochastic_processes import sim_gbm_terminal
from ..types import MarketData, OptionType, PricingInputs
from ..typing import FloatArray, FloatDType


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

    values: Callable[[FloatArray], FloatArray]
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
        return np.asarray(out, dtype=FloatDType)

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


def _rng_from_random_config(rc: RandomConfig) -> np.random.Generator:
    """Construct a NumPy RNG from a :class:`~option_pricing.config.RandomConfig`.

    Notes
    -----
    - ``pcg64`` uses :func:`numpy.random.default_rng`.
    - ``mt19937`` uses :class:`numpy.random.MT19937`.
    - ``sobol`` is not provided by NumPy; raise to make the limitation explicit.
    """
    seed = int(rc.seed)
    if rc.rng_type == "pcg64":
        return np.random.default_rng(seed)
    if rc.rng_type == "mt19937":
        return np.random.Generator(np.random.MT19937(seed))
    if rc.rng_type == "sobol":
        raise NotImplementedError(
            "Sobol sequences are not available in NumPy. "
            "Either pass an explicit rng=... in MCConfig, or choose rng_type='pcg64'/'mt19937'."
        )
    # Defensive: RandomConfig is type-restricted, but keep runtime robust.
    raise ValueError(f"Unknown rng_type: {rc.rng_type!r}")


def _make_mc_rng(cfg: MCConfig) -> np.random.Generator:
    """Select the RNG for MC from MCConfig."""
    return cfg.rng if cfg.rng is not None else _rng_from_random_config(cfg.random)


def mc_price_from_ctx(
    *,
    ctx: PricingContext,
    kind: OptionType,
    strike: float,
    sigma: float,
    tau: float,
    cfg: MCConfig | None = None,
) -> tuple[float, float]:
    """Monte Carlo GBM pricer using curves-first inputs.

    Notes
    -----
    The internal GBM model still needs single-number (r, q) inputs. For a curves-first
    context we use the *average* rates consistent with ``df(tau)`` and ``fwd(tau)``:

    - ``r_avg = -log(df)/tau``
    - ``(r-q)_avg = log(F/S)/tau``  -> ``q_avg = r_avg - (r-q)_avg``

    This is exact for flat curves and provides a reasonable reduction for deterministic
    term structures when using a terminal-only (European) GBM simulator.
    """
    cfg = MCConfig() if cfg is None else cfg
    tau = float(tau)
    df = ctx.df(tau)
    F = ctx.fwd(tau)
    r = avg_rate_from_df(df, tau)
    b = avg_carry_from_forward(ctx.spot, F, tau)  # (r-q)_avg
    q = r - b

    model = McGBMModel(
        S0=ctx.spot,
        r=r,
        q=q,
        sigma=float(sigma),
        tau=tau,
        n_paths=int(cfg.n_paths),
        antithetic=bool(cfg.antithetic),
        rng=_make_mc_rng(cfg),
    )

    payoff = make_vanilla_payoff(kind, K=float(strike))
    return model.price_european(payoff)


def _to_ctx(market: MarketData | PricingContext) -> PricingContext:
    """Normalize flat MarketData or an already-curves-first PricingContext."""
    if isinstance(market, PricingContext):
        return market
    return market.to_context()


def mc_price_instrument_from_ctx(
    *,
    ctx: PricingContext,
    inst: TerminalInstrument,
    sigma: float,
    cfg: MCConfig | None = None,
) -> tuple[float, float]:
    """Monte Carlo GBM pricer for a terminal-payoff instrument.

    This is the instrument-based equivalent of :func:`mc_price_from_ctx`.

    Notes
    -----
    - ``inst.expiry`` is interpreted as time-to-expiry (tau).
    - Only European exercise is supported by this terminal-only GBM simulator.
    """
    if inst.exercise != ExerciseStyle.EUROPEAN:
        raise ValueError("mc_price_instrument_from_ctx supports European exercise only")

    cfg = MCConfig() if cfg is None else cfg
    tau = float(inst.expiry)
    df = ctx.df(tau)
    F = ctx.fwd(tau)
    r = avg_rate_from_df(df, tau)
    b = avg_carry_from_forward(ctx.spot, F, tau)
    q = r - b

    model = McGBMModel(
        S0=ctx.spot,
        r=r,
        q=q,
        sigma=float(sigma),
        tau=tau,
        n_paths=int(cfg.n_paths),
        antithetic=bool(cfg.antithetic),
        rng=_make_mc_rng(cfg),
    )

    return model.price_european(inst.payoff)


def mc_price_instrument(
    inst: TerminalInstrument,
    *,
    market: MarketData | PricingContext,
    sigma: float,
    cfg: MCConfig | None = None,
) -> tuple[float, float]:
    """Convenience wrapper accepting flat :class:`~option_pricing.types.MarketData`."""
    return mc_price_instrument_from_ctx(
        ctx=_to_ctx(market), inst=inst, sigma=sigma, cfg=cfg
    )


def mc_price(
    p: PricingInputs,
    *,
    cfg: MCConfig | None = None,
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
    cfg : MCConfig or None, optional
        Monte Carlo configuration (path count, variance reduction, RNG policy).
        If ``None``, ``MCConfig()`` is used.

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
    - Reproducibility is controlled via ``cfg.random.seed`` (default 0) or by
      passing an explicit ``cfg.rng``.

    See Also
    --------
    McGBMModel.price_european : Prices a European payoff via Monte Carlo.
    make_vanilla_payoff : Constructs the call/put payoff function.

    Examples
    --------
    >>> from option_pricing.config import MCConfig, RandomConfig
    >>> cfg = MCConfig(n_paths=200_000, antithetic=True, random=RandomConfig(seed=123))
    >>> price, err = mc_price(p, cfg=cfg)
    >>> price, err
    (10.42, 0.03)
    """
    cfg = MCConfig() if cfg is None else cfg
    inst = from_pricing_inputs(p, exercise=ExerciseStyle.EUROPEAN)
    return mc_price_instrument_from_ctx(ctx=p.ctx, inst=inst, sigma=p.sigma, cfg=cfg)


def mc_price_call(
    p: PricingInputs,
    *,
    cfg: MCConfig | None = None,
) -> tuple[float, float]:
    cfg = MCConfig() if cfg is None else cfg
    ctx = p.ctx
    df = ctx.df(p.tau)
    F = ctx.fwd(p.tau)
    r = avg_rate_from_df(df, p.tau)
    b = avg_carry_from_forward(ctx.spot, F, p.tau)
    q = r - b

    model = McGBMModel(
        S0=ctx.spot,
        r=r,
        q=q,
        sigma=p.sigma,
        tau=p.tau,
        n_paths=int(cfg.n_paths),
        antithetic=bool(cfg.antithetic),
        rng=_make_mc_rng(cfg),
    )

    def payoff(ST: FloatArray) -> FloatArray:
        return cast(FloatArray, call_payoff(ST, K=p.K))

    return model.price_european(payoff)


def mc_price_put(
    p: PricingInputs,
    *,
    cfg: MCConfig | None = None,
) -> tuple[float, float]:
    cfg = MCConfig() if cfg is None else cfg
    ctx = p.ctx
    df = ctx.df(p.tau)
    F = ctx.fwd(p.tau)
    r = avg_rate_from_df(df, p.tau)
    b = avg_carry_from_forward(ctx.spot, F, p.tau)
    q = r - b

    model = McGBMModel(
        S0=ctx.spot,
        r=r,
        q=q,
        sigma=p.sigma,
        tau=p.tau,
        n_paths=int(cfg.n_paths),
        antithetic=bool(cfg.antithetic),
        rng=_make_mc_rng(cfg),
    )

    def payoff(ST: FloatArray) -> FloatArray:
        return cast(FloatArray, put_payoff(ST, K=p.K))

    return model.price_european(payoff)
