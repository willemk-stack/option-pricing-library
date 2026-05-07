"""Initial parameter seeds for Heston calibration.

The routines in this module are deliberately heuristic. They are not intended
as parameter estimators; they only provide deterministic, market-aware starting
points for nonlinear least-squares calibration.
"""

from __future__ import annotations

import numpy as np

from ..params import HestonParams
from .bounds import HestonCalibrationBounds
from .heston_types import HestonQuoteSet


def _clip(value: float, lo: float, hi: float) -> float:
    return float(np.clip(float(value), float(lo), float(hi)))


def _weighted_median(
    values: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Return a robust weighted median.

    Falls back to the ordinary median if weights are unavailable or unusable.
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x)

    if not np.any(valid):
        raise ValueError("No finite values available for median.")

    if weights is None:
        return float(np.median(x[valid]))

    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape != x.shape:
        raise ValueError("weights must have the same shape as values.")

    valid &= np.isfinite(w) & (w > 0.0)

    if not np.any(valid):
        return float(np.median(x[np.isfinite(x)]))

    x = x[valid]
    w = w[valid]

    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = w[order]

    cdf = np.cumsum(w_sorted) / np.sum(w_sorted)
    return float(x_sorted[np.searchsorted(cdf, 0.5)])


def _safe_iv_mid(quotes: HestonQuoteSet) -> np.ndarray:
    """Return validated mid implied vols.

    ``quotes.mid`` is the option price mid.

    ``quotes.iv_mid`` is the Black-76 / Black-Scholes implied volatility
    corresponding to that mid price. It is not the price midpoint itself.
    """
    if quotes.iv_mid is None:
        raise ValueError(
            "default_heston_seed requires quotes.iv_mid. "
            "Pass implied vols computed from the mid option prices before calibration."
        )

    iv = np.asarray(quotes.iv_mid, dtype=np.float64).reshape(-1)
    valid = np.isfinite(iv) & (iv > 0.0)

    if not np.any(valid):
        raise ValueError("quotes.iv_mid contains no positive finite implied vols.")

    return iv


def _collapse_duplicate_moneyness(
    log_mny: np.ndarray,
    iv: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Collapse duplicate log-moneyness points.

    Duplicate points can happen if the quote set contains both calls and puts at
    the same strike/expiry. For interpolation, each x-coordinate should have one
    representative IV value.
    """
    x = np.asarray(log_mny, dtype=np.float64).reshape(-1)
    y = np.asarray(iv, dtype=np.float64).reshape(-1)

    if x.shape != y.shape:
        raise ValueError("log_mny and iv must have the same shape.")

    if weights is None:
        w = None
        valid = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape != x.shape:
            raise ValueError("weights must have the same shape as log_mny.")
        valid = (
            np.isfinite(x) & np.isfinite(y) & (y > 0.0) & np.isfinite(w) & (w >= 0.0)
        )

    if not np.any(valid):
        raise ValueError("No valid quote points available.")

    x = x[valid]
    y = y[valid]
    if w is not None:
        w = w[valid]

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if w is not None:
        w = w[order]

    x_unique = np.unique(x)

    y_unique = np.empty_like(x_unique, dtype=np.float64)
    w_unique = None if w is None else np.empty_like(x_unique, dtype=np.float64)

    for i, x0 in enumerate(x_unique):
        mask = x == x0

        if w is None:
            y_unique[i] = float(np.median(y[mask]))
        else:
            y_unique[i] = _weighted_median(y[mask], w[mask])
            assert w_unique is not None
            w_unique[i] = float(np.sum(w[mask]))

    return x_unique, y_unique, w_unique


def _atm_iv_from_slice(
    log_mny: np.ndarray,
    iv: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    fallback_count: int = 3,
    atm_tol: float = 1e-10,
) -> float:
    """Estimate ATM IV for one expiry slice.

    Preferred logic:
    1. If an exact/near-exact ATM quote exists, use its robust median IV.
    2. If the slice brackets ATM, linearly interpolate IV at log-moneyness 0.
    3. If the slice does not bracket ATM, use the closest few quotes as fallback.

    This is deliberately simple. It is only for a calibration seed, not for
    final smile construction.
    """
    if fallback_count <= 0:
        raise ValueError("fallback_count must be positive.")

    x, y, w = _collapse_duplicate_moneyness(log_mny, iv, weights)

    exact_atm = np.abs(x) <= float(atm_tol)
    if np.any(exact_atm):
        exact_weights = None if w is None else w[exact_atm]
        return _weighted_median(y[exact_atm], exact_weights)

    if x.size >= 2 and x[0] < 0.0 < x[-1]:
        return float(np.interp(0.0, x, y))

    n_keep = min(int(fallback_count), int(x.size))
    closest = np.argsort(np.abs(x))[:n_keep]
    closest_weights = None if w is None else w[closest]

    return _weighted_median(y[closest], closest_weights)


def _atm_variance_by_expiry(
    quotes: HestonQuoteSet,
    *,
    fallback_count: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate one ATM implied variance per expiry.

    For each expiry:
    - read quoted mid implied vols,
    - use log-moneyness x = log(K / F),
    - estimate IV at x = 0,
    - square that IV to get variance.
    """
    iv = _safe_iv_mid(quotes)

    expiry = np.asarray(quotes.expiry, dtype=np.float64).reshape(-1)
    log_mny = np.asarray(quotes.log_moneyness, dtype=np.float64).reshape(-1)

    if expiry.shape != iv.shape or log_mny.shape != iv.shape:
        raise ValueError("expiry, log_moneyness, and iv_mid must have matching shapes.")

    weights = None
    if quotes.bs_vega is not None:
        weights = np.asarray(quotes.bs_vega, dtype=np.float64).reshape(-1)
        if weights.shape != iv.shape:
            raise ValueError("bs_vega must have the same shape as iv_mid.")

    rows: list[tuple[float, float]] = []

    for tau in np.unique(expiry):
        idx = np.flatnonzero(expiry == tau)
        if idx.size == 0:
            continue

        slice_weights = None if weights is None else weights[idx]

        try:
            atm_iv = _atm_iv_from_slice(
                log_mny[idx],
                iv[idx],
                weights=slice_weights,
                fallback_count=fallback_count,
            )
        except ValueError:
            continue

        rows.append((float(tau), float(atm_iv * atm_iv)))

    if not rows:
        raise ValueError("Could not estimate ATM variance for any expiry.")

    rows.sort(key=lambda row: row[0])

    taus = np.array([row[0] for row in rows], dtype=np.float64)
    atm_variances = np.array([row[1] for row in rows], dtype=np.float64)

    return taus, atm_variances


def _estimate_atm_skew(
    quotes: HestonQuoteSet,
    *,
    max_points_per_expiry: int = 7,
) -> float:
    """Estimate rough near-ATM IV skew dIV / dlog_moneyness.

    Negative skew means IV is higher for lower strikes, which usually implies
    negative Heston rho for equity-like surfaces.
    """
    min_points = 3
    if max_points_per_expiry < min_points:
        raise ValueError("max_points_per_expiry must be at least 3.")

    iv = _safe_iv_mid(quotes)

    expiry = np.asarray(quotes.expiry, dtype=np.float64).reshape(-1)
    log_mny = np.asarray(quotes.log_moneyness, dtype=np.float64).reshape(-1)

    slopes: list[float] = []

    for tau in np.unique(expiry):
        idx = np.flatnonzero(expiry == tau)
        if idx.size < min_points:
            continue

        x = log_mny[idx]
        y = iv[idx]

        valid = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
        if np.count_nonzero(valid) < min_points:
            continue

        x = x[valid]
        y = y[valid]

        n_keep = min(max_points_per_expiry, x.size)

        central = np.argsort(np.abs(x))[:n_keep]
        x_c = x[central]
        y_c = y[central]

        if np.ptp(x_c) <= 1e-10:
            continue

        slope, _intercept = np.polyfit(x_c, y_c, deg=1)

        if np.isfinite(slope):
            slopes.append(float(slope))

    if not slopes:
        return 0.0

    return float(np.median(slopes))


def _rho_seed_from_skew(
    skew: float,
    *,
    fallback_rho: float = 0.0,
    rho_skew_scale: float = 1.25,
    rho_min: float = -0.85,
    rho_max: float = 0.50,
    skew_tol: float = 1e-10,
) -> float:
    """Map near-ATM IV skew to a Heston rho seed.

    Negative IV skew versus log-moneyness suggests negative Heston rho.
    This is a heuristic initialization rule, not an estimator.
    """
    if not np.isfinite(float(fallback_rho)):
        raise ValueError("fallback_rho must be finite.")
    if not np.isfinite(float(rho_skew_scale)) or float(rho_skew_scale) < 0.0:
        raise ValueError("rho_skew_scale must be finite and nonnegative.")
    if rho_min >= rho_max:
        raise ValueError("rho_min must be strictly less than rho_max.")

    if abs(float(skew)) <= float(skew_tol):
        rho = float(fallback_rho)
    else:
        rho = float(np.tanh(float(rho_skew_scale) * float(skew)))

    return float(np.clip(rho, rho_min, rho_max))


def default_heston_seed(
    quotes: HestonQuoteSet,
    *,
    bounds: HestonCalibrationBounds | None = None,
    fallback_rho: float = 0.0,
    rho_skew_scale: float = 1.25,
    kappa_seed: float = 1.50,
) -> HestonParams:
    """Build a deterministic market-aware initial Heston parameter guess.

    This is a seed, not an estimator.

    Heuristic
    ---------
    v
        Shortest-expiry ATM implied variance.

    vbar
        Longer-dated ATM implied variance, blended with the median ATM variance
        for robustness.

    kappa
        Moderate fixed mean-reversion speed. Kappa is deliberately not inferred
        aggressively from the ATM variance term structure because vanilla
        calibration can identify it weakly.

    rho
        Inferred from near-ATM implied-vol skew. Negative skew gives negative
        rho. If skew cannot be estimated, use ``fallback_rho``.

    eta
        Moderate vol-of-vol, increased when near-ATM skew is stronger.

    Notes
    -----
    The seed is designed to be good enough for nonlinear least-squares
    initialization. Final calibration should still use multi-start diagnostics.
    """
    if not np.isfinite(float(kappa_seed)) or float(kappa_seed) <= 0.0:
        raise ValueError("kappa_seed must be positive and finite.")

    resolved_bounds = HestonCalibrationBounds() if bounds is None else bounds

    _taus, atm_vars = _atm_variance_by_expiry(quotes)

    short_var = float(atm_vars[0])
    long_var = float(atm_vars[-1])
    median_var = float(np.median(atm_vars))

    v = short_var

    if atm_vars.size >= 2:
        vbar = 0.70 * long_var + 0.30 * median_var
    else:
        vbar = median_var

    kappa = float(kappa_seed)

    skew = _estimate_atm_skew(quotes)
    rho = _rho_seed_from_skew(
        skew,
        fallback_rho=fallback_rho,
        rho_skew_scale=rho_skew_scale,
    )

    eta = 0.50 + min(1.00, 2.00 * abs(skew))

    kappa_lo, kappa_hi = resolved_bounds.kappa
    vbar_lo, vbar_hi = resolved_bounds.vbar
    eta_lo, eta_hi = resolved_bounds.eta
    rho_lo, rho_hi = resolved_bounds.rho
    v_lo, v_hi = resolved_bounds.v

    return HestonParams(
        kappa=_clip(kappa, kappa_lo, kappa_hi),
        vbar=_clip(vbar, vbar_lo, vbar_hi),
        eta=_clip(eta, eta_lo, eta_hi),
        rho=_clip(rho, rho_lo, rho_hi),
        v=_clip(v, v_lo, v_hi),
    )


def _clip_params_to_bounds(
    params: HestonParams,
    bounds: HestonCalibrationBounds,
) -> HestonParams:
    kappa_lo, kappa_hi = bounds.kappa
    vbar_lo, vbar_hi = bounds.vbar
    eta_lo, eta_hi = bounds.eta
    rho_lo, rho_hi = bounds.rho
    v_lo, v_hi = bounds.v

    return HestonParams(
        kappa=_clip(params.kappa, kappa_lo, kappa_hi),
        vbar=_clip(params.vbar, vbar_lo, vbar_hi),
        eta=_clip(params.eta, eta_lo, eta_hi),
        rho=_clip(params.rho, rho_lo, rho_hi),
        v=_clip(params.v, v_lo, v_hi),
    )


def _dedupe_heston_seeds(
    seeds: list[HestonParams],
    *,
    decimals: int = 10,
) -> tuple[HestonParams, ...]:
    seen: set[tuple[float, ...]] = set()
    out: list[HestonParams] = []

    for seed in seeds:
        key = tuple(np.round(seed.as_array(), decimals=decimals))
        if key in seen:
            continue
        seen.add(key)
        out.append(seed)

    return tuple(out)


def heston_seed_grid(
    quotes: HestonQuoteSet,
    *,
    bounds: HestonCalibrationBounds | None = None,
    fallback_rho: float = 0.0,
    rho_skew_scale: float = 1.25,
    kappa_values: tuple[float, ...] = (0.50, 1.50, 3.00, 6.00),
    eta_multipliers: tuple[float, ...] = (0.60, 1.00, 1.75),
    vbar_multipliers: tuple[float, ...] = (0.75, 1.00, 1.25),
    rho_offsets: tuple[float, ...] = (-0.25, 0.0, 0.25),
    max_seeds: int | None = 12,
    include_default: bool = True,
) -> tuple[HestonParams, ...]:
    """Return deterministic market-aware Heston calibration seeds.

    The grid is intentionally compact. It explores the main weakly identified
    Heston directions without exploding into a large Cartesian product:
    mean reversion, vol-of-vol/rho skew tradeoff, and long-run variance level.

    The returned seeds are clipped to practical calibration bounds and
    deduplicated after clipping.
    """
    if max_seeds is not None and max_seeds <= 0:
        raise ValueError("max_seeds must be positive or None.")

    resolved_bounds = HestonCalibrationBounds() if bounds is None else bounds

    base = default_heston_seed(
        quotes,
        bounds=resolved_bounds,
        fallback_rho=fallback_rho,
        rho_skew_scale=rho_skew_scale,
    )

    seeds: list[HestonParams] = []

    def add_seed(
        *,
        kappa: float = base.kappa,
        vbar: float = base.vbar,
        eta: float = base.eta,
        rho: float = base.rho,
        v: float = base.v,
    ) -> None:
        seeds.append(
            _clip_params_to_bounds(
                HestonParams(
                    kappa=float(kappa),
                    vbar=float(vbar),
                    eta=float(eta),
                    rho=float(rho),
                    v=float(v),
                ),
                resolved_bounds,
            )
        )

    if include_default:
        add_seed()

    # 1. Mean-reversion spokes.
    for kappa in kappa_values:
        add_seed(kappa=kappa)

    # 2. Long-run variance spokes.
    for mult in vbar_multipliers:
        add_seed(vbar=base.vbar * float(mult))

    # 3. Vol-of-vol / rho skew tradeoff spokes.
    for eta_mult in eta_multipliers:
        add_seed(eta=base.eta * float(eta_mult))

    for rho_offset in rho_offsets:
        add_seed(rho=base.rho + float(rho_offset))

    # 4. A few combined skew-heavy starts.
    rho_lo, rho_hi = resolved_bounds.rho
    skew = _estimate_atm_skew(quotes)

    if skew < -1.0e-10:
        skew_rhos = (min(base.rho, -0.30), -0.60, -0.80)
    elif skew > 1.0e-10:
        skew_rhos = (max(base.rho, 0.20), 0.40, 0.60)
    else:
        skew_rhos = (-0.30, 0.0, 0.30)

    for rho in skew_rhos:
        add_seed(
            kappa=base.kappa,
            eta=max(base.eta, 1.00),
            rho=float(np.clip(rho, rho_lo, rho_hi)),
        )

    unique = _dedupe_heston_seeds(seeds)

    if max_seeds is not None:
        return unique[: int(max_seeds)]

    return unique
