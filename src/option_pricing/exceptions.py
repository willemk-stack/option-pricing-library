class InvalidOptionPriceError(ValueError):
    """Raised when an input option price violates no-arbitrage bounds.

    This error is raised by :func:`_validate_bounds` (and therefore by
    :func:`implied_vol_bs_result` / :func:`implied_vol_bs`) when the provided market
    option price is inconsistent with tight no-arbitrage bounds for European options
    under continuous rates and dividend yield.

    Notes
    -----
    The bounds used are:

    - Call: ``max(Fp - K*df, 0) <= C <= Fp``
    - Put : ``max(K*df - Fp, 0) <= P <= K*df``

    where ``df = exp(-r*tau)`` and ``Fp = S*exp(-q*tau)`` is the prepaid forward.
    """


class RootFindingError(Exception):
    """Base class for root-finding failures."""


class NotBracketedError(RootFindingError):
    """Raised when a bracketing method is called without a valid sign change."""


class NoConvergenceError(RootFindingError):
    """Raised when the method fails to converge within max_iter."""


class DerivativeTooSmallError(RootFindingError):
    """Raised when Newton's method cannot proceed due to tiny derivative."""


class NoBracketError(NotBracketedError):
    """Raised by ensure_bracket when it cannot find a bracketing interval."""
