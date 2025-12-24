from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# ---------------------------
# Results + Exceptions
# ---------------------------


@dataclass(frozen=True, slots=True)
class RootResult:
    root: float
    converged: bool
    iterations: int
    method: str
    f_at_root: float
    bracket: tuple[float, float] | None = None


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


def _clamp(x: float, domain: tuple[float, float] | None) -> float:
    if domain is None:
        return x
    lo, hi = domain
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


# ---------------------------
# Root finders (unified signature)
# All root methods accept:
#   (Fn, lo, hi, *, x0=None, dFn=None, tol_f=..., tol_x=..., max_iter=..., domain=None, **kwargs)
# and return RootResult
# ---------------------------


def bisection_method(
    Fn: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    x0: float | None = None,  # ignored (kept for signature compatibility)
    dFn: Callable[[float], float] | None = None,  # ignored
    tol_f: float = 1e-8,
    tol_x: float = 1e-12,
    max_iter: int = 10_000,
    domain: tuple[float, float] | None = None,
    **ignored_kwargs: Any,
) -> RootResult:
    a, b = (lo, hi) if lo <= hi else (hi, lo)
    a = _clamp(a, domain)
    b = _clamp(b, domain)

    fa = Fn(a)
    if abs(fa) <= tol_f:
        return RootResult(
            root=a,
            converged=True,
            iterations=0,
            method="bisection",
            f_at_root=fa,
            bracket=(a, b),
        )

    fb = Fn(b)
    if abs(fb) <= tol_f:
        return RootResult(
            root=b,
            converged=True,
            iterations=0,
            method="bisection",
            f_at_root=fb,
            bracket=(a, b),
        )

    if fa * fb > 0:
        raise NotBracketedError(
            "Bisection requires Fn(lo) and Fn(hi) to have opposite signs."
        )

    for it in range(1, max_iter + 1):
        mid = a + (b - a) / 2.0
        mid = _clamp(mid, domain)
        fmid = Fn(mid)

        if abs(fmid) <= tol_f:
            return RootResult(
                root=mid,
                converged=True,
                iterations=it,
                method="bisection",
                f_at_root=fmid,
                bracket=(a, b),
            )

        # Maintain the bracket
        if fa * fmid < 0:
            b, fb = mid, fmid
        else:
            a, fa = mid, fmid

        # Interval tolerance
        if (b - a) / 2.0 <= tol_x:
            root = a + (b - a) / 2.0
            root = _clamp(root, domain)
            froot = Fn(root)
            return RootResult(
                root=root,
                converged=True,
                iterations=it,
                method="bisection",
                f_at_root=froot,
                bracket=(a, b),
            )

    raise NoConvergenceError("Bisection did not converge within max_iter.")


def newton_method(
    Fn: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    x0: float | None = None,
    dFn: Callable[[float], float] | None = None,
    tol_f: float = 1e-10,
    tol_x: float = 1e-12,
    max_iter: int = 50,
    domain: tuple[float, float] | None = None,
    **ignored_kwargs: Any,
) -> RootResult:
    # Unified interface: use x0 if provided, else midpoint(lo, hi)
    a, b = (lo, hi) if lo <= hi else (hi, lo)
    x = x0 if x0 is not None else 0.5 * (a + b)
    x = _clamp(x, domain)

    for it in range(1, max_iter + 1):
        fx = Fn(x)
        if abs(fx) <= tol_f:
            return RootResult(
                root=x, converged=True, iterations=it - 1, method="newton", f_at_root=fx
            )

        if dFn is None:
            eps = 1e-5 * max(1.0, abs(x))
            dfx = (Fn(x + eps) - Fn(x - eps)) / (2.0 * eps)
        else:
            dfx = dFn(x)

        if dfx == 0.0 or abs(dfx) < 1e-14:
            raise DerivativeTooSmallError("Newton failed: derivative too small.")

        x_new = x - fx / dfx
        x_new = _clamp(x_new, domain)

        # Step tolerance (relative)
        if abs(x_new - x) <= tol_x * max(1.0, abs(x_new)):
            f_new = Fn(x_new)
            return RootResult(
                root=x_new,
                converged=True,
                iterations=it,
                method="newton",
                f_at_root=f_new,
            )

        x = x_new

    raise NoConvergenceError("Newton did not converge within max_iter.")


def bracketed_newton(
    Fn: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    x0: float | None = None,
    dFn: Callable[[float], float] | None = None,
    tol_f: float = 1e-10,
    tol_x: float = 1e-12,
    max_iter: int = 100,
    domain: tuple[float, float] | None = None,
    **ignored_kwargs: Any,
) -> RootResult:
    a, b = (lo, hi) if lo <= hi else (hi, lo)
    a = _clamp(a, domain)
    b = _clamp(b, domain)

    fa = Fn(a)
    if abs(fa) <= tol_f:
        return RootResult(
            root=a,
            converged=True,
            iterations=0,
            method="bracketed_newton",
            f_at_root=fa,
            bracket=(a, b),
        )

    fb = Fn(b)
    if abs(fb) <= tol_f:
        return RootResult(
            root=b,
            converged=True,
            iterations=0,
            method="bracketed_newton",
            f_at_root=fb,
            bracket=(a, b),
        )

    if fa * fb > 0:
        raise NotBracketedError(
            "Root not bracketed: Fn(lo) and Fn(hi) must have opposite signs."
        )

    x = x0 if (x0 is not None and a < x0 < b) else 0.5 * (a + b)
    x = _clamp(x, domain)

    for it in range(1, max_iter + 1):
        fx = Fn(x)
        if abs(fx) <= tol_f:
            return RootResult(
                root=x,
                converged=True,
                iterations=it - 1,
                method="bracketed_newton",
                f_at_root=fx,
                bracket=(a, b),
            )

        # Maintain bracket
        if fa * fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx

        # Bracket tolerance
        if (b - a) / 2.0 <= tol_x:
            root = a + (b - a) / 2.0
            root = _clamp(root, domain)
            froot = Fn(root)
            return RootResult(
                root=root,
                converged=True,
                iterations=it,
                method="bracketed_newton",
                f_at_root=froot,
                bracket=(a, b),
            )

        # Derivative for Newton candidate
        if dFn is None:
            eps = 1e-5 * max(1.0, abs(x))
            dfx = (Fn(x + eps) - Fn(x - eps)) / (2.0 * eps)
        else:
            dfx = dFn(x)

        # Newton step if safe, else bisection
        if dfx != 0.0 and abs(dfx) >= 1e-14:
            cand = x - fx / dfx
            cand = _clamp(cand, domain)
            x_new = cand if (a < cand < b) else 0.5 * (a + b)
        else:
            x_new = 0.5 * (a + b)

        x_new = _clamp(x_new, domain)

        # Step tolerance
        if abs(x_new - x) <= tol_x * max(1.0, abs(x_new)):
            f_new = Fn(x_new)
            return RootResult(
                root=x_new,
                converged=True,
                iterations=it,
                method="bracketed_newton",
                f_at_root=f_new,
                bracket=(a, b),
            )

        x = x_new

    raise NoConvergenceError("Bracketed Newton did not converge within max_iter.")


def ensure_bracket(
    Fn: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    hi_max: float = 10.0,
    grow: float = 2.0,
    max_steps: int = 60,
    domain: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """
    Expand `hi` geometrically until Fn(lo) and Fn(hi) have opposite signs (or hi hits hi_max).
    - Designed to be IV-friendly (lo can be 0.0), but also usable generally.
    - If domain is provided, lo/hi are clamped to it.

    Returns (lo, hi) such that Fn(lo) == 0 or Fn(hi) == 0 or Fn(lo)*Fn(hi) < 0.
    """
    if hi <= lo:
        raise ValueError("Require lo < hi.")
    if grow <= 1.0:
        raise ValueError("Require grow > 1.0.")
    if hi_max <= hi:
        # Still allow: user might have hi==hi_max; we just won't grow.
        hi_max = hi

    lo_c = _clamp(lo, domain)
    hi_c = _clamp(hi, domain)
    if hi_c <= lo_c:
        raise ValueError("Clamped bounds invalid: require lo < hi within domain.")

    f_lo = Fn(lo_c)
    f_hi = Fn(hi_c)

    if f_lo == 0.0 or f_hi == 0.0 or f_lo * f_hi < 0:
        return lo_c, hi_c

    steps = 0
    while f_lo * f_hi > 0 and hi_c < hi_max and steps < max_steps:
        hi_c = min(hi_c * grow, hi_max)
        hi_c = _clamp(hi_c, domain)
        f_hi = Fn(hi_c)
        steps += 1

        if f_hi == 0.0 or f_lo * f_hi < 0:
            return lo_c, hi_c

        if hi_c >= hi_max:
            break

    # Still no bracket: interpret common IV-like cases but keep generic info too.
    if f_lo > 0 and f_hi > 0:
        raise NoBracketError(
            "No bracket found: Fn(lo) and Fn(hi) stayed positive while expanding hi. "
            "For IV this often means market price is too low (below theoretical lower bound) "
            "or model price is always above market over the searched range."
        )
    else:
        raise NoBracketError(
            "No bracket found: Fn(lo) and Fn(hi) stayed negative while expanding hi. "
            "For IV this often means market price is too high (above theoretical upper bound) "
            "or hi_max is too small for the needed root."
        )


# ---------------------------
# Convenience wrappers (optional): keep old "float return" style
# ---------------------------


def root_as_float(result: RootResult) -> float:
    return result.root


def bisection_root(*args: Any, **kwargs: Any) -> float:
    return bisection_method(*args, **kwargs).root


def newton_root(*args: Any, **kwargs: Any) -> float:
    return newton_method(*args, **kwargs).root


def bracketed_newton_root(*args: Any, **kwargs: Any) -> float:
    return bracketed_newton(*args, **kwargs).root
