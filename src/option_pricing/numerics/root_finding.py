from __future__ import annotations

from collections.abc import Callable


def bisection_method(
    Fn: Callable[[float], float],
    a_0: float,
    b_0: float,
    tol_f: float = 1e-8,
    tol_x: float = 1e-12,
    max_iter: int = 10_000,
    **_,  # ignore x0, dFn, etc.
) -> float:
    low, high = (a_0, b_0) if a_0 <= b_0 else (b_0, a_0)

    f_low = Fn(low)
    if abs(f_low) <= tol_f:
        return low

    f_high = Fn(high)
    if abs(f_high) <= tol_f:
        return high

    if f_low * f_high > 0:
        raise ValueError(
            "Bisection requires Fn(low) and Fn(high) to have opposite signs."
        )

    for _ in range(max_iter):
        mid = low + (high - low) / 2.0
        f_mid = Fn(mid)

        if abs(f_mid) <= tol_f:
            return mid

        if f_low * f_mid < 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid

        if (high - low) / 2.0 <= tol_x:
            return low + (high - low) / 2.0

    raise RuntimeError("Bisection did not converge within max_iter.")


def newton_method(
    Fn: Callable[[float], float],
    x0: float,
    dFn: Callable[[float], float] | None = None,
    tol_f: float = 1e-10,
    tol_x: float = 1e-12,
    max_iter: int = 50,
) -> float:
    x = x0
    for _ in range(max_iter):
        fx = Fn(x)
        if abs(fx) <= tol_f:
            return x

        if dFn is None:
            eps = 1e-5 * max(1.0, abs(x))
            dfx = (Fn(x + eps) - Fn(x - eps)) / (2.0 * eps)
        else:
            dfx = dFn(x)

        if dfx == 0.0 or abs(dfx) < 1e-14:
            raise RuntimeError("Newton failed: derivative too small.")

        x_new = x - fx / dfx

        if abs(x_new - x) <= tol_x * max(1.0, abs(x)):
            return x_new

        x = x_new

    raise RuntimeError("Newton did not converge within max_iter.")


def bracketed_newton(
    Fn: Callable[[float], float],
    a_0: float,
    b_0: float,
    dFn: Callable[[float], float] | None = None,
    x0: float | None = None,
    tol_f: float = 1e-10,
    tol_x: float = 1e-12,
    max_iter: int = 100,
) -> float:
    low, high = (a_0, b_0) if a_0 <= b_0 else (b_0, a_0)

    f_low = Fn(low)
    if abs(f_low) <= tol_f:
        return low

    f_high = Fn(high)
    if abs(f_high) <= tol_f:
        return high

    if f_low * f_high > 0:
        raise ValueError(
            "Root not bracketed: Fn(low) and Fn(high) must have opposite signs."
        )

    x = x0 if (x0 is not None and low < x0 < high) else 0.5 * (low + high)

    for _ in range(max_iter):
        fx = Fn(x)
        if abs(fx) <= tol_f:
            return x

        # Always maintain bracket
        if f_low * fx < 0:
            high, f_high = x, fx
        else:
            low, f_low = x, fx

        # Stop if bracket is tiny
        if (high - low) / 2.0 <= tol_x:
            return low + (high - low) / 2.0

        # Derivative for Newton candidate
        if dFn is None:
            eps = 1e-5 * max(1.0, abs(x))
            dfx = (Fn(x + eps) - Fn(x - eps)) / (2.0 * eps)
        else:
            dfx = dFn(x)

        # Newton step if safe, else bisection
        if dfx != 0.0 and abs(dfx) >= 1e-14:
            cand = x - fx / dfx
            x_new = cand if (low < cand < high) else 0.5 * (low + high)
        else:
            x_new = 0.5 * (low + high)

        # Step tolerance
        if abs(x_new - x) <= tol_x * max(1.0, abs(x)):
            return x_new

        x = x_new

    raise RuntimeError("Bracketed Newton did not converge within max_iter.")


def ensure_bracket(
    Fn: Callable[[float], float],
    lo: float,
    hi: float,
    hi_max: float = 10.0,
    grow: float = 2.0,
) -> tuple[float, float]:
    if lo <= 0 or hi <= 0 or lo >= hi:
        raise ValueError("Require 0 < lo < hi.")

    f_lo = Fn(lo)
    f_hi = Fn(hi)

    # already bracketed (or exactly at root)
    if f_lo == 0.0 or f_hi == 0.0 or f_lo * f_hi < 0:
        return lo, hi

    # expand hi until sign flips or we hit cap
    while f_lo * f_hi > 0 and hi < hi_max:
        hi = min(hi * grow, hi_max)
        f_hi = Fn(hi)

        if f_hi == 0.0 or f_lo * f_hi < 0:
            return lo, hi

    # If we get here, still no bracket.
    # Interpret common cases for IV:
    if f_lo > 0 and f_hi > 0:
        raise ValueError(
            "No bracket: model price > market price even at very low vol (market too low?)."
        )
    else:
        raise ValueError(
            "No bracket: model price < market price even at very high vol (market too high or hi_max too small?)."
        )
