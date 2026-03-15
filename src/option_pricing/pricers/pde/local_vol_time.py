from __future__ import annotations


def solver_tau_to_surface_expiry(
    *,
    option_expiry: float,
    solver_tau: float,
    surface_expiry_floor: float = 1.0e-8,
) -> float:
    """Map PDE solver time to the LocalVolSurface calendar-time convention.

    The 1D PDE stack advances forward in solver time
    ``tau = expiry - t`` starting from the terminal payoff at ``tau = 0``.
    ``LocalVolSurface.local_var(K, T)`` is parameterized by calendar time from
    the valuation origin, so the correct surface query is ``T = expiry - tau``.
    A small floor avoids querying exactly at zero maturity where some local-vol
    constructions are numerically delicate.
    """

    return max(float(option_expiry) - float(solver_tau), float(surface_expiry_floor))
