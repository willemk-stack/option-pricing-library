from __future__ import annotations


def solver_tau_to_surface_expiry(*, solver_tau: float) -> float:
    """Map PDE solver time to the LocalVolSurface expiry convention.

    The 1D PDE stack advances forward in solver time
    ``tau = expiry - valuation_time`` starting from the terminal payoff at
    ``tau = 0``. In this codebase ``LocalVolSurface.local_var(K, T)`` uses the
    same expiry/time-to-expiry convention. The adapter is therefore
    intentionally the identity.
    """

    return float(solver_tau)
