"""
numerics/fd/stencils.py (pure coefficients/weights)
Responsibility: return stencil coefficients; no “apply along axis” logic.

Suggested functions:
- d1_central_nonuniform_coeffs(hm, hp) -> (dl, dd, du)
- d2_central_nonuniform_coeffs(hm, hp) -> (dl, dd, du)
- lagrange_3pt_weights(x0, x1, x2, x_eval, deriv=1|2) -> (w0, w1, w2)

Optional (if you want to de-duplicate more):
    d1_upwind_nonuniform_coeffs(hm, hp, b) (your PDE upwind logic)

Why this file exists: both PDE assembly and Dupire want the same math objects (dl, dd, du). PDE multiplies by coefficients to build tri-diags; Dupire multiplies by prices to compute derivatives.
"""


def d1_central_nonuniform_coeffs(hm, hp):
    """Central 3-point coefficients for the first derivative on a nonuniform grid.

    Given grid spacings:
        hm = x_i - x_{i-1}
        hp = x_{i+1} - x_i

    returns coefficients (dl, dd, du) such that:
        y'(x_i) ≈ dl*y_{i-1} + dd*y_i + du*y_{i+1}

    This stencil is second-order accurate for smooth functions on nonuniform grids.

    Parameters
    ----------
    hm, hp:
        Left/right spacings. May be scalars or arrays (vectorized).

    Returns
    -------
    (dl, dd, du):
        Coefficients with the same shape as `hm`/`hp`.
    """
    denom = hm * hp * (hm + hp)
    dl = -hp * hp / denom
    dd = (hp * hp - hm * hm) / denom
    du = hm * hm / denom
    return dl, dd, du


def d2_central_nonuniform_coeffs(hm, hp):
    """Central 3-point coefficients for the second derivative on a nonuniform grid.

    Given grid spacings:
        hm = x_i - x_{i-1}
        hp = x_{i+1} - x_i

    returns coefficients (dl, dd, du) such that:
        y''(x_i) ≈ dl*y_{i-1} + dd*y_i + du*y_{i+1}

    This stencil is second-order accurate for smooth functions on nonuniform grids.

    Parameters
    ----------
    hm, hp:
        Left/right spacings. May be scalars or arrays (vectorized).

    Returns
    -------
    (dl, dd, du):
        Coefficients with the same shape as `hm`/`hp`.
    """
    dl = 2.0 / (hm * (hm + hp))
    dd = -2.0 / (hm * hp)
    du = 2.0 / (hp * (hm + hp))
    return dl, dd, du


def d1_backward_coeffs(hm):  # (u_i - u_{i-1})/hm
    return -1.0 / hm, 1.0 / hm, 0.0


def d1_forward_coeffs(hp):  # (u_{i+1} - u_i)/hp
    return 0.0, -1.0 / hp, 1.0 / hp


def lagrange_3pt_weights(
    x0: float, x1: float, x2: float, x_eval: float, deriv: int
) -> tuple[float, float, float]:
    """
    Quadratic Lagrange 3-point finite-difference weights for derivative at x_eval
    using points x0, x1, x2.
    deriv: 1 or 2
    """
    if deriv == 1:
        w0 = (2.0 * x_eval - x1 - x2) / ((x0 - x1) * (x0 - x2))
        w1 = (2.0 * x_eval - x0 - x2) / ((x1 - x0) * (x1 - x2))
        w2 = (2.0 * x_eval - x0 - x1) / ((x2 - x0) * (x2 - x1))
        return w0, w1, w2
    if deriv == 2:
        w0 = 2.0 / ((x0 - x1) * (x0 - x2))
        w1 = 2.0 / ((x1 - x0) * (x1 - x2))
        w2 = 2.0 / ((x2 - x0) * (x2 - x1))
        return w0, w1, w2
    raise ValueError("deriv must be 1 or 2")
