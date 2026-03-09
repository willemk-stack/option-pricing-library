"""Diagnostics-style tests for time seams / "jumps" in implied + Dupire paths.

These tests are intentionally written to *explain* where a visible seam/jump can
come from:

1) theta isotonic regression can change ATM total variance at nodes. If the
   no-arb interpolator uses the regressed theta but the surface "snaps" to the
   stored smile at exact expiries, you can create an actual discontinuity.

2) No-arb time interpolation uses implied-vol inversion with hard sigma bounds.
   If a required sigma exceeds sigma_hi (default 5.0), off-node slices will
   clip, while node slices (stored smiles) may not. This can produce a large
   visible step in plots.

3) Even when the surface is C0 in time, piecewise-in-time construction is not
   C1 at expiry boundaries. Dupire/Gatheral local vol depends on w_T, so this
   lack of time smoothness shows up as discontinuous w_T (and banding /
   instability).
"""

from __future__ import annotations

import numpy as np
import pytest

from option_pricing.vol import LocalVolSurface, NoArbInterpolatedSmileSlice, VolSurface
from option_pricing.vol.svi.models import SVIParams, SVISmile


@pytest.fixture
def _fwd() -> callable:
    return lambda _T: 100.0


class _ConstSmile:
    """Constant-IV smile slice (no y-dependence).

    w(y,T) = T * sigma^2.
    """

    def __init__(self, *, T: float, sigma: float):
        self.T = float(T)
        self.sigma = float(sigma)
        self.y_min = -10.0
        self.y_max = 10.0

    def w_at(self, yq):
        y = np.asarray(yq, dtype=np.float64)
        out = np.full_like(y, self.T * self.sigma**2, dtype=np.float64)
        return out.reshape(y.shape) if y.ndim else np.asarray(out.reshape(()))

    def iv_at(self, yq):
        y = np.asarray(yq, dtype=np.float64)
        out = np.full_like(y, self.sigma, dtype=np.float64)
        return out.reshape(y.shape) if y.ndim else np.asarray(out.reshape(()))


def _finite_diff_wT(surface: VolSurface, *, y: float, T: float, dt: float) -> float:
    """One-sided slope approximation for w_T at constant y."""
    w1 = float(np.asarray(surface.w(np.asarray([y]), T + dt))[0])
    w0 = float(np.asarray(surface.w(np.asarray([y]), T))[0])
    return (w1 - w0) / dt


def _svi_total_variance(y: np.ndarray, p: SVIParams) -> np.ndarray:
    """SVI total variance w(y) for the library's raw-parameterization."""
    z = y - np.float64(p.m)
    return np.float64(p.a) + np.float64(p.b) * (
        np.float64(p.rho) * z + np.sqrt(z * z + np.float64(p.sigma) ** 2)
    )


def _rows_from_svi(
    *,
    expiries: np.ndarray,
    params: list[SVIParams],
    forward: callable,
    y_grid: np.ndarray,
) -> list[tuple[float, float, float]]:
    """Generate synthetic (T, K, iv) rows from exact SVI smiles."""
    rows: list[tuple[float, float, float]] = []
    for T, p in zip(expiries, params, strict=True):
        F = float(forward(float(T)))
        K = (np.float64(F) * np.exp(y_grid)).astype(np.float64)
        w = _svi_total_variance(y_grid, p)
        iv = np.sqrt(np.maximum(w / np.float64(T), 0.0))
        for k, sig in zip(K.tolist(), iv.tolist(), strict=True):
            rows.append((float(T), float(k), float(sig)))
    return rows


def test_slice_snaps_to_node_expiry_but_interpolates_off_node(_fwd) -> None:
    """This is the mechanism behind 'node snapping' seams in plots."""
    exp = np.array([0.5, 1.0], dtype=np.float64)
    s0 = _ConstSmile(T=0.5, sigma=0.2)
    s1 = _ConstSmile(T=1.0, sigma=0.25)

    surf = VolSurface(expiries=exp, smiles=(s0, s1), forward=_fwd)

    # Exact expiry => returns the stored slice object.
    assert surf.slice(1.0) is s1

    # Nearby but not equal => returns an interpolated slice (different code path).
    sl = surf.slice(1.0 - 1e-6)
    assert isinstance(sl, NoArbInterpolatedSmileSlice)


def test_isotonic_theta_can_create_a_true_jump_at_nodes(_fwd) -> None:
    """If isotonic regression changes theta(T_i), the surface can become discontinuous.

    Rationale:
    - No-arb interpolator's alpha(T) uses theta_interp(T).
    - VolSurface.slice(T_i) *snaps* to the stored smile at exact nodes.
    - If theta_interp(T_i) != smile_i.w_at(0), then the limiting interpolated
      slice as T->T_i^+/- generally won't match the stored node slice.
    """
    # Construct calendar-arbitrage ATM total variance: theta decreases.
    exp = np.array([1.0, 2.0], dtype=np.float64)
    s0 = _ConstSmile(T=1.0, sigma=0.30)  # theta0 = 0.09
    s1 = _ConstSmile(T=2.0, sigma=0.10)  # theta1 = 0.02  (decreasing)

    surf = VolSurface(expiries=exp, smiles=(s0, s1), forward=_fwd)

    # Node value comes from the stored slice.
    iv_node = float(np.asarray(surf.iv(np.array([100.0]), 1.0))[0])

    # Immediately off-node uses the no-arb interpolator which uses *regressed* theta.
    # If regression changed theta at T=1.0, this difference will persist as eps -> 0.
    eps = 1e-8
    iv_off = float(np.asarray(surf.iv(np.array([100.0]), 1.0 + eps))[0])

    assert abs(iv_off - iv_node) > 1e-3, (
        "Expected a persistent jump when isotonic regression modifies theta at the node. "
        f"Got iv_node={iv_node:.6g}, iv_off={iv_off:.6g}, diff={iv_off-iv_node:.3g}"
    )


def test_noarb_implied_vol_inversion_sigma_bounds_can_create_large_steps(_fwd) -> None:
    """If required sigma exceeds implied-vol solver bounds, no-arb slices clip.

    The no-arb interpolator inverts call prices with sigma_hi=5.0. If the stored
    endpoint smile implies sigma > 5, then:
      * surface.slice(T_i) at the node returns the stored smile (sigma can be > 5)
      * surface.slice(T_i + eps) uses the interpolator and will clip to <= 5
    yielding a visible step.
    """

    exp = np.array([0.10, 0.20], dtype=np.float64)
    s0 = _ConstSmile(T=0.10, sigma=10.0)  # far above the 5.0 inversion cap
    s1 = _ConstSmile(T=0.20, sigma=10.0)
    surf = VolSurface(expiries=exp, smiles=(s0, s1), forward=_fwd)

    K = np.array([100.0], dtype=np.float64)
    iv_node = float(np.asarray(surf.iv(K, 0.10))[0])
    iv_off = float(np.asarray(surf.iv(K, 0.10 + 1e-6))[0])

    assert iv_node > 9.0  # sanity
    assert iv_off <= 5.0 + 1e-12
    assert (iv_node - iv_off) > 4.0, (
        "Expected a large step when the no-arb inversion clips at sigma_hi=5.0. "
        f"Got iv_node={iv_node:.6g}, iv_off={iv_off:.6g}"
    )


def test_implied_surface_is_C0_but_not_C1_in_time_across_expiry_boundaries(
    _fwd,
) -> None:
    """Quantify the time-derivative seam in the implied surface.

    For piecewise-in-time construction, w(y,T) is typically continuous, but its
    time derivative w_T(y,T) changes when the bracketing expiries change.

    This is the root cause for 'seams' in 3D plots and for Dupire sensitivity.
    """

    exp = np.array([0.25, 0.50, 1.00], dtype=np.float64)
    smiles = (
        SVISmile(T=0.25, params=SVIParams(a=0.01, b=0.05, rho=-0.3, m=0.0, sigma=0.30)),
        SVISmile(T=0.50, params=SVIParams(a=0.02, b=0.06, rho=-0.2, m=0.0, sigma=0.25)),
        SVISmile(T=1.00, params=SVIParams(a=0.04, b=0.07, rho=-0.1, m=0.0, sigma=0.22)),
    )
    surf = VolSurface(expiries=exp, smiles=smiles, forward=_fwd)

    y = 0.50  # off-ATM: ensures the seam is visible in w_T
    T_knot = 0.50
    dt = 1e-5

    # left slope inside [0.25, 0.50)
    wT_left = _finite_diff_wT(surf, y=y, T=T_knot - 2 * dt, dt=dt)
    # right slope inside (0.50, 1.00]
    wT_right = _finite_diff_wT(surf, y=y, T=T_knot + dt, dt=dt)

    assert abs(wT_right - wT_left) > 1e-2, (
        "Expected a noticeable jump in w_T across the expiry boundary (C0 but not C1). "
        f"Got wT_left={wT_left:.6g}, wT_right={wT_right:.6g}, diff={wT_right-wT_left:.3g}"
    )


def test_localvolsurface_wT_jumps_off_atm_at_expiry_boundaries(_fwd) -> None:
    """LocalVolSurface fallback uses a secant/interval-based w_T which jumps at knots."""

    exp = np.array([0.25, 0.50, 1.00], dtype=np.float64)
    smiles = (
        SVISmile(T=0.25, params=SVIParams(a=0.01, b=0.05, rho=-0.3, m=0.0, sigma=0.30)),
        SVISmile(T=0.50, params=SVIParams(a=0.02, b=0.06, rho=-0.2, m=0.0, sigma=0.25)),
        SVISmile(T=1.00, params=SVIParams(a=0.04, b=0.07, rho=-0.1, m=0.0, sigma=0.22)),
    )
    implied = VolSurface(expiries=exp, smiles=smiles, forward=_fwd)

    with pytest.warns(FutureWarning):
        lv = LocalVolSurface.from_implied(
            implied, forward=_fwd, discount=lambda _T: 1.0
        )

    y = np.array([0.50], dtype=np.float64)
    eps = 1e-6

    *_, wT_left = lv._w_and_derivs(y, 0.50 - eps)
    *_, wT_right = lv._w_and_derivs(y, 0.50 + eps)

    jump = float(wT_right[0] - wT_left[0])
    assert abs(jump) > 1e-3, (
        "Expected a w_T jump off-ATM at the expiry boundary under the current secant-in-time fallback. "
        f"Got wT_left={float(wT_left[0]):.6g}, wT_right={float(wT_right[0]):.6g}, jump={jump:.3g}"
    )


# ---------------------------------------------------------------------------
# End-to-end: the *recommended* construction path (VolSurface.from_svi)
# ---------------------------------------------------------------------------


def test_from_svi_snaps_to_node_expiries_and_interpolates_off_node(_fwd) -> None:
    """from_svi produces SVISmile nodes, but time slices still snap/interpolate."""

    pytest.importorskip("scipy")

    expiries = np.array([0.25, 0.50, 1.00], dtype=np.float64)
    params = [
        # (Nearly) flat smiles via b=0, but different levels to make time seams visible.
        SVIParams(a=0.25 * 0.20**2, b=0.0, rho=0.0, m=0.0, sigma=0.20),
        SVIParams(a=0.50 * 0.30**2, b=0.0, rho=0.0, m=0.0, sigma=0.20),
        SVIParams(a=1.00 * 0.25**2, b=0.0, rho=0.0, m=0.0, sigma=0.20),
    ]
    y_grid = np.linspace(-0.4, 0.4, 9, dtype=np.float64)
    rows = _rows_from_svi(expiries=expiries, params=params, forward=_fwd, y_grid=y_grid)

    surf = VolSurface.from_svi(
        rows,
        forward=_fwd,
        calibrate_kwargs={
            # keep the optimizer cheap + deterministic for tests
            "loss": "linear",
            "irls_max_outer": 1,
            "robust_data_only": False,
            "repair_butterfly": False,
            "refit_max_nfev": 300,
        },
    )

    # exact node => stored analytic SVI slice
    sl_node = surf.slice(0.50)
    assert isinstance(sl_node, SVISmile)

    # off node => no-arb interpolated slice
    sl_off = surf.slice(0.50 + 1e-6)
    assert isinstance(sl_off, NoArbInterpolatedSmileSlice)


def test_from_svi_surface_is_C0_but_wT_has_a_knot_seam(_fwd) -> None:
    """End-to-end seam: continuity holds, but w_T jumps across an expiry boundary."""

    pytest.importorskip("scipy")

    expiries = np.array([0.25, 0.50, 1.00], dtype=np.float64)
    params = [
        # Use genuinely different SVI shapes so the piecewise time construction
        # has a measurable w_T seam away from ATM.
        SVIParams(a=0.01, b=0.06, rho=-0.3, m=0.0, sigma=0.35),
        SVIParams(a=0.02, b=0.07, rho=-0.2, m=0.0, sigma=0.28),
        SVIParams(a=0.04, b=0.08, rho=-0.1, m=0.0, sigma=0.23),
    ]

    y_grid = np.linspace(-0.8, 0.8, 17, dtype=np.float64)
    rows = _rows_from_svi(expiries=expiries, params=params, forward=_fwd, y_grid=y_grid)
    surf = VolSurface.from_svi(
        rows,
        forward=_fwd,
        calibrate_kwargs={
            "loss": "linear",
            "irls_max_outer": 1,
            "robust_data_only": False,
            "repair_butterfly": False,
            "refit_max_nfev": 500,
        },
    )

    y = 0.50
    T_knot = 0.50
    eps = 1e-6

    # C0 check (total variance should be continuous to plotting tolerance)
    w_node = float(np.asarray(surf.w(np.asarray([y]), T_knot))[0])
    w_right = float(np.asarray(surf.w(np.asarray([y]), T_knot + eps))[0])
    w_left = float(np.asarray(surf.w(np.asarray([y]), T_knot - eps))[0])

    # These are tiny-difference checks: if this fails, you likely have a *true jump*
    # (most commonly isotonic-theta mismatch or sigma_hi clipping).
    assert abs(w_right - w_node) <= 5e-5 * max(1.0, abs(w_node))
    assert abs(w_left - w_node) <= 5e-5 * max(1.0, abs(w_node))

    # But the time derivative is not smooth (non-C1): measure it with a one-sided slope.
    dt = 1e-4
    wT_left = _finite_diff_wT(surf, y=y, T=T_knot - 2 * dt, dt=dt)
    wT_right = _finite_diff_wT(surf, y=y, T=T_knot + dt, dt=dt)

    assert abs(wT_right - wT_left) > 1e-3, (
        "Expected a noticeable w_T seam across the expiry boundary even when built via from_svi. "
        f"Got wT_left={wT_left:.6g}, wT_right={wT_right:.6g}, diff={wT_right-wT_left:.3g}"
    )


def test_from_svi_isotonic_theta_mismatch_creates_a_real_jump(_fwd) -> None:
    """End-to-end: if ATM total variance is non-monotone, isotonic can create a discontinuity."""

    pytest.importorskip("scipy")

    # Force decreasing ATM theta across expiries (calendar arbitrage).
    expiries = np.array([1.00, 2.00], dtype=np.float64)

    # Choose nearly-flat smiles so the jump is attributable to theta regression, not smile shape.
    p0 = SVIParams(
        a=1.00 * 0.35**2, b=0.0, rho=0.0, m=0.0, sigma=0.20
    )  # theta0 ~ 0.1225
    p1 = SVIParams(
        a=2.00 * 0.10**2, b=0.0, rho=0.0, m=0.0, sigma=0.20
    )  # theta1 ~ 0.02 (decrease)
    y_grid = np.array([-0.4, -0.2, 0.0, 0.2, 0.4], dtype=np.float64)
    rows = _rows_from_svi(
        expiries=expiries, params=[p0, p1], forward=_fwd, y_grid=y_grid
    )

    surf = VolSurface.from_svi(
        rows,
        forward=_fwd,
        calibrate_kwargs={
            "loss": "linear",
            "irls_max_outer": 1,
            "robust_data_only": False,
            "repair_butterfly": False,
            "refit_max_nfev": 300,
        },
    )

    K_atm = np.array([100.0], dtype=np.float64)
    iv_node = float(np.asarray(surf.iv(K_atm, 1.00))[0])
    iv_off = float(np.asarray(surf.iv(K_atm, 1.00 + 1e-8))[0])

    assert abs(iv_off - iv_node) > 1e-3, (
        "Expected a persistent node jump when isotonic regression modifies ATM theta at the node. "
        f"Got iv_node={iv_node:.6g}, iv_off={iv_off:.6g}, diff={iv_off-iv_node:.3g}"
    )


def test_from_svi_diagnosis_report_prints_metrics(_fwd) -> None:
    """Pure diagnostic report for the from_svi pipeline.

    This test is designed to **always pass** (unless something is fundamentally broken)
    and to print a small metrics table that helps you quantify:

      * node snapping impact: max |IV(T_i) - IV(T_i±eps)|
      * time-smoothness seam: max |w_T^+ - w_T^-| across expiry boundaries
      * potential implied-vol inversion clipping: how often IV is ~ sigma_hi (5.0)

    Run with `-s` to see output:
        pytest -q -s tests/test_surface_time_diagnosis.py -k diagnosis_report
    """

    pytest.importorskip("scipy")

    expiries = np.array([0.25, 0.50, 1.00, 2.00], dtype=np.float64)
    params = [
        # Deliberately different shapes/levels to make seams measurable.
        SVIParams(a=0.008, b=0.055, rho=-0.35, m=-0.05, sigma=0.33),
        SVIParams(a=0.014, b=0.065, rho=-0.25, m=-0.02, sigma=0.28),
        SVIParams(a=0.028, b=0.075, rho=-0.15, m=0.00, sigma=0.24),
        SVIParams(a=0.060, b=0.085, rho=-0.05, m=0.02, sigma=0.22),
    ]
    y_grid = np.linspace(-1.0, 1.0, 41, dtype=np.float64)
    rows = _rows_from_svi(expiries=expiries, params=params, forward=_fwd, y_grid=y_grid)

    surf = VolSurface.from_svi(
        rows,
        forward=_fwd,
        calibrate_kwargs={
            # Keep it cheap/deterministic.
            "loss": "linear",
            "irls_max_outer": 1,
            "robust_data_only": False,
            "repair_butterfly": False,
            "refit_max_nfev": 800,
        },
    )

    eps = 1e-6
    dt = 1e-4
    sigma_hi = 5.0

    # ---------------------------------
    # Node-jump metrics (node snapping)
    # ---------------------------------
    node_rows: list[tuple[str, float, float, float]] = []
    y_eval = np.array([-0.75, -0.25, 0.0, 0.25, 0.75], dtype=np.float64)
    F0 = float(_fwd(0.0))
    K_eval = (F0 * np.exp(y_eval)).astype(np.float64)

    for i, T in enumerate(expiries.tolist()):
        # IV at node
        iv_node = np.asarray(surf.iv(K_eval, float(T)), dtype=np.float64)

        # One-sided nearby IV
        iv_minus = None
        iv_plus = None
        if i > 0 and T - eps > 0:
            iv_minus = np.asarray(surf.iv(K_eval, float(T - eps)), dtype=np.float64)
        if i < len(expiries) - 1:
            iv_plus = np.asarray(surf.iv(K_eval, float(T + eps)), dtype=np.float64)

        jump_minus = (
            float(np.max(np.abs(iv_node - iv_minus)))
            if iv_minus is not None
            else float("nan")
        )
        jump_plus = (
            float(np.max(np.abs(iv_node - iv_plus)))
            if iv_plus is not None
            else float("nan")
        )
        max_iv = float(np.max(iv_node))
        node_rows.append((f"{T:>5.2f}", jump_minus, jump_plus, max_iv))

    # ---------------------------------
    # w_T seam across expiry boundaries
    # ---------------------------------
    seam_rows: list[tuple[str, float, float, float]] = []
    for T_knot in expiries[1:-1].tolist():
        # Evaluate on a few y points (off-ATM makes seams more obvious)
        y_pts = [float(y) for y in (-0.75, -0.25, 0.25, 0.75)]
        diffs = []
        for y in y_pts:
            wT_left = _finite_diff_wT(surf, y=y, T=float(T_knot) - 2 * dt, dt=dt)
            wT_right = _finite_diff_wT(surf, y=y, T=float(T_knot) + dt, dt=dt)
            diffs.append(abs(wT_right - wT_left))
        seam_rows.append(
            (
                f"{T_knot:>5.2f}",
                float(max(diffs)),
                float(np.mean(diffs)),
                float(min(diffs)),
            )
        )

    # ---------------------------------
    # Potential sigma_hi clipping count
    # ---------------------------------
    clip_rows: list[tuple[str, int, float]] = []
    mids = 0.5 * (expiries[:-1] + expiries[1:])
    K_wide = (F0 * np.exp(y_grid)).astype(np.float64)
    for Tm in mids.tolist():
        iv_mid = np.asarray(surf.iv(K_wide, float(Tm)), dtype=np.float64)
        # Count values that are "at" the cap (within tolerance)
        at_cap = int(np.sum(iv_mid >= sigma_hi - 1e-10))
        clip_rows.append((f"{Tm:>5.2f}", at_cap, float(np.max(iv_mid))))

    # -----------------
    # Pretty printing
    # -----------------
    def _fmt(x: float) -> str:
        return "   n/a" if not np.isfinite(x) else f"{x:7.3e}"

    print("\n[from_svi diagnosis report]")
    print(
        "\nNode snap / near-node IV differences (max over y in [-0.75,-0.25,0,0.25,0.75])"
    )
    print("   T     |IV(T)-IV(T-eps)|  |IV(T)-IV(T+eps)|   max_IV(T)")
    print("---------+-----------------+-----------------+------------")
    for T, jm, jp, mx in node_rows:
        print(f" {T}   {_fmt(jm)}       {_fmt(jp)}      {mx:10.6f}")

    print("\nTime-derivative seam estimate at expiry boundaries (|wT_right - wT_left|)")
    print("  knotT   max_diff    mean_diff   min_diff")
    print("--------  --------    ---------   --------")
    for T, mx, mean, mn in seam_rows:
        print(f" {T}   {mx:8.3e}   {mean:9.3e}  {mn:8.3e}")

    print(
        "\nPotential sigma_hi clipping (count of IV >= 5.0 - tol on mid-interval slices)"
    )
    print("  midT   at_cap_count   max_IV")
    print("------  ------------   ------")
    for T, cnt, mx in clip_rows:
        print(f" {T}      {cnt:6d}     {mx:7.4f}")

    # Always-pass sanity checks: no NaNs/Infs in core evals.
    assert np.isfinite(np.asarray(surf.w(np.asarray([0.0]), float(expiries[0])))[0])
    assert np.all(np.isfinite(np.asarray(surf.iv(K_eval, float(expiries[0])))))


@pytest.mark.slow
def test_from_svi_diagnosis_report_on_capstone2_synthetic_dataset() -> None:
    """Run the seam/jump metrics on the Capstone2 demo calibration dataset.

    This matches the dataset used in the flagship demo notebook/recipe, and builds
    the surface via the recommended no-arb path:

        VolSurface.from_svi(rows_obs, ...)

    This is a *diagnostic* test: it prints a small table of metrics and performs
    only basic sanity assertions (finite outputs).

    Run with:
        pytest -q -s tests/test_surface_time_diagnosis.py -k capstone2
    """

    pytest.importorskip("scipy")

    from option_pricing.data_generators.recipes import (
        generate_synthetic_surface_latent_noarb,
    )
    from option_pricing.demos._capstone2_defaults import get_capstone2_defaults
    from option_pricing.vol.surface_core import isotonic_regression

    cfg = get_capstone2_defaults(seed=7)
    syn = generate_synthetic_surface_latent_noarb(
        enforce=bool(cfg.get("ENFORCE_ARB_FREE_LATENT_TRUTH", True)),
        max_rounds=int(cfg.get("SYNTH_MAX_ROUNDS", 8)),
        **cfg["SYNTH_CFG"],
    )

    synthetic = syn.synthetic
    rows_obs = [(float(t), float(k), float(iv)) for t, k, iv in synthetic.rows_obs]
    forward = synthetic.forward

    # Build surface via from_svi (recommended path). Keep it reasonably cheap.
    calib = dict(cfg.get("SVI_CALIB_NO_REPAIR", {}))
    calib.setdefault("refit_max_nfev", 600)

    surf = VolSurface.from_svi(rows_obs, forward=forward, calibrate_kwargs=calib)

    expiries = np.array(sorted({t for t, _, _ in rows_obs}), dtype=np.float64)
    y_grid = np.array(
        sorted({float(x) for x in np.asarray(synthetic.x, dtype=np.float64)}),
        dtype=np.float64,
    )

    eps = 1e-6
    dt = 1e-4
    sigma_hi = 5.0

    # --- theta / isotonic impact ---
    theta_raw = np.asarray(
        [float(np.asarray(s.w_at(0.0))) for s in surf.smiles], dtype=np.float64
    )
    theta_reg = isotonic_regression(theta_raw)
    theta_interp_at_nodes = np.asarray(surf._theta_interp(expiries), dtype=np.float64)

    # --- node snapping metrics ---
    y_eval = np.array([-0.30, -0.15, 0.0, 0.15, 0.30], dtype=np.float64)
    node_rows: list[tuple[str, float, float, float]] = []
    for i, T in enumerate(expiries.tolist()):
        F = float(forward(float(T)))
        K_eval = (np.float64(F) * np.exp(y_eval)).astype(np.float64)
        iv_node = np.asarray(surf.iv(K_eval, float(T)), dtype=np.float64)

        iv_minus = None
        iv_plus = None
        if i > 0 and T - eps > 0:
            Fm = float(forward(float(T - eps)))
            K_minus = (np.float64(Fm) * np.exp(y_eval)).astype(np.float64)
            iv_minus = np.asarray(surf.iv(K_minus, float(T - eps)), dtype=np.float64)
        if i < len(expiries) - 1:
            Fp = float(forward(float(T + eps)))
            K_plus = (np.float64(Fp) * np.exp(y_eval)).astype(np.float64)
            iv_plus = np.asarray(surf.iv(K_plus, float(T + eps)), dtype=np.float64)

        jm = (
            float(np.max(np.abs(iv_node - iv_minus)))
            if iv_minus is not None
            else float("nan")
        )
        jp = (
            float(np.max(np.abs(iv_node - iv_plus)))
            if iv_plus is not None
            else float("nan")
        )
        node_rows.append((f"{T:>5.2f}", jm, jp, float(np.max(iv_node))))

    # --- w_T seam across expiry boundaries ---
    seam_rows: list[tuple[str, float, float, float]] = []
    for T_knot in expiries[1:-1].tolist():
        y_pts = [float(y) for y in (-0.30, -0.15, 0.15, 0.30)]
        diffs = []
        for y in y_pts:
            wT_left = _finite_diff_wT(surf, y=y, T=float(T_knot) - 2 * dt, dt=dt)
            wT_right = _finite_diff_wT(surf, y=y, T=float(T_knot) + dt, dt=dt)
            diffs.append(abs(wT_right - wT_left))
        seam_rows.append(
            (
                f"{T_knot:>5.2f}",
                float(max(diffs)),
                float(np.mean(diffs)),
                float(min(diffs)),
            )
        )

    # --- sigma_hi clipping estimate ---
    clip_rows: list[tuple[str, int, float]] = []
    mids = 0.5 * (expiries[:-1] + expiries[1:])
    for Tm in mids.tolist():
        Fm = float(forward(float(Tm)))
        K_wide = (np.float64(Fm) * np.exp(y_grid)).astype(np.float64)
        iv_mid = np.asarray(surf.iv(K_wide, float(Tm)), dtype=np.float64)
        at_cap = int(np.sum(iv_mid >= sigma_hi - 1e-10))
        clip_rows.append((f"{Tm:>5.2f}", at_cap, float(np.max(iv_mid))))

    def _fmt(x: float) -> str:
        return "   n/a" if not np.isfinite(x) else f"{x:7.3e}"

    print("\n[from_svi diagnosis report: capstone2 synthetic dataset]")
    print("\nTheta (ATM total variance) monotonicity + isotonic impact")
    print(
        f"  max|theta_reg - theta_raw|   : {float(np.max(np.abs(theta_reg - theta_raw))):.3e}"
    )
    print(
        f"  max|theta_interp(nodes)-raw| : {float(np.max(np.abs(theta_interp_at_nodes - theta_raw))):.3e}"
    )
    print(
        f"  theta_raw monotone?          : {bool(np.all(np.diff(theta_raw) >= -1e-14))}"
    )

    print(
        "\nNode snap / near-node IV differences (max over y in [-0.30,-0.15,0,0.15,0.30])"
    )
    print("   T     |IV(T)-IV(T-eps)|  |IV(T)-IV(T+eps)|   max_IV(T)")
    print("---------+-----------------+-----------------+------------")
    for T, jm, jp, mx in node_rows:
        print(f" {T}   {_fmt(jm)}       {_fmt(jp)}      {mx:10.6f}")

    print("\nTime-derivative seam estimate at expiry boundaries (|wT_right - wT_left|)")
    print("  knotT   max_diff    mean_diff   min_diff")
    print("--------  --------    ---------   --------")
    for T, mx, mean, mn in seam_rows:
        print(f" {T}   {mx:8.3e}   {mean:9.3e}  {mn:8.3e}")

    print(
        "\nPotential sigma_hi clipping (count of IV >= 5.0 - tol on mid-interval slices)"
    )
    print("  midT   at_cap_count   max_IV")
    print("------  ------------   ------")
    for T, cnt, mx in clip_rows:
        print(f" {T}      {cnt:6d}     {mx:7.4f}")

    # Sanity: finiteness.
    assert np.isfinite(theta_raw).all()
    assert np.isfinite(theta_reg).all()
    assert np.isfinite(theta_interp_at_nodes).all()
    assert np.isfinite(np.asarray(surf.w(np.asarray([0.0]), float(expiries[0])))[0])
