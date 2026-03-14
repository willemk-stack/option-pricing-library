from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ...market.curves import PricingContext
from ...types import MarketData
from ...typing import ArrayLike
from ..arbitrage import CalendarVarianceReport, SurfaceNoArbReport, check_surface_noarb
from ..surface_core import VolSurface
from .mingone import compute_Apsi_Cpsi, compute_p_sequence
from .models import ESSVINodeSet, ESSVITermStructures
from .surface import ESSVISmileSlice


def _as_float_vector(name: str, value: ArrayLike) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
    return arr


def _to_ctx(market: MarketData | PricingContext) -> PricingContext:
    if isinstance(market, PricingContext):
        return market
    return market.to_context()


def _validation_expiry_grid(expiries: np.ndarray) -> np.ndarray:
    expiries = np.unique(np.asarray(expiries, dtype=np.float64))
    if expiries.size == 0:
        raise ValueError("expiries must not be empty.")
    if np.any(expiries <= 0.0):
        raise ValueError("expiries must be > 0.")
    if expiries.size == 1:
        return expiries
    midpoints = 0.5 * (expiries[:-1] + expiries[1:])
    return np.unique(np.concatenate([expiries, midpoints])).astype(np.float64)


@dataclass(frozen=True, slots=True)
class ESSVIConstraintReport:
    ok: bool
    T: np.ndarray
    theta: np.ndarray
    psi: np.ndarray
    eta: np.ndarray
    theta_margin: np.ndarray
    psi_margin: np.ndarray
    rho_margin: np.ndarray
    lee_margin: np.ndarray
    gj_butterfly_margin: np.ndarray
    bad_indices: np.ndarray
    violations: tuple[str, ...]
    message: str

    @property
    def calendar_margin(self) -> np.ndarray:
        warnings.warn(
            "ESSVIConstraintReport.calendar_margin is deprecated; use "
            "gj_butterfly_margin instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return np.asarray(self.gj_butterfly_margin, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ESSVINodeConstraintReport:
    ok: bool
    expiries: np.ndarray
    theta: np.ndarray
    psi: np.ndarray
    rho: np.ndarray
    eta: np.ndarray
    p: np.ndarray
    theta_margin: np.ndarray
    psi_margin: np.ndarray
    rho_margin: np.ndarray
    lee_margin: np.ndarray
    gj_butterfly_margin: np.ndarray
    theta_increase_margin: np.ndarray
    psi_lower_margin: np.ndarray
    psi_upper_margin: np.ndarray
    g_plus_margin: np.ndarray
    g_minus_margin: np.ndarray
    bad_node_indices: np.ndarray
    bad_pair_indices: np.ndarray
    violations: tuple[str, ...]
    message: str


@dataclass(frozen=True, slots=True)
class ESSVIValidationReport:
    ok: bool
    expiries: np.ndarray
    y_grid: np.ndarray
    constraints: ESSVIConstraintReport
    static_noarb: SurfaceNoArbReport
    message: str


def evaluate_essvi_constraints(
    params: ESSVITermStructures,
    expiries: ArrayLike,
    *,
    tol: float = 1e-10,
) -> ESSVIConstraintReport:
    T = _as_float_vector("expiries", expiries)
    try:
        theta = np.asarray(params.theta(T), dtype=np.float64)
        psi = np.asarray(params.psi(T), dtype=np.float64)
        eta = np.asarray(params.eta(T), dtype=np.float64)
    except Exception as exc:
        bad = np.arange(T.size, dtype=np.int64)
        nan = np.full_like(T, np.nan, dtype=np.float64)
        return ESSVIConstraintReport(
            ok=False,
            T=T,
            theta=nan,
            psi=nan,
            eta=nan,
            theta_margin=nan,
            psi_margin=nan,
            rho_margin=nan,
            lee_margin=nan,
            gj_butterfly_margin=nan,
            bad_indices=bad,
            violations=("evaluation_error",),
            message=f"Constraint evaluation failed: {exc}",
        )

    abs_eta = np.abs(eta)
    theta_margin = theta
    psi_margin = psi
    rho_margin = psi - abs_eta
    lee_margin = (4.0 - float(tol)) - (psi + abs_eta)
    gj_butterfly_margin = (4.0 * theta - float(tol)) - psi * (psi + abs_eta)

    masks = {
        "theta_positive": (~np.isfinite(theta)) | (theta_margin <= 0.0),
        "psi_positive": (~np.isfinite(psi)) | (psi_margin <= 0.0),
        "rho_constraint": (~np.isfinite(eta)) | (rho_margin <= 0.0),
        "lee_constraint": lee_margin <= 0.0,
        "gj_butterfly_constraint": gj_butterfly_margin < 0.0,
    }

    bad_mask = np.zeros_like(T, dtype=bool)
    violations: list[str] = []
    for name, mask in masks.items():
        mask_arr = np.asarray(mask, dtype=bool)
        if np.any(mask_arr):
            violations.append(name)
            bad_mask |= mask_arr

    bad_indices = np.flatnonzero(bad_mask).astype(np.int64)
    ok = bad_indices.size == 0
    message = "OK" if ok else f"Violations found: {', '.join(violations)}."

    return ESSVIConstraintReport(
        ok=ok,
        T=T,
        theta=theta,
        psi=psi,
        eta=eta,
        theta_margin=theta_margin,
        psi_margin=psi_margin,
        rho_margin=rho_margin,
        lee_margin=lee_margin,
        gj_butterfly_margin=gj_butterfly_margin,
        bad_indices=bad_indices,
        violations=tuple(violations),
        message=message,
    )


def validate_essvi_nodes(
    nodes: ESSVINodeSet,
    *,
    strict: bool = False,
    tol: float = 1e-10,
) -> ESSVINodeConstraintReport:
    theta = np.asarray(nodes.theta, dtype=np.float64)
    psi = np.asarray(nodes.psi, dtype=np.float64)
    rho = np.asarray(nodes.rho, dtype=np.float64)
    eta = np.asarray(nodes.eta, dtype=np.float64)

    p = compute_p_sequence(rho)
    A, C = compute_Apsi_Cpsi(theta, rho, psi)

    theta_margin = theta
    psi_margin = psi
    rho_margin = 1.0 - np.abs(rho)
    lee_margin = (4.0 - float(tol)) - (psi + np.abs(eta))
    gj_butterfly_margin = (4.0 * theta - float(tol)) - psi * (psi + np.abs(eta))

    theta_increase_margin = (
        np.diff(theta) if theta.size > 1 else np.empty((0,), dtype=np.float64)
    )
    psi_lower_margin = (
        psi[1:] - A[1:] if psi.size > 1 else np.empty((0,), dtype=np.float64)
    )
    psi_upper_margin = (
        C[1:] - psi[1:] if psi.size > 1 else np.empty((0,), dtype=np.float64)
    )
    g_plus_margin = (
        np.diff(nodes.g_plus) if psi.size > 1 else np.empty((0,), dtype=np.float64)
    )
    g_minus_margin = (
        np.diff(nodes.g_minus) if psi.size > 1 else np.empty((0,), dtype=np.float64)
    )

    node_masks = {
        "theta_positive": theta_margin <= 0.0,
        "psi_positive": psi_margin <= 0.0,
        "rho_constraint": rho_margin <= 0.0,
        "lee_constraint": lee_margin <= 0.0,
        "gj_butterfly_constraint": gj_butterfly_margin < 0.0,
    }
    pair_masks = {
        "theta_increase": theta_increase_margin <= 0.0,
        "psi_lower": psi_lower_margin <= 0.0,
        "psi_upper": psi_upper_margin <= 0.0,
        "g_plus_monotone": g_plus_margin <= 0.0,
        "g_minus_monotone": g_minus_margin <= 0.0,
    }

    bad_node_mask = np.zeros_like(theta, dtype=bool)
    violations: list[str] = []
    for name, mask in node_masks.items():
        mask_arr = np.asarray(mask, dtype=bool)
        if np.any(mask_arr):
            bad_node_mask |= mask_arr
            violations.append(name)

    bad_pair_mask = np.zeros(max(theta.size - 1, 0), dtype=bool)
    for name, mask in pair_masks.items():
        mask_arr = np.asarray(mask, dtype=bool)
        if np.any(mask_arr):
            bad_pair_mask |= mask_arr
            violations.append(name)

    bad_node_indices = np.flatnonzero(bad_node_mask).astype(np.int64)
    bad_pair_indices = np.flatnonzero(bad_pair_mask).astype(np.int64)
    ok = bad_node_indices.size == 0 and bad_pair_indices.size == 0
    message = "OK" if ok else f"Violations found: {', '.join(violations)}."

    report = ESSVINodeConstraintReport(
        ok=ok,
        expiries=np.asarray(nodes.expiries, dtype=np.float64),
        theta=theta,
        psi=psi,
        rho=rho,
        eta=eta,
        p=p,
        theta_margin=theta_margin,
        psi_margin=psi_margin,
        rho_margin=rho_margin,
        lee_margin=lee_margin,
        gj_butterfly_margin=gj_butterfly_margin,
        theta_increase_margin=np.asarray(theta_increase_margin, dtype=np.float64),
        psi_lower_margin=np.asarray(psi_lower_margin, dtype=np.float64),
        psi_upper_margin=np.asarray(psi_upper_margin, dtype=np.float64),
        g_plus_margin=np.asarray(g_plus_margin, dtype=np.float64),
        g_minus_margin=np.asarray(g_minus_margin, dtype=np.float64),
        bad_node_indices=bad_node_indices,
        bad_pair_indices=bad_pair_indices,
        violations=tuple(violations),
        message=message,
    )
    if strict and not report.ok:
        raise ValueError(report.message)
    return report


def validate_essvi_continuous(
    params: ESSVITermStructures,
    market: MarketData | PricingContext,
    *,
    expiries: ArrayLike | None = None,
    y_grid: ArrayLike | None = None,
    strict: bool = False,
    tol: float = 1e-10,
) -> ESSVIValidationReport:
    if expiries is None:
        inferred = getattr(params.theta_term, "sample_expiries", None)
        if inferred is None:
            raise ValueError(
                "expiries must be provided unless theta_term exposes sample_expiries."
            )
        expiries_arr = _validation_expiry_grid(np.asarray(inferred, dtype=np.float64))
    else:
        expiries_arr = _validation_expiry_grid(_as_float_vector("expiries", expiries))

    y_grid_arr = (
        np.linspace(-2.5, 2.5, 81, dtype=np.float64)
        if y_grid is None
        else _as_float_vector("y_grid", y_grid)
    )
    if np.any(np.diff(y_grid_arr) <= 0.0):
        raise ValueError("y_grid must be strictly increasing.")

    constraint_report = evaluate_essvi_constraints(params, expiries_arr, tol=tol)
    ctx = _to_ctx(market)
    try:
        smiles = tuple(
            ESSVISmileSlice(
                T=float(T_i),
                params=params,
                y_min=float(y_grid_arr[0]),
                y_max=float(y_grid_arr[-1]),
            )
            for T_i in expiries_arr
        )
        surface = VolSurface(expiries=expiries_arr, smiles=smiles, forward=ctx.fwd)
        static_noarb = check_surface_noarb(
            surface,
            df=ctx.df,
            calendar_x_grid=y_grid_arr,
        )
    except Exception as exc:
        static_noarb = SurfaceNoArbReport(
            ok=False,
            smile_monotonicity=(),
            smile_convexity=(),
            calendar_total_variance=CalendarVarianceReport(
                performed=False,
                ok=False,
                x_grid=np.asarray([], dtype=np.float64),
                bad_pairs=np.empty((0, 2), dtype=np.int64),
                max_violation=0.0,
                message=f"Surface sampling failed: {exc}",
            ),
            message=f"Surface sampling failed: {exc}",
        )

    ok = bool(constraint_report.ok and static_noarb.ok)
    message = (
        "OK"
        if ok
        else f"constraints={constraint_report.message}, static_noarb={static_noarb.message}"
    )

    report = ESSVIValidationReport(
        ok=ok,
        expiries=expiries_arr,
        y_grid=y_grid_arr,
        constraints=constraint_report,
        static_noarb=static_noarb,
        message=message,
    )
    if strict and not report.ok:
        raise ValueError(report.message)
    return report


def validate_essvi_surface(
    params: ESSVITermStructures,
    market: MarketData | PricingContext,
    *,
    expiries: ArrayLike | None = None,
    y_grid: ArrayLike | None = None,
    strict: bool = False,
    tol: float = 1e-10,
) -> ESSVIValidationReport:
    warnings.warn(
        "validate_essvi_surface is deprecated; use validate_essvi_continuous.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return validate_essvi_continuous(
        params,
        market,
        expiries=expiries,
        y_grid=y_grid,
        strict=strict,
        tol=tol,
    )
