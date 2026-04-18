from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import cache

import numpy as np
from scipy.special import roots_legendre

from ..typing import ArrayLike, FloatArray


class PanelSpacing(StrEnum):
    UNIFORM = "uniform"
    CLUSTERED = "clustered"


@dataclass(frozen=True, slots=True)
class QuadratureConfig:
    u_max: float
    n_panels: int
    nodes_per_panel: int
    panel_spacing: PanelSpacing = PanelSpacing.UNIFORM
    cluster_strength: float = 2.0

    def validate(self) -> None:
        if self.u_max <= 0.0:
            raise ValueError("u_max must be > 0")
        if self.n_panels < 1:
            raise ValueError("n_panels must be >= 1")
        if self.nodes_per_panel < 1:
            raise ValueError("nodes_per_panel must be >= 1")
        if (
            self.panel_spacing == PanelSpacing.CLUSTERED
            and self.cluster_strength <= 0.0
        ):
            raise ValueError("cluster_strength must be > 0")


@dataclass(frozen=True, slots=True)
class CompositeRule:
    """Reusable object for rule caching."""

    panel_edges: FloatArray
    u_panel: FloatArray
    omega_panel: FloatArray
    u_flat: FloatArray
    omega_flat: FloatArray


@dataclass(frozen=True, slots=True)
class CompositeIntegrationResult:
    total: float
    panel_contribs: FloatArray


def build_panels(cfg: QuadratureConfig) -> FloatArray:
    cfg.validate()

    if cfg.panel_spacing == PanelSpacing.UNIFORM:
        return np.linspace(0.0, cfg.u_max, cfg.n_panels + 1, dtype=float)

    t = np.linspace(0.0, 1.0, cfg.n_panels + 1, dtype=float)
    b = float(cfg.cluster_strength)
    raw = np.sinh(b * t) / np.sinh(b)
    return np.asarray(cfg.u_max * raw, dtype=float)


def _validate_rule(
    nodes: FloatArray,
    weights: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    x = np.asarray(nodes, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    if x.ndim != 1 or w.ndim != 1:
        raise ValueError("nodes and weights must be 1D")
    if x.shape != w.shape:
        raise ValueError("nodes and weights must have the same shape")
    if x.size == 0:
        raise ValueError("nodes and weights must be non-empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("nodes must be finite")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite")

    return x, w


def gauss_legendre_nodes_weights(n: int) -> tuple[FloatArray, FloatArray]:
    if n < 1:
        raise ValueError("n must be >= 1")
    x, w = roots_legendre(n)
    return np.asarray(x, dtype=np.float64), np.asarray(w, dtype=np.float64)


def map_rule_to_interval(
    nodes: FloatArray,
    weights: FloatArray,
    a: float,
    b: float,
) -> tuple[FloatArray, FloatArray]:
    if not (a < b):
        raise ValueError(f"Require a < b, got [{a}, {b}]")

    x, w = _validate_rule(nodes, weights)

    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)

    u = mid + half * x
    omega = half * w
    return np.asarray(u, dtype=np.float64), np.asarray(omega, dtype=np.float64)


def build_composite_rule(
    cfg: QuadratureConfig,
    nodes: FloatArray,
    weights: FloatArray,
) -> CompositeRule:
    cfg.validate()
    x, w = _validate_rule(nodes, weights)

    if x.size != cfg.nodes_per_panel:
        raise ValueError(
            "Rule size does not match cfg.nodes_per_panel: "
            f"{x.size} != {cfg.nodes_per_panel}"
        )

    panel_edges = build_panels(cfg)
    n_panels = cfg.n_panels
    n_nodes = cfg.nodes_per_panel

    u_panel = np.empty((n_panels, n_nodes), dtype=np.float64)
    omega_panel = np.empty((n_panels, n_nodes), dtype=np.float64)

    for i, (a, b) in enumerate(zip(panel_edges[:-1], panel_edges[1:], strict=False)):
        u_i, omega_i = map_rule_to_interval(x, w, float(a), float(b))
        u_panel[i, :] = u_i
        omega_panel[i, :] = omega_i

    return CompositeRule(
        panel_edges=np.asarray(panel_edges, dtype=np.float64),
        u_panel=u_panel,
        omega_panel=omega_panel,
        u_flat=np.asarray(u_panel.ravel(), dtype=np.float64),
        omega_flat=np.asarray(omega_panel.ravel(), dtype=np.float64),
    )


@cache
def build_gauss_legendre_rule(cfg: QuadratureConfig) -> CompositeRule:
    cfg.validate()
    nodes, weights = gauss_legendre_nodes_weights(cfg.nodes_per_panel)
    return build_composite_rule(cfg, nodes, weights)


def integrate_composite_rule(
    eval_fn: Callable[[ArrayLike], ArrayLike],
    rule: CompositeRule,
) -> CompositeIntegrationResult:
    values_panel = np.asarray(eval_fn(rule.u_panel), dtype=np.float64)

    if values_panel.shape != rule.u_panel.shape:
        raise ValueError(
            "eval_fn must return an array with the same shape as rule.u_panel "
            f"for scalar integration. Got {values_panel.shape} vs {rule.u_panel.shape}."
        )

    panel_contribs = np.sum(rule.omega_panel * values_panel, axis=1)
    total = float(np.sum(panel_contribs))

    return CompositeIntegrationResult(
        total=total,
        panel_contribs=np.asarray(panel_contribs, dtype=np.float64),
    )


def composite_fixed_rule(
    eval_fn: Callable[[ArrayLike], ArrayLike],
    cfg: QuadratureConfig,
    nodes: FloatArray,
    weights: FloatArray,
) -> float:
    rule = build_composite_rule(cfg, nodes, weights)
    result = integrate_composite_rule(eval_fn, rule)
    return result.total


def composite_gauss_legendre(
    eval_fn: Callable[[ArrayLike], ArrayLike],
    cfg: QuadratureConfig,
) -> float:
    rule = build_gauss_legendre_rule(cfg)
    result = integrate_composite_rule(eval_fn, rule)
    return result.total
