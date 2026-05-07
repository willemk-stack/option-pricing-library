"""Internal helpers for notebook-facing Heston diagnostics provenance."""

from __future__ import annotations

from typing import Any

from ...models.heston.fourier import _default_heston_quadrature_config
from ...numerics.quadrature import CompositeRule, QuadratureConfig


def _null_backend_config_details() -> dict[str, Any]:
    return {
        "u_max": None,
        "n_panels": None,
        "nodes_per_panel": None,
        "panel_spacing": None,
        "cluster_strength": None,
    }


def gauss_config_details(quad_cfg: QuadratureConfig | None) -> dict[str, Any]:
    if quad_cfg is None:
        return _null_backend_config_details()

    return {
        "u_max": float(quad_cfg.u_max),
        "n_panels": int(quad_cfg.n_panels),
        "nodes_per_panel": int(quad_cfg.nodes_per_panel),
        "panel_spacing": str(quad_cfg.panel_spacing),
        "cluster_strength": float(quad_cfg.cluster_strength),
    }


def infer_backend_config_resolution(
    *,
    backend: str,
    quad_cfg: QuadratureConfig | None,
    rule: CompositeRule | None,
    config_resolution: str | None = None,
) -> str | None:
    if config_resolution is not None:
        return config_resolution

    if backend != "gauss_legendre":
        return None

    if quad_cfg is not None and rule is not None:
        raise ValueError(
            "Pass either quad_cfg or rule, not both. "
            "If you already built a rule, it is the authoritative discretization."
        )

    if quad_cfg is not None:
        return "explicit_quad_cfg"
    if rule is not None:
        return "explicit_rule"
    return "default_hard_coded"


def backend_config_meta(
    *,
    backend: str,
    quad_cfg: QuadratureConfig | None,
    rule: CompositeRule | None,
    config_resolution: str | None = None,
) -> dict[str, Any]:
    resolution = infer_backend_config_resolution(
        backend=backend,
        quad_cfg=quad_cfg,
        rule=rule,
        config_resolution=config_resolution,
    )

    if backend != "gauss_legendre":
        details = _null_backend_config_details()
    elif rule is not None:
        # A prebuilt rule is authoritative, but the rule object does not keep a
        # round-trippable QuadratureConfig, so only the resolution path is exact.
        details = _null_backend_config_details()
    elif quad_cfg is not None:
        details = gauss_config_details(quad_cfg)
    else:
        details = gauss_config_details(_default_heston_quadrature_config())

    return {
        "backend": backend,
        "config_resolution": resolution,
        **details,
    }


__all__ = [
    "backend_config_meta",
    "gauss_config_details",
    "infer_backend_config_resolution",
]
