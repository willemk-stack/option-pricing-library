from __future__ import annotations

import numpy as np

from option_pricing.vol import LocalVolSurface
from option_pricing.vol.ssvi import ESSVINodalSurface, ESSVINodeSet, project_essvi_nodes


def _sample_nodes() -> ESSVINodeSet:
    return ESSVINodeSet(
        expiries=np.array([0.25, 0.5, 1.0, 1.5], dtype=np.float64),
        theta=np.array([0.035, 0.048, 0.071, 0.092], dtype=np.float64),
        psi=np.array([0.16, 0.18, 0.205, 0.225], dtype=np.float64),
        rho=np.array([-0.20, -0.12, -0.04, 0.05], dtype=np.float64),
    )


def test_projected_surface_supports_dupire_local_vol_queries() -> None:
    nodes = _sample_nodes()
    projection = project_essvi_nodes(nodes)

    assert projection.success
    assert projection.surface is not None
    assert projection.diag.dupire_invalid_count == 0

    lv = LocalVolSurface.from_implied(
        projection.surface,
        forward=lambda _T: 100.0,
    )
    out = lv.local_var(np.array([85.0, 100.0, 115.0], dtype=np.float64), 0.9)

    assert out.shape == (3,)
    assert np.all(np.isfinite(out))


def test_exact_nodal_surface_remains_available_as_fallback() -> None:
    nodes = _sample_nodes()
    projection = project_essvi_nodes(nodes)
    fallback = projection.fallback_surface

    assert isinstance(fallback, ESSVINodalSurface)
    assert np.all(
        np.isfinite(fallback.w(np.array([-0.3, 0.0, 0.3], dtype=np.float64), 0.8))
    )


def test_projected_surface_matches_nodal_short_end_extension() -> None:
    nodes = _sample_nodes()
    projection = project_essvi_nodes(nodes)

    assert projection.success
    assert projection.surface is not None

    T_short = 0.10
    y = np.array([-0.2, 0.0, 0.2], dtype=np.float64)
    smooth_w = projection.surface.w(y, T_short)
    nodal_w = projection.fallback_surface.w(y, T_short)

    assert np.allclose(smooth_w, nodal_w, rtol=1.0e-8, atol=1.0e-10)

    lv = LocalVolSurface.from_implied(
        projection.surface,
        forward=lambda _T: 100.0,
    )
    sigma = lv.local_vol(np.array([100.0], dtype=np.float64), T_short)

    assert np.all(np.isfinite(sigma))
    assert float(sigma[0]) > 0.0
