import numpy as np
import pytest

from option_pricing.numerics.grids import (
    GridConfig,
    SpacingPolicy,
    build_grid,
    build_time_grid,
    build_x_grid,
)


def test_grid_config_validation_errors():
    with pytest.raises(ValueError):
        GridConfig(Nx=2, Nt=2, x_lb=0.0, x_ub=1.0, T=1.0).validate()

    with pytest.raises(ValueError):
        GridConfig(Nx=3, Nt=1, x_lb=0.0, x_ub=1.0, T=1.0).validate()

    with pytest.raises(ValueError):
        GridConfig(Nx=3, Nt=2, x_lb=1.0, x_ub=1.0, T=1.0).validate()

    with pytest.raises(ValueError):
        GridConfig(Nx=3, Nt=2, x_lb=0.0, x_ub=1.0, T=0.0).validate()

    cfg = GridConfig(
        Nx=3,
        Nt=2,
        x_lb=0.0,
        x_ub=1.0,
        T=1.0,
        spacing=SpacingPolicy.CLUSTERED,
    )
    with pytest.raises(ValueError):
        cfg.validate()

    cfg = GridConfig(
        Nx=3,
        Nt=2,
        x_lb=0.0,
        x_ub=1.0,
        T=1.0,
        spacing=SpacingPolicy.CLUSTERED,
        x_center=2.0,
    )
    with pytest.raises(ValueError):
        cfg.validate()

    cfg = GridConfig(
        Nx=3,
        Nt=2,
        x_lb=0.0,
        x_ub=1.0,
        T=1.0,
        spacing=SpacingPolicy.CLUSTERED,
        x_center=0.5,
        cluster_strength=0.0,
    )
    with pytest.raises(ValueError):
        cfg.validate()


def test_build_grid_uniform_and_clustered():
    uniform = GridConfig(Nx=5, Nt=4, x_lb=0.0, x_ub=1.0, T=1.0)
    x_uniform = build_x_grid(uniform)
    t_uniform = build_time_grid(uniform)
    assert x_uniform.shape == (5,)
    assert t_uniform.shape == (4,)

    clustered = GridConfig(
        Nx=7,
        Nt=3,
        x_lb=0.0,
        x_ub=2.0,
        T=1.0,
        spacing=SpacingPolicy.CLUSTERED,
        x_center=1.0,
        cluster_strength=3.0,
    )
    grid = build_grid(clustered)
    assert grid.x.shape == (7,)
    assert grid.t.shape == (3,)
    assert np.all(np.diff(grid.x) > 0)
