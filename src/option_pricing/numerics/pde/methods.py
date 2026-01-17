"""Time-stepping methods and a small registry.

This module provides two things:

1) A lightweight *method interface* (:class:`PDEMethod1D`) so the high-level
   solver can call different time-steppers in a uniform way.
2) A string-to-method *registry* so users can do ``method="cn"`` (or register
   their own methods) without editing :func:`~option_pricing.numerics.pde.solve_pde_1d`.

For now the only built-in family is the theta-scheme (explicit / CN / implicit),
but the registry is intended to make it easy to add ADI, splitting, Rannacher,
etc. later.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..grids import Grid
from .operators import AdvectionScheme, LinearParabolicPDE1D, build_theta_system_1d
from .time_steppers import crank_nicolson_linear_step_robin


@runtime_checkable
class PDEMethod1D(Protocol):
    """A 1D PDE time-stepping method.

    A method is called once per time step.
    """

    @property
    def name(self) -> str:  # pragma: no cover
        ...

    def step(
        self,
        *,
        problem: LinearParabolicPDE1D,
        grid: Grid,
        u_n: NDArray[np.floating],
        t_n: float,
        t_np1: float,
        advection: AdvectionScheme,
        solve_tridiag: Callable[
            [
                NDArray[np.floating],
                NDArray[np.floating],
                NDArray[np.floating],
                NDArray[np.floating],
            ],
            NDArray[np.floating],
        ],
    ) -> NDArray[np.floating]:  # pragma: no cover
        ...


@dataclass(frozen=True, slots=True)
class ThetaMethod:
    """User-facing representation of the theta-scheme.

    Common choices:
    - theta=0.0: explicit Euler
    - theta=0.5: Crank-Nicolson
    - theta=1.0: implicit Euler
    """

    theta: float

    @property
    def name(self) -> str:
        if self.theta == 0.5:
            return "cn"
        if self.theta == 0.0:
            return "explicit"
        if self.theta == 1.0:
            return "implicit"
        return f"theta={self.theta:g}"


@dataclass(frozen=True, slots=True)
class ThetaScheme1D:
    """Theta-scheme method implementation (covers explicit/CN/implicit)."""

    theta: float

    @property
    def name(self) -> str:
        return ThetaMethod(theta=float(self.theta)).name

    def step(
        self,
        *,
        problem: LinearParabolicPDE1D,
        grid: Grid,
        u_n: NDArray[np.floating],
        t_n: float,
        t_np1: float,
        advection: AdvectionScheme,
        solve_tridiag: Callable[
            [
                NDArray[np.floating],
                NDArray[np.floating],
                NDArray[np.floating],
                NDArray[np.floating],
            ],
            NDArray[np.floating],
        ],
    ) -> NDArray[np.floating]:
        theta = float(self.theta)
        if not (0.0 <= theta <= 1.0):
            raise ValueError("theta must be in [0, 1]")

        system, rhs_extra = build_theta_system_1d(
            problem=problem,
            grid=grid,
            t_n=float(t_n),
            t_np1=float(t_np1),
            theta=theta,
            advection=advection,
        )

        return crank_nicolson_linear_step_robin(
            grid=grid,
            u_n=u_n,
            t_n=float(t_n),
            t_np1=float(t_np1),
            A=system.A,
            B=system.B,
            bc=problem.bc,  # <-- pass the whole BC object (DirichletBC or RobinBC)
            rhs_extra=rhs_extra,
            solve_tridiag=solve_tridiag,
        )


# -----------------------------
# Registry
# -----------------------------

MethodFactory = Callable[[], PDEMethod1D]
_METHOD_REGISTRY: dict[str, MethodFactory] = {}


def register_method(
    name: str,
    factory: MethodFactory,
    *,
    overwrite: bool = False,
    aliases: tuple[str, ...] = (),
) -> None:
    """Register a method factory under one or more names.

    Parameters
    ----------
    name:
        Primary key users will pass to ``solve_pde_1d(..., method=...)``.
    factory:
        Callable returning a new method instance.
    overwrite:
        If False (default), raising if ``name`` or any alias already exists.
    aliases:
        Additional strings that should resolve to the same factory.
    """

    keys = (name, *aliases)
    for k in keys:
        kk = str(k).lower().strip()
        if not kk:
            raise ValueError("Method name/alias cannot be empty")
        if (not overwrite) and (kk in _METHOD_REGISTRY):
            raise KeyError(f"Method '{kk}' is already registered")
        _METHOD_REGISTRY[kk] = factory


def available_methods() -> list[str]:
    """Return the currently registered method keys (sorted)."""

    return sorted(_METHOD_REGISTRY.keys())


def resolve_method(
    *,
    method: str | ThetaMethod | PDEMethod1D | None,
    theta: float | None,
) -> PDEMethod1D:
    """Resolve the user's method choice into a concrete :class:`PDEMethod1D`.

    Resolution order:
    1) If ``theta`` is provided -> :class:`ThetaScheme1D(theta)`.
    2) If ``method`` is already a :class:`PDEMethod1D` instance -> return it.
    3) If ``method`` is a :class:`ThetaMethod` -> :class:`ThetaScheme1D(method.theta)`.
    4) If ``method`` is a string -> look up in the registry.
    5) If ``method`` is None -> default to "cn".
    """

    if theta is not None:
        return ThetaScheme1D(theta=float(theta))

    if method is None:
        method = "cn"

    if isinstance(method, ThetaMethod):
        return ThetaScheme1D(theta=float(method.theta))

    # If a user passes a concrete method object, trust it.
    if isinstance(method, PDEMethod1D):
        return method

    key = str(method).lower().strip()
    try:
        factory = _METHOD_REGISTRY[key]
    except KeyError as e:
        raise ValueError(
            f"Unknown method '{method}'. Available: {', '.join(available_methods())}"
        ) from e
    return factory()


def _register_builtin_methods() -> None:
    # CN / implicit / explicit via theta-scheme
    register_method(
        "cn",
        lambda: ThetaScheme1D(theta=0.5),
        overwrite=True,
        aliases=("crank-nicolson", "crank_nicolson", "crank", "cn-θ=0.5"),
    )
    register_method(
        "implicit",
        lambda: ThetaScheme1D(theta=1.0),
        overwrite=True,
        aliases=("backward-euler", "backward", "be", "implicit-euler"),
    )
    register_method(
        "explicit",
        lambda: ThetaScheme1D(theta=0.0),
        overwrite=True,
        aliases=("forward-euler", "forward", "fe", "explicit-euler"),
    )


_register_builtin_methods()
