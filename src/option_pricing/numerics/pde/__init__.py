"""Finite-difference solvers for 1D linear parabolic PDEs.

This subpackage aims to offer a small, reusable "PDE core" that can be used
both inside the option-pricing package and by external users.

Currently supported PDE form (1D):

    u_t = a(x,t) u_xx + b(x,t) u_x + c(x,t) u + d(x,t)

with Dirichlet boundary conditions.
"""

from .boundary import RobinBC, RobinBCSide, dirichlet_side, neumann_side
from .methods import (
    PDEMethod1D,
    ThetaMethod,
    ThetaScheme1D,
    available_methods,
    register_method,
)
from .operators import AdvectionScheme, LinearParabolicPDE1D, build_theta_system_1d
from .solver import PDESolution1D, solve_pde_1d
from .types import CNSystem

__all__ = [
    # Boundary conditions
    "RobinBC",
    "dirichlet_side",
    "neumann_side",
    "RobinBCSide",
    # Problems / operators
    "AdvectionScheme",
    "LinearParabolicPDE1D",
    "build_theta_system_1d",
    # Methods / registry
    "PDEMethod1D",
    "ThetaMethod",
    "ThetaScheme1D",
    "register_method",
    "available_methods",
    # Solver
    "PDESolution1D",
    "solve_pde_1d",
    # Low-level system container
    "CNSystem",
]
