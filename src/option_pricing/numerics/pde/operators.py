from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .boundary import DirichletBC


@dataclass(frozen=True, slots=True)
class PDEProblem:
    a_fn: Callable[[float, float], float]
    b_fn: Callable[[float, float], float]
    c_fn: Callable[[float, float], float]
    bc: DirichletBC


class OperatorBuilder1D:
    """
    Docstring for OperatorBuilder1D
    Builds the tridiagonal matrix operator based on:
    u_τ = Lu = a(x,τ)u_{xx} + b(x,τ)u_x + c(x,τ)u + d(x,τ)
    """

    def __init__(self, x: np.ndarray):
        self.x = x

    # def build_L(
    #     self, t: float, *, a_fn, b_fn, c_fn
    # ) -> tuple[Tridiag, BoundaryCoupling]:
    #     """
    #     Returns L as tridiag on interior nodes: i=1..N-2
    #     """
    #     ...
    #     return  Tridiag, BoundaryCoupling
