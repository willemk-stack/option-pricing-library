from dataclasses import dataclass

from ..tridiag import BoundaryCoupling, Tridiag


@dataclass(frozen=True, slots=True)
class CNSystem:
    A: Tridiag  # interior system matrix for u^{n+1}
    B: Tridiag  # interior matrix applied to u^{n}
    bc: BoundaryCoupling  # how boundaries couple into RHS
