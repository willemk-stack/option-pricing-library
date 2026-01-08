# EXAMPLE STRUCTURE
# numerics/pde/
#   __init__.py      # from .types import CNSystem
#   types.py         # CNSystem (+ maybe PDEConfig-ish dataclasses later)
#   operators.py     # builds CNSystem from coefficients + grid + bc
#   steppers.py      # uses CNSystem + u^n -> u^{n+1}
#   boundary.py      # boundary condition helpers, BoundaryCoupling builders
