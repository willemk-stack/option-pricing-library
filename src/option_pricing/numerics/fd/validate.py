"""
Docstring for option_pricing.numerics.fd.validate
numerics/fd/validate.py (optional)

Responsibility: tiny helpers like:
- assert_strictly_increasing(x, name)
- assert_min_points(x, n, name)



Why: both PDE and Dupire do the same validation (monotonic grid, min sizes). This avoids “slightly different errors” across codepaths.
"""
