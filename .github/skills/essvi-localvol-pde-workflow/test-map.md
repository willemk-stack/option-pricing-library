# Test map for the eSSVI / local-vol / PDE workflow

Use the nearest representative tests first.

## eSSVI / SVI / surface calibration
- `tests/test_essvi_calibrate.py`
- `tests/test_essvi_surface.py`
- `tests/test_essvi_objective.py`
- `tests/test_essvi_mingone_projection.py`
- `tests/test_svi.py`
- `tests/test_svi_repair.py`
- `tests/test_surface_svi_and_localvol.py`
- `tests/test_arbitrage.py`

## Local-vol extraction and diagnostics
- `tests/test_dupire_constant_vol.py`
- `tests/test_surface_svi_and_localvol.py`
- `tests/test_localvol_digital_vs_bs.py`
- `tests/test_localvol_digital_vs_call_strike_derivative.py`
- `tests/test_localvol_digital_vs_quantlib.py`
- `tests/test_localvol_digital_convergence_sweep.py`

## PDE pricing and convergence
- `tests/test_pde_pricer.py`
- `tests/test_localvol_pde_vanilla_vs_bs.py`
- `tests/test_convergence_remedies_digital.py`
- `tests/test_pde_digital_wrappers.py`
- `tests/test_pde_domain_branches.py`
- `tests/test_pricers_finite_diff.py`

## Public API / import boundaries when signatures move
- `tests/test_api_contracts.py`
- `tests/test_api_smoke.py`
- `tests/test_import_boundaries.py`
