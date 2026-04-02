# Test map for the eSSVI / local-vol / PDE workflow

Use the nearest representative tests first.

## eSSVI / SVI / surface calibration
- `tests/vol/essvi/test_essvi_calibrate.py`
- `tests/vol/essvi/test_essvi_surface.py`
- `tests/vol/essvi/test_essvi_objective.py`
- `tests/vol/essvi/test_essvi_mingone_projection.py`
- `tests/vol/svi/test_svi.py`
- `tests/vol/svi/test_svi_repair.py`
- `tests/vol/surface/test_surface_svi_and_localvol.py`
- `tests/vol/surface/test_arbitrage.py`

## Local-vol extraction and diagnostics
- `tests/vol/localvol/test_dupire_constant_vol.py`
- `tests/vol/surface/test_surface_svi_and_localvol.py`
- `tests/vol/localvol/test_localvol_digital_vs_bs.py`
- `tests/vol/localvol/test_localvol_digital_vs_call_strike_derivative.py`
- `tests/vol/localvol/test_localvol_digital_vs_quantlib.py`
- `tests/vol/localvol/test_localvol_digital_convergence_sweep.py`

## PDE pricing and convergence
- `tests/pricers/test_pde_pricer.py`
- `tests/vol/localvol/test_localvol_pde_vanilla_vs_bs.py`
- `tests/vol/localvol/test_convergence_remedies_digital.py`
- `tests/pricers/test_pde_digital_wrappers.py`
- `tests/numerics/test_pde_domain_branches.py`
- `tests/pricers/test_pricers_finite_diff.py`

## Public API / import boundaries when signatures move
- `tests/api/test_api_contracts.py`
- `tests/api/test_config_validation.py`
- `tests/api/test_import_boundaries.py`
