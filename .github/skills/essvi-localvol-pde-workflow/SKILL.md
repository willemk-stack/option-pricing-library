---
name: essvi-localvol-pde-workflow
description: Use when changing eSSVI, SVI surface, no-arbitrage checks, Dupire local volatility, local-vol diagnostics, or PDE pricing. Review impacted invariants, docs, and targeted tests before and after edits.
---
This skill is for the repo's most specialized workflow:

quotes or smiles -> surface construction -> arbitrage diagnostics -> SVI/eSSVI fitting or projection -> local-vol extraction -> PDE pricing -> convergence and repricing checks.

Use this skill when the task mentions any of these terms:
- eSSVI, SSVI, SVI
- no-arb, arbitrage, smile, surface
- Dupire, local vol, local volatility
- PDE, finite difference, convergence, repricing
- diagnostics, invalid masks, denominator failures

## Goal
Make changes safely in the surface/local-vol/PDE stack by:
1. identifying the affected sub-workflow,
2. preserving the right numerical or API invariants,
3. choosing the smallest useful validation set first,
4. updating docs or demos when behavior changes.

## First-pass triage
Before editing, classify the change into one or more buckets:
- **Surface calibration/projection**: SVI/eSSVI parameters, fitting, smoothing, projection, validation
- **Surface diagnostics/no-arbitrage**: no-arb checks, smile/surface construction, surface diagnostics
- **Local-vol extraction**: Dupire inputs, finite-difference derivatives, invalid region handling, diagnostics
- **PDE pricing**: domain config, coefficients, IC remedies, wrappers, convergence behavior
- **Typing/signatures**: callable signatures, data structures, Optional narrowing, public object shapes

Then read:
- `test-map.md`
- `invariants.md`
- the touched production files and their nearest representative tests

## Validation strategy
Prefer targeted validation before broad validation.

### Surface / eSSVI focused
Start with one or more of:
```bash
pytest -q tests/test_essvi_calibrate.py
pytest -q tests/test_essvi_surface.py
pytest -q tests/test_essvi_objective.py
pytest -q tests/test_essvi_mingone_projection.py
pytest -q tests/test_svi.py tests/test_svi_repair.py
pytest -q tests/test_arbitrage.py tests/test_surface_svi_and_localvol.py
```

### Local-vol / Dupire focused
Start with one or more of:
```bash
pytest -q tests/test_dupire_constant_vol.py
pytest -q tests/test_surface_svi_and_localvol.py
pytest -q tests/test_localvol_pde_vanilla_vs_bs.py
pytest -q tests/test_localvol_digital_vs_bs.py
pytest -q tests/test_localvol_digital_vs_call_strike_derivative.py
```

### PDE focused
Start with one or more of:
```bash
pytest -q tests/test_pde_pricer.py
pytest -q tests/test_localvol_pde_vanilla_vs_bs.py
pytest -q tests/test_convergence_remedies_digital.py
pytest -q tests/test_pde_digital_wrappers.py
pytest -q tests/test_pde_domain_branches.py
```

### When the change is broad
Run the broader numerics checks after targeted tests pass:
```bash
pytest -q tests/test_surface_svi_and_localvol.py tests/test_dupire_constant_vol.py tests/test_localvol_pde_vanilla_vs_bs.py tests/test_convergence_remedies_digital.py
```

### When public signatures or typing changed
Also run:
```bash
mypy src/option_pricing
pytest -q tests/test_api_contracts.py tests/test_api_smoke.py tests/test_import_boundaries.py
```

## Decision rules
- Do **not** change tolerances casually. If you change a tolerance, explain what numerical behavior changed and why the new threshold is still meaningful.
- Do **not** treat a passing broad suite as enough if a representative targeted test failed earlier. Explain the failure mode.
- Prefer fixing the numerical root cause over masking it with looser checks.
- If diagnostics semantics change, update tests and docs together so the expected behavior stays clear.
- If the change touches public entry points or documented behavior, hand off to the `public-api-doc-sync` skill after code validation.
- If the change looks performance-sensitive, hand off to the `benchmark-regression-triage` skill after correctness checks.

## Expected output
When you use this skill, end with a short maintainer-style summary:
1. **Area touched**
2. **Invariants checked**
3. **Tests run**
4. **Any tolerance or diagnostics changes**
5. **Docs/demo follow-up needed**

## Examples
- "Use the eSSVI/local-vol/PDE workflow to review a change in nodal projection."
- "Use the eSSVI/local-vol/PDE workflow before changing Dupire local-vol diagnostics."
- "Use the eSSVI/local-vol/PDE workflow to validate a PDE convergence fix."
