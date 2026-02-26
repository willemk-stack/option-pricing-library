# Capstone 2 — Local Volatility + PDE Pricing

This capstone turns a volatility surface into a **validated numerical pricing pipeline**:

**surface quotes → implied surface → local vol diagnostics → PDE pricing → convergence / repricing checks**

It is the main numerics-focused project in the repo and is designed to show three things at once:

1. **Modeling depth** — local volatility derived from an implied-vol surface.
2. **Numerical engineering** — finite-difference PDE pricing with stability controls.
3. **Validation discipline** — convergence checks, failure-mode diagnostics, and baseline comparisons.

---

## Why this capstone matters

Many student projects stop after fitting an implied-vol smile or building a pricing function. This capstone goes further:

- it treats **derivative stability** as a first-class problem,
- it exposes **diagnostics instead of hiding them**,
- it validates the PDE stack against known baselines,
- and it shows an end-to-end pricing workflow rather than a disconnected collection of modules.

That is the difference between “I implemented a formula” and “I built a numerical pricing pipeline.”

---

## The pipeline

### 1. Build an implied-vol surface

Start from either:

- a grid-based `VolSurface`, or
- an SVI-based `VolSurface` when derivative information is needed.

For the local-vol workflow, the **SVI route is the intended one**, because Dupire-style local-vol extraction depends on stable total-variance derivatives.

### 2. Derive local volatility

The local-vol bridge constructs a `LocalVolSurface` from the implied surface and exposes:

- local variance / local vol queries,
- invalid masks,
- denominator and stability diagnostics,
- pointwise reports explaining unreliable regions.

This is important because local volatility is not just a formula application. It is highly sensitive to interpolation and numerical differentiation choices.

### 3. Price under a PDE

Once a `LocalVolSurface` is available, the PDE pricers can price:

- European vanilla options under Black–Scholes,
- European vanilla options under local vol,
- digital options through lower-level PDE pathways.

The PDE stack supports the main controls you would expect in a serious numerical project:

- configurable domain policies,
- clustered grids,
- Crank–Nicolson-style workflows,
- Rannacher-style remedies for discontinuous payoffs,
- explicit grid-resolution controls (`Nx`, `Nt`).

### 4. Validate the result

This capstone is not complete unless the result is checked.

The intended validation story is:

- **constant-vol recovery** for local vol,
- **PDE vs analytic Black–Scholes** in regimes where closed form is available,
- **digital payoff convergence** under grid refinement,
- **repricing consistency** from surface → local vol → PDE,
- **diagnostics-first handling** of unstable regions.

---

## What to run first

If you want the shortest path to the full story, start here:

### Main showcase notebook

- `demos/07_vol_surfaces_localvol_pde.ipynb`

This is the best current end-to-end capstone demo. It already presents the strongest narrative:

- synthetic quotes,
- raw grid vs SVI surfaces,
- repair-aware surface construction,
- no-arbitrage diagnostics,
- local-vol diagnostics,
- PDE convergence / stability checks,
- repricing checks.

### Supporting PDE notebook

- `demos/06_pde_pricing_and_diagnostics.ipynb`

Use this when you want the PDE side explained more directly before returning to the full local-vol pipeline.

---

## Recommended talk track

If you are presenting this in an interview or portfolio review, the cleanest sequence is:

### A. Start with the problem

> Local volatility is attractive because it matches the implied surface, but it is numerically fragile. PDE pricing is flexible, but only if the solver and the surface are stable.

### B. Show the engineering judgment

Emphasize that the project does **not** treat local vol as a black box. Instead it explicitly handles:

- derivative sensitivity,
- unstable denominators,
- invalid regions,
- repair-aware smile construction,
- convergence and boundary effects.

### C. Show the proof

Then point to the concrete evidence:

- constant-vol sanity checks,
- PDE-vs-analytic baselines,
- digital convergence tests,
- local-vol repricing diagnostics,
- notebook plots that expose failure modes.

### D. End with limits

A strong ending is:

> This is a research-grade / portfolio-grade local-vol and PDE workflow. The next step is to harden interpolation and move toward real-data calibration.

That makes you sound credible rather than overclaiming.

---

## Validation map

A good flagship page should not just say “tested”; it should show where the evidence lives.

### Surface / local-vol validation

Relevant tests include:

- `tests/test_dupire_constant_vol.py`
- `tests/test_surface_svi_and_localvol.py`
- `tests/test_localvol_pde_vanilla_vs_bs.py`
- `tests/test_localvol_digital_vs_bs.py`
- `tests/test_localvol_digital_vs_quantlib.py`
- `tests/test_localvol_digital_convergence_sweep.py`
- `tests/test_localvol_digital_worst_cases.py`

### PDE / discontinuous-payoff validation

Relevant tests include:

- `tests/test_pde_pricer.py`
- `tests/test_convergence_remedies_digital.py`
- `tests/test_fd_diff.py`
- `tests/test_fd_stencils.py`
- `tests/test_tridiag_solvers.py`

### Diagnostics / surface-reporting support

Relevant tests include:

- `tests/test_diagnostics_vol_surface_report.py`
- `tests/test_svi.py`
- `tests/test_svi_repair.py`

---

## Minimal code path

The core user-facing idea is intentionally compact:

```python
from option_pricing.vol.surface_core import VolSurface
from option_pricing.vol.local_vol_surface import LocalVolSurface
from option_pricing.pricers.pde_pricer import local_vol_price_pde_european

surface_svi = VolSurface.from_svi(rows, forward=forward)
localvol = LocalVolSurface.from_implied(surface_svi, forward=forward, discount=df)

price = local_vol_price_pde_european(
    p,
    lv=localvol,
    Nx=201,
    Nt=201,
)
```

The key point is that the simple API sits on top of a much richer diagnostics and validation layer.

---

## Why this stands out

This capstone is strongest when it is presented as **numerical judgment**, not just feature count.

What makes it different from an ordinary student repo:

- it acknowledges that **Dupire is fragile**,
- it treats **diagnostics as outputs**,
- it includes **grid-convergence and discontinuity remedies**,
- it connects **surface modeling** to **actual pricing**,
- and it gives a believable path toward more advanced calibration work later.

That combination is exactly what makes the project interview-worthy.

---

## Known limits

To keep the presentation honest, say these explicitly:

- local-vol extraction is highly interpolation-sensitive,
- the bridge from implied surface to local vol should be treated as evolving,
- synthetic-data workflows are excellent for validation but are not a substitute for a full real-market calibration pipeline,
- the next major step is real-data surface handling or a stronger calibration model such as Heston.

---

## Suggested README placement

This page works best if the README links to it from a capstone section like:

- **Capstone 1:** Implied vol + surface + no-arb + SVI
- **Capstone 2:** Local vol + PDE pricing + convergence + diagnostics

and then links directly to:

- this page,
- `demos/07_vol_surfaces_localvol_pde.ipynb`,
- the most convincing validation tests.

---

## One-sentence portfolio summary

> A portfolio-grade local-vol and finite-difference PDE capstone that connects implied-surface modeling, numerical diagnostics, and validated pricing in one reproducible workflow.
