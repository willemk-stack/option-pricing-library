# Roadmap

This repository is organized as a sequence of **capstones**. Each capstone should meet the same quality bar:

- installable package
- docs updated
- unit tests passing
- demo notebooks runnable end-to-end (CI enforced)

A lightweight issue workflow is recommended:

- **Milestones** track capstones (Capstone 1–4)
- **Labels** track type/area/priority
- A single **Project board** (Backlog → Ready → In Progress → Review → Done) is optional

## Now: keep `main` green

- [ ] CI: run demo notebooks with `nbmake` (in addition to unit tests)
- [ ] CI: keep README generation enforced (`python scripts/render_readme.py --check`)
- [ ] Dev extras: ensure `pip install -e ".[dev]"` is sufficient to run all demos
- [ ] Public API: keep `option_pricing/__init__.py` exports stable and minimal

## Capstone 1 — Volatility stack (IV + Surface + No-arb diagnostics)

**Goal:** robust implied vol inversion, `Smile`/`VolSurface` objects, and diagnostics reports.

- [ ] Implied vol solver: edge-case coverage (near-expiry, deep ITM/OTM), clear failure modes
- [ ] Surface query API: semantics for `x = ln(K/F(T))` and total variance `w(T,x)` + tests
- [ ] Diagnostics: strike monotonicity/convexity + calendar total variance checks + documentation
- [ ] Demo 05: diagnostics summary table + plots; runs on a clean environment

**Exit criteria:** tests + mypy + docs + demo notebooks all pass in CI.

## Capstone 2 — Local vol + PDE

**Goal:** local vol extraction (Dupire-style) + a finite-difference PDE solver and a convergence story.

- [ ] Local vol extraction with smoothing/regularization + guardrails for bad regions
- [ ] PDE solver scaffold (e.g., Crank–Nicolson) with documented boundary conditions
- [ ] Validation: constant-vol regime reproduces BS; grid refinement / convergence plots
- [ ] Notebook: local vol build + PDE pricing + convergence diagnostics

## Capstone 3 — Stochastic vol + calibration (Heston)

**Goal:** a Heston pricer (semi-analytic) + calibration pipeline and diagnostics.

- [ ] Heston characteristic-function pricer
- [ ] Monte Carlo cross-check
- [ ] Calibration objective, constraints, and stability checks
- [ ] Notebook: surface fit quality + parameter stability

## Capstone 4 — Hedging realism (P&L under misspecification)

**Goal:** hedging simulator + experiments showing model risk in hedge P&L.

- [ ] Delta/vega hedging simulator (transaction costs optional)
- [ ] Model misspecification experiments (hedge under model A, simulate under model B)
- [ ] P&L attribution (delta/theta/vega + residual) and reporting notebook

## Suggested GitHub labels

Create a small, consistent label set:

- Capstones: `capstone:1`, `capstone:2`, `capstone:3`, `capstone:4`
- Type: `type:bug`, `type:enhancement`, `type:docs`, `type:chore`
- Area: `area:pricing`, `area:vol`, `area:diagnostics`, `area:ci`, `area:docs`
- Priority: `priority:p0`, `priority:p1`, `priority:p2`

## Release tags (optional)

A simple set of version gates:

- `v0.1.x`: BS/MC/tree baseline
- `v0.2.x`: implied vol solver hardened
- `v0.3.x`: vol surface + diagnostics
- `v0.4.x`: local vol + PDE
- `v0.5.x`: Heston + calibration
- `v1.0.0`: stable API + polished docs + end-to-end demos


## Suggested initial backlog (next 10 issues)

1. **CI: execute demo notebooks with nbmake** (`capstone:1`, `area:ci`, `priority:p0`, `type:chore`)
2. **CI: ensure notebook runs headless (MPLBACKEND=Agg)** (`area:ci`, `priority:p0`)
3. **Docs: add Roadmap to MkDocs nav** (`area:docs`, `type:docs`, `priority:p1`)
4. **IV solver: near-expiry + deep ITM/OTM tests** (`capstone:1`, `area:vol`, `priority:p0`)
5. **IV solver: standardize exceptions + messages** (`area:vol`, `priority:p1`)
6. **VolSurface: define + test x-grid overlap behavior** (`capstone:1`, `area:vol`, `priority:p1`)
7. **Diagnostics: interpretation guide + tolerances** (`area:diagnostics`, `type:docs`, `priority:p1`)
8. **Demo 05: diagnostics summary table + flagged points plot** (`capstone:1`, `area:diagnostics`, `priority:p1`)
9. **Packaging: keep dev extras aligned with demos** (`type:chore`, `priority:p1`)
10. **Release: tag v0.3.x (Capstone 1 complete)** (`type:chore`, `priority:p2`)
