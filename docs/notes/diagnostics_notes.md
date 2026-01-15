# Diagnostics design engineering notes

This will be a small type up to be used as fallback when engineering / modifying diagnostics modules. Here I clarify and defend design choices which will ultimately shape the user guide.

## Requirements

Diagnostics modules need to enable the user to discern between multiple issues that can arise. Some common ones when comparing methods to bs are:

- Did I pose the same mathematical problem?
    - Same PDE, same payoff function (evaluated at the exact point?), discounting, dididends, time convention ect ect.
- Did I evaluate the PDE solution at the same points as BS?
    - Does my inter- or extrapolation impact the solution. Different mappings can cause discrepancies
- Are conflicting results a consequence of loose numerics or model related?
    - Domain truncation, boundary condition handling/discretization, timestepping.

To tackle this, we split the diagnostics into multiple standalone modules that each adresses one of the points above. This way if we can focus the boundary handling, run diagnostics without having to worry about simultaneosly messing with the PDE.

Some must haves diagnostics msut adress for PDE interal stuff:

1. Domain truncation: BS formula assumes $S\in(0,\infty)$; PDE uses $[S_{min},S_{max}]$. Too tight $\rightarrow$ error.

2. Boundary conditions: wrong asymptotic form (especially for calls at high S).

3. Coordinate transform: solving in $x=\log S$ but extracting/interpreting as S.

4. Extraction/interpolation: spot not on grid \(\leftarrow\) interpolation error mistaken for PDE error.

5. Time direction: maturity vs time-to-maturity conventions; sign mistakes in coefficients.

6. Discount/dividend conventions: \(r\) vs \(r−q\) in drift; discounting in terminal/initial conditions

The conceptual product: a decision report, not a framework

## Example diagnostics result usage/interp

- “For this contract set, PDE with (scheme, $N_x$, $N_t$, domain policy) achieves error $\epsilon$ at runtime T.”

- “BS achieves zero model error at negligible runtime.”

- “Therefore use BS for vanilla European options; retain PDE only if you need X (barriers, local vol, early exercise, state-dependent coefficients, etc.).”

To answer we need:

- convergence (error decreases at expected rate),

- sensitivity (error dominated by domain/BC vs grid vs time),

- sanity checks (PDE reproduces BS on the grid, not just at spot).

## Example layout

```pgsql
option_pricing/diagnostics/pde_vs_bs/
  __init__.py
  compute.py          # public entrypoints (decision + convergence + case runner)
  tables.py           # DataFrame shaping / legacy column aliases (optional but consistent)
  plots.py            # optional quick plots (error vs runtime, convergence curves)
  cases.py            # optional curated cases (like mc_vs_bs)

  modules/
    __init__.py
    map.py            # “BS PDE wiring”: coefficients + payoff + BCs + coord transforms
    domain.py         # domain/grid policy => GridConfig (bounds + spacing)
    extract.py        # interpolate PDE solution at spot (and optionally greeks)
    metrics.py        # error + runtime metrics + limitation classification
    solve.py          # thin wrapper around numerics.pde.solve_pde_1d (timed)
```

### To look into
Pareto frontier style output
