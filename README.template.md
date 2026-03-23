# Option Pricing Library

Typed Python library for pricing vanilla options, repairing implied-volatility surfaces, and validating the local-vol/PDE path with published evidence.

[![Tests](https://github.com/willemk-stack/option-pricing-library/actions/workflows/tests.yaml/badge.svg)](https://github.com/willemk-stack/option-pricing-library/actions/workflows/tests.yaml)
[![Codecov](https://codecov.io/gh/willemk-stack/option-pricing-library/branch/main/graph/badge.svg)](https://codecov.io/gh/willemk-stack/option-pricing-library)
[![Docs](https://github.com/willemk-stack/option-pricing-library/actions/workflows/deploy-docs.yml/badge.svg)](https://github.com/willemk-stack/option-pricing-library/actions/workflows/deploy-docs.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)

Package / import name: `option_pricing`

- Prices vanilla options through a typed public API that scales from single trades to surface-aware workflows.
- Makes the hard quant path reviewable: quote repair, smooth Dupire handoff, local-vol extraction, and PDE repricing stay documented and instrumented.
- Publishes benchmarks, proof pages, and generated visuals from committed artifacts rather than screenshot-only claims.

> **Hiring-manager path:** Start with the [Decision guide](https://willemk-stack.github.io/option-pricing-library/user_guides/decision_guide/), then [Performance evidence](https://willemk-stack.github.io/option-pricing-library/performance/), then [Architecture](https://willemk-stack.github.io/option-pricing-library/architecture/).

## What this project demonstrates

- Numerical finance code packaged as a typed library instead of notebook-only snippets.
- Surface repair and eSSVI smoothing with diagnostics that stay visible during review.
- Local-vol and PDE validation backed by repricing, convergence, and no-arbitrage evidence.
- CI-checked docs, generated visuals, and benchmark publishing tied to committed sources.

## Why I built it this way

I built this repo to make the difficult middle of quant engineering inspectable: not only pricing formulas, but the path from noisy implied-vol data to repaired surfaces, smooth local-vol inputs, and validated PDE outputs. The emphasis is on typed interfaces, reproducible artifacts, and proof pages that explain failure modes instead of hiding them.

## Recommended example

Instruments separate what is being priced from how it is priced.

```python
from option_pricing import (
    ExerciseStyle,
    MarketData,
    OptionType,
    VanillaOption,
    binom_price_instrument,
    bs_price_instrument,
    mc_price_instrument,
)

inst = VanillaOption(
    expiry=1.0,
    strike=100.0,
    kind=OptionType.CALL,
    exercise=ExerciseStyle.EUROPEAN,
)

market = MarketData(spot=100.0, rate=0.02, dividend_yield=0.0)
sigma = 0.2

bs_price_instrument(inst, market=market, sigma=sigma)
mc_price_instrument(inst, market=market, sigma=sigma)
binom_price_instrument(inst, market=market, sigma=sigma, n_steps=200)
```

For the flat-input, curves-first, and implied-vol workflows, start with the [Instruments guide](https://willemk-stack.github.io/option-pricing-library/user_guides/instruments/), the [API reference](https://willemk-stack.github.io/option-pricing-library/api/), and the [surface workflow](https://willemk-stack.github.io/option-pricing-library/user_guides/surface_workflow/).

## Proof and evidence

![README proof card summarizing surface repair, smooth Dupire handoff, local-vol and PDE validation, and benchmark plus delivery evidence.](./docs/assets/generated/showcase/readme_proof_card.light.svg#gh-light-mode-only)
![README proof card summarizing surface repair, smooth Dupire handoff, local-vol and PDE validation, and benchmark plus delivery evidence.](./docs/assets/generated/showcase/readme_proof_card.dark.svg#gh-dark-mode-only)

The proof card above is generated from the published eSSVI and local-vol validation pages plus the committed benchmark artifacts, so the README stays aligned with the same source of truth as the performance page.

| Area | What to review | Open |
| --- | --- | --- |
| Surface repair | Quote-vs-repaired surfaces, no-arbitrage checks, and per-expiry SVI residual tables | [Surface workflow](https://willemk-stack.github.io/option-pricing-library/user_guides/surface_workflow/) |
| Smooth Dupire handoff | Published seam diagnostics, smoothed projection, and Dupire invalid-count checks | [eSSVI smooth handoff](https://willemk-stack.github.io/option-pricing-library/user_guides/essvi_smooth_handoff/) |
| Local-vol and PDE validation | Repricing summaries, convergence plots, and local-vol diagnostics from the published sweep | [Local-vol and PDE validation](https://willemk-stack.github.io/option-pricing-library/user_guides/localvol_pde_validation/) |
| Performance evidence | Committed IV scaling, PDE runtime/error, digital-remedy, and stage-budget benchmarks | [Performance evidence](https://willemk-stack.github.io/option-pricing-library/performance/) |

## Go deeper

- [Decision guide](https://willemk-stack.github.io/option-pricing-library/user_guides/decision_guide/) for the strongest end-to-end review path
- [Instruments guide](https://willemk-stack.github.io/option-pricing-library/user_guides/instruments/) for the recommended public API
- [Architecture](https://willemk-stack.github.io/option-pricing-library/architecture/) for package boundaries and dependency direction
- [API reference](https://willemk-stack.github.io/option-pricing-library/api/) for the generated surface area
- [Performance evidence](https://willemk-stack.github.io/option-pricing-library/performance/) for the committed benchmark snapshot

## Installation

Install directly from GitHub:

```bash
pip install "git+https://github.com/willemk-stack/option-pricing-library.git"
```

For a local editable checkout:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .
```

Supported extras from `pyproject.toml`:

- `pip install -e ".[plot]"` for plotting helpers used by diagnostics and docs figures
- `pip install -e ".[notebooks]"` for the demo notebook environment
- `pip install -e ".[dev]"` for tests, benchmarks, linting, formatting, and typing
- `pip install -e ".[docs]"` for MkDocs and API-reference generation

Python requirement:

- **Python 3.12+**

## What is implemented

### Pricing engines

- **Black-Scholes(-Merton)** price and Greeks
- **CRR binomial tree** for European and American vanilla options
- **Monte Carlo under GBM** with optional variance-reduction features
- **Finite-difference PDE pricing** for selected advanced workflows

### Volatility and diagnostics

- **BS implied-volatility inversion** with bracketing-based solvers
- **Smile** and **VolSurface** objects with interpolation support
- **Static no-arbitrage diagnostics** for surfaces
- **SVI fitting and repair** workflows
- **eSSVI calibration, validation, and smooth-surface projection**
- **Local-vol extraction and diagnostics** from differentiable implied surfaces
- **Convergence and repricing validation utilities**

## Project layout

| Layer | Purpose |
| --- | --- |
| **`instruments/`** | Contracts, payoffs, and exercise-style abstractions |
| **`market/`** | Spot, rates, dividends, curves, and pricing contexts |
| **`pricers/`** | Public pricing entry points for analytic, tree, Monte Carlo, and PDE workflows |
| **`models/`** | Model-specific internals such as Black-Scholes and local-vol components |
| **`vol/`** | Implied vol, smiles, surfaces, SVI/eSSVI tooling, and local-vol extraction |
| **`numerics/`** | Root-finding, finite differences, tridiagonal solvers, and PDE building blocks |
| **`diagnostics/`** | Arbitrage checks, convergence studies, repricing audits, and reports |
| **`viz/`** | Plotting helpers for surfaces, diagnostics, and published figures |

## Validation and development

<details>
<summary>Contributor notes</summary>

This file is auto-generated from `README.template.md`.

```bash
python scripts/render_readme.py
```

Refresh the committed visual bundle with:

```bash
python scripts/build_visual_artifacts.py all --profile publish
```

</details>

Development checks:

```bash
ruff check .
black --check .
pytest -q
mypy
```

The repo also includes GitHub Actions for tests and docs, README freshness checks, and CI notebook execution via `nbmake`.

## Future work

See the published Future work page: [docs/roadmap.md](https://willemk-stack.github.io/option-pricing-library/roadmap/)

## License

Licensed under the **Apache-2.0** License. See [LICENSE](./LICENSE) for details.
