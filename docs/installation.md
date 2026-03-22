# Installation

This repo targets Python 3.12+ and supports a small core install plus extras for plotting, notebooks, docs, and development.

## Fastest install from GitHub

```bash
pip install "git+https://github.com/willemk-stack/option-pricing-library.git"
```

Quick import check:

```bash
python -c "import option_pricing; print(option_pricing.__name__)"
```

## Editable local checkout

From the repository root:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .
```

Use the editable install while working on the library, notebooks, or docs.

## Supported extras

These extras come directly from `pyproject.toml`:

| Extra | Install command | Use it for |
| --- | --- | --- |
| Core only | `pip install -e .` | Pricing library development without optional tooling |
| Plotting | `pip install -e ".[plot]"` | Matplotlib and pandas-backed diagnostics or figure generation |
| Notebooks | `pip install -e ".[notebooks]"` | Jupyter-based demo and exploration workflows |
| Development | `pip install -e ".[dev]"` | Tests, benchmarks, linting, formatting, and type checks |
| Docs | `pip install -e ".[docs]"` | MkDocs Material and API-reference generation |

## Common local commands

Run the same checks used throughout the repo:

```bash
ruff check .
black --check .
pytest -q
mypy
```

Build the published-style benchmark bundle:

```bash
RUN_BENCHMARKS=1 pytest benchmarks -q --benchmark-only --benchmark-json benchmarks/artifacts/pytest-benchmark.json --benchmark-verbose
python scripts/build_benchmark_artifacts.py --pytest-benchmark-json benchmarks/artifacts/pytest-benchmark.json
```

Build the docs locally after installing `.[docs]`:

```bash
mkdocs serve
```
