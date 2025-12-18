# Installation

## Requirements

- Python **3.10+** (tested with 3.12)
- Core dependencies: **numpy**, **scipy**

Optional extras are available for plotting and notebook workflows.

## Install from a local checkout

From the repository root:

```bash
python -m venv .venv
# activate it (Windows)
.\.venv\Scripts\activate
# or (macOS/Linux)
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .
```

Editable install (`-e`) is the nicest setup while you’re iterating on the library.

## Optional extras

If your `pyproject.toml` defines extras (recommended), you can install them like:

```bash
# Plotting helpers (matplotlib)
pip install -e ".[plot]"

# Notebook workflow (jupyter + matplotlib + pandas)
pip install -e ".[notebooks]"

# Dev tools (pytest, ruff, black, mypy, …)
pip install -e ".[dev]"
```

## Development quality-of-life

### Pre-commit

If you use pre-commit hooks in this repo:

```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-push
```

- On commit: **ruff** + **black** keep formatting consistent.
- On push: **mypy** can run type checks (often slower, so it’s commonly `pre-push`).

### Run the same checks as CI

```bash
ruff check .
black --check .
pytest -q
mypy
```
