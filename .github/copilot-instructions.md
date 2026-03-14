# Repository-wide Copilot instructions

This repository is a typed Python 3.12+ option-pricing library with a strong emphasis on numerical validation, API discipline, generated docs, and reproducible demos.

## Core expectations

- Prefer small, targeted edits over broad refactors.
- Preserve the current layered design: simple pricing APIs, instrument-based workflows, curves/context workflows, then advanced vol/local-vol/PDE workflows.
- Keep implementation details internal unless there is a clear reason to expose them.
- When a task is specialized, use the matching skill under `.github/skills` instead of improvising a generic workflow.

## Public API discipline

- Treat anything re-exported from `src/option_pricing/__init__.py` and anything documented in the published API docs as public API.
- Do not rename, remove, or move public symbols casually.
- If a change affects a public symbol, public signature, documented behavior, or recommended import path, also update the relevant docs and `CHANGELOG.md`.
- Follow `API_STABILITY.md` when deciding whether a change is internal-only, additive, deprecated, or breaking.

## Python and code style

- Target Python 3.12 features only.
- Keep code type-aware and consistent with the existing typed style in `src/option_pricing`.
- Prefer explicit, readable code over clever shortcuts.
- Match existing naming and module structure where possible.
- Avoid introducing unnecessary dependencies.
- Keep imports organized and compatible with Ruff/isort behavior.
- Do not weaken typing or remove useful annotations without a strong reason.

## Numerical and quant-specific guidance

- This repo is validation-first. For numerical changes, preserve invariants, tolerances, and representative baselines.
- Do not loosen tolerances casually. If a tolerance changes, explain the numerical reason.
- Prefer fixing the numerical root cause over masking failures with broader tolerances or skipped checks.
- When touching SVI/eSSVI, local-vol, or PDE code, run focused tests first and then expand only if needed.
- If diagnostics semantics change, update tests and docs together.

## Testing and validation

Run the smallest relevant validation set first, then broaden only when the change is wider.

Common repo-wide checks:

```bash
python -m pip install -e ".[dev]"
python scripts/render_readme.py --check
ruff check .
black --check .
mypy src/option_pricing
pytest -q tests
```

Minimum expectations by change type:

- **General source change:** run the nearest targeted tests under `tests/`.
- **Public API or typing change:** run `mypy src/option_pricing` and targeted API tests.
- **README/examples/docs-facing change:** run `python scripts/render_readme.py --check`.
- **Notebook or demo-facing change:** run the relevant notebook or `pytest -q demos --nbmake --nbmake-timeout=300` when appropriate.
- **Benchmark-sensitive change:** use the `benchmark-regression-triage` skill. Full performance benchmarks may be too expensive for routine local use. Prefer cheap targeted perf checks and benchmark subset selection first. Only run full benchmark workflows when explicitly requested or when a change is clearly performance-critical.

- **Surface/local-vol/PDE change:** use the `essvi-localvol-pde-workflow` skill.
- **Public export/docs change:** use the `public-api-doc-sync` skill.

Do not claim a change is complete if the relevant targeted validation has not been run.

## Docs and generated artifacts

- `README.md` is generated. Do not hand-edit it unless the task is explicitly about the generated file itself.
- For README changes, edit `README.template.md` and the relevant example sources, then run:

```bash
python scripts/render_readme.py
```

- If docs pages, diagrams, or generated figures change, validate the docs flow:

```bash
python -m pip install -e ".[docs,plot]"
python scripts/render_d2_diagrams.py
MPLBACKEND=Agg python scripts/make_docs_figures.py
mkdocs build --strict
```

## Project structure guidance

- Production code lives under `src/option_pricing`.
- Tests live under `tests/`.
- Benchmarks live under `benchmarks/`.
- User-facing notebooks live under `demos/`.
- Published docs live under `docs/`.
- Helper scripts for generated artifacts live under `scripts/`.

Place new code in the nearest existing module unless there is a strong architectural reason to create a new one.

## Preferred response style for code changes

When making or proposing changes:

1. State the area touched.
2. Mention any API or numerical risk.
3. List the targeted tests or checks run.
4. Note any docs, README, notebook, or changelog follow-up.

If validation could not be run, say so clearly instead of implying completion.

## Improving agent guidance over time

After completing a task, notice whether the work uncovered a stable repo-specific pattern that should be captured for future use.

- Use `copilot-instructions.md` for broad repo rules, default expectations, and guidance that should apply to most tasks.
- Use a skill or skill support file for narrow, repeatable workflows such as eSSVI/local-vol/PDE validation, public API/doc sync, or benchmark triage.

Do not edit skills or instructions unless explicitly asked to do so.

Do not capture one-off bugs, temporary fixes, flaky failures, or branch-specific workarounds.

Only suggest an update when the pattern is stable, reusable, and likely to help with future work in this repo.
