---
name: public-api-doc-sync
description: Use when changing top-level exports, pricing entrypoints, public types, examples, README, docs, or changelog behavior. Classify public API impact and sync generated docs and examples.
---
This skill keeps public code changes aligned with documentation and release notes.

Use it when a task touches any of these:
- `src/option_pricing/__init__.py`
- public pricing entrypoints or public types
- `README.template.md`, `README.md`, `examples/`
- `docs/api/`, `docs/user_guides/`, `docs/architecture.md`, `mkdocs.yml`
- `API_STABILITY.md` or `CHANGELOG.md`

## Goal
Determine whether a change is internal-only, public-additive, deprecated, or breaking, then update the repo's public surfaces accordingly.

## Classification checklist
Classify the change before editing docs:
- **Internal-only**: implementation detail, no public export or documented behavior changed
- **Public additive**: new public symbol, option, behavior, or documented workflow
- **Deprecation**: old path still works but should warn or be phased out
- **Breaking**: public export, signature, documented behavior, or import path changed incompatibly

If unsure, inspect:
- `src/option_pricing/__init__.py`
- `API_STABILITY.md`
- the relevant docs page in `docs/api/` or `docs/user_guides/`

## Sync rules
### If a public export or entrypoint changes
Check and update as needed:
- `src/option_pricing/__init__.py`
- `docs/api/public.md`
- the nearest API pages under `docs/api/`
- `docs/architecture.md` if the architectural map or canonical entry point changed
- `CHANGELOG.md`

### If examples or README-facing behavior changes
Check and update as needed:
- `README.template.md`
- files in `examples/`
- relevant user guides in `docs/user_guides/`

Then regenerate or validate the README:
```bash
python scripts/render_readme.py
# or check-only
python scripts/render_readme.py --check
```

### If docs pages, figures, or diagrams changed
For a full docs validation flow, use:
```bash
python -m pip install -e ".[docs,plot]"
python scripts/render_d2_diagrams.py
MPLBACKEND=Agg python scripts/make_docs_figures.py
mkdocs build --strict
```

## Changelog and stability policy
- If the change is public-additive, deprecated, or breaking, update `CHANGELOG.md`.
- If a symbol moves or is renamed, follow the repo's API stability expectations instead of silently changing imports.
- Do not present internal refactors as public features.

## Expected output
End with a concise sync summary:
1. **Classification**
2. **Files updated**
3. **Generated artifacts refreshed**
4. **Changelog needed or updated**
5. **Anything intentionally left unchanged**

## Examples
- "Use the public API/doc sync skill after I change top-level exports for vol surfaces."
- "Use the public API/doc sync skill after I rename a public pricing helper."
- "Use the public API/doc sync skill after changing an example that feeds the README."
