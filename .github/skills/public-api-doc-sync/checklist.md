# Public API and docs sync checklist

## Public surface files
- `src/option_pricing/__init__.py`
- `API_STABILITY.md`
- `CHANGELOG.md`

## Docs likely to need updates
- `README.template.md`
- `README.md` (generated)
- `docs/api/public.md`
- `docs/api/pricers.md`
- `docs/api/vol.md`
- `docs/architecture.md`
- relevant pages under `docs/user_guides/`

## Generated or validated artifacts
- README regenerated with `python scripts/render_readme.py`
- D2 diagrams rendered with `python scripts/render_d2_diagrams.py`
- docs figures built with `MPLBACKEND=Agg python scripts/make_docs_figures.py`
- docs validated with `mkdocs build --strict`

## Ask before merging
- Did a public export, signature, or import path change?
- Did an example or user-facing workflow change?
- Does the changelog reflect the user-visible impact?
