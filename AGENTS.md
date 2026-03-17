# AGENTS.md

## Purpose
This repo contains a MkDocs Material documentation site for option-pricing-library.

## Ground rules
- Always fix root causes, not screenshot symptoms.
- Prefer editing the source or generator over editing built artifacts directly.
- Validate both light and dark theme.
- Validate at 375, 768, 1280, and 1536 widths.
- Rebuild and retest after each fix.

## Local environment
- Repo root: .
- Docs serve command: mkdocs serve -a 127.0.0.1:8000
- Docs base URL to inspect: http://127.0.0.1:8000/option-pricing-library/

## Important targets
- Homepage
- Proof path pages
- Performance evidence
- Any page containing generated figures or embedded SVGs

## Visual bug classes
1. DOM/CSS layout bugs
2. Generated SVG/image bugs
3. Theme/color/contrast bugs

## Expected outputs from any audit
- issue summary
- root cause
- files changed
- verification steps