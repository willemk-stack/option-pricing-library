# API Stability Policy

This repository is a portfolio-quality pricing/volatility library. The project may remain closed to external contributions, but it still follows a clear API discipline so users (and future maintainers) know what is safe to depend on.

## 1. Definitions

### Public API (stable)
The **Public API** is:
- Anything re-exported at the package top level: `import option_pricing as op` (i.e., symbols in `option_pricing/__init__.py`).
- Anything explicitly documented in the published docs under "Public API" / "API Reference".

If it’s public, it will not be broken without a version bump and a changelog entry.

### Internal API (may change anytime)
The **Internal API** is:
- Any module/function/class not in the Public API list.
- Any name starting with `_` (private by convention).
- Most submodules intended for implementation detail (numerics internals, diagnostics helpers, plot helpers, etc.), unless explicitly documented as public.

Internal APIs may be refactored freely between releases.

## 2. Versioning rules

This project aims to follow Semantic Versioning:

- **PATCH** (x.y.Z): bug fixes only; no intentional public API breaks.
- **MINOR** (x.Y.z): new features; public API remains backwards-compatible.
- **MAJOR** (X.y.z): breaking changes permitted.

Until **1.0.0**, the public API is still curated, but changes will be handled via:
- deprecation warnings where feasible
- explicit changelog notes for any break

## 3. Deprecation policy

If a public symbol needs to move or be renamed:
1. Keep the old import path for at least one MINOR release cycle when feasible.
2. Emit a `DeprecationWarning` with:
   - the new import path
   - the version where the old path will be removed
3. Document the deprecation in `CHANGELOG.md`.

## 4. Compatibility notes

- Supported Python versions are defined in `pyproject.toml`.
- Numerical outputs may vary slightly with dependency versions; tests enforce invariants, bounds, and tolerances rather than exact bitwise equality.

## 5. What this policy is not

- This is not a commitment to accept outside contributions.
- This is not a guarantee of perpetual API immutability.
It is a commitment to make breaking changes deliberately, visibly, and professionally.