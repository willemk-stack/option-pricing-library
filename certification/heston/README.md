# Heston certification

This directory contains tooling and policy evidence for certifying a frozen
OPL Heston implementation for OMRR. Certification is performed against a
separate, clean, detached checkout of the implementation commit. The tooling
branch is never treated as the implementation being certified.

Run:

```powershell
python scripts/generate_heston_certification.py `
  --implementation-worktree C:\path\to\clean\frozen\opl `
  --output-dir certification\evidence\omrr_opl_heston_certification_v2
```

The generated certificate conforms exactly to
`omrr_opl_heston_certification.v2`. OMRR's v2 validator rejects unknown
top-level fields, so richer convergence, bounds, parity, scalar/batch, tooling,
and environment details are stored in evidence artifacts whose SHA-256 hashes
are bound by the certificate.

The normalized OMRR policy is intentionally small because it is the exact
consumer contract. The separate certification scope defines the tested
numerical envelope, grid, tolerances, and mandatory categories. This split is
explicit and does not broaden the certified domain.

Library warning flags and warning-to-failure promotion are certified. OPL does
not expose an automatic warning-triggered rerun API at the frozen commit;
runtime fallback orchestration remains an E1.1 responsibility.

V2 replaces the misleading single Cartesian parameter box with named coupled
parameter regimes. The broad envelope is only a quick outer bound: a production
parameter set is certified only when every parameter lies inside the same named
regime. The production price-bound tolerance is aligned with OMRR's persisted
price qualification tolerance.
