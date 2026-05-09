# Strong claims requiring support

Raw scan: `strong_claims_raw.txt`.

The table below groups duplicate raw hits by claim type. Strong language tied to code defaults or diagnostic actions is treated as repository policy unless a note explicitly ties it to a cited source or generated artifact.

| File | Strong claim / phrase | Classification | Resolution |
|---|---|---|---|
| `README.md`, `README.template.md` | "validated PDE outputs" | backed by test/artifact | README links to proof pages and generated artifacts; no bibliography added. |
| `docs/architecture/docs-pipeline.md` | "stable contract", "validated workflow" | repository policy / backed by CI design | No citation needed; docs-pipeline policy language. |
| `docs/api/heston.md` | Bounds and seed helpers are safeguards, not uniqueness claims | repository policy | Already framed as safeguards. |
| `docs/notes/heston/heston_calibration.md` | "robust proxy for IV error" | repository policy + literature context | Added Christoffersen-Jacobs and explicit loss-function policy wording. |
| `docs/notes/heston/heston_calibration_seeds.md` | Bounds are safeguards, not no-arbitrage guarantees | repository policy | Already framed as heuristic/policy; retained. |
| `docs/notes/heston/heston_monte_carlo.md` | Scheme recommendation / "always better" warning | repository policy | Already says the summary is not proof of global superiority; Andersen/Glasserman/Broadie-Kaya references normalized. |
| `docs/notes/heston/heston_pricing_conventions.md` | "stable in singular near-zero-vol-of-vol regime" | backed by test/artifact | Kept as implementation fallback; references now include Cui for analytic gradients and quadrature/MC sources where relevant. |
| `docs/notes/heston/heston_pricing_conventions.md` | "trustworthy" Fourier result | repository policy | Diagnostics policy explicitly states finite value is insufficient; references normalized. |
| `docs/notes/heston/heston_pricing_conventions.md` | `robust` quadrature for final capstone reporting | repository policy | Kept in resolved implementation policy; quadrature sources added but concrete tier remains repo policy. |
| `docs/notes/heston/heston_pricing_conventions.md` | "must" diagnostic warnings surfaced | repository policy | Kept as public implementation contract. |
| `docs/notes/heston/heston_qe_notes.md` | Positive-correlation moment conditions "must be checked" | backed by citation | Andersen reference normalized. |
| `docs/notes/heston/heston_qe_notes.md` | Broadie-Kaya not practical production default | repository policy + literature context | Kept as implementation interpretation; Broadie-Kaya reference normalized. |
| `docs/notes/heston/heston_stochastic_vol.md` | Stable characteristic-function branch handling | backed by citation + tests | Added Albrecher et al. and pointed detailed policy to Fourier diagnostics. |
| `docs/notes/heston/heston_fourier_diagnostics.md` | Warning severities and `robust` reruns | repository policy | Header already says diagnostics/policy unless cited; quadrature references added. |
| `docs/notes/heston/heston_quadrature_policy.md` | "benchmark-backed", `gauss_legendre` default, tier thresholds | repository policy + backed by tests/artifacts | Added numerical-method sources while keeping concrete defaults explicitly policy. |
| `docs/notes/local-vol-pde/dupire_local_vol.md` | "stable numerical sense", "least trustworthy" boundaries | repository policy | Header already distinguishes Dupire math from masking/trimming policy; references normalized. |
| `docs/notes/volatility/essvi_calibration_design.md` | Smooth surface not always more correct | repository policy / should be retained | Explicitly marked `theta/psi/eta` notation as repository notation. |
| `docs/notes/volatility/interpolation.md` | "arbitrage-safe", "never create negative density" | should be softened in future | This remains a provisional heuristic note; references now identifiable. A future content pass should qualify LP/grid assumptions more carefully. |
| `docs/notes/volatility/implied_volatility.md` | "robust solver", "safe domain" | repository implementation context | Retained as solver guidance; references normalized. |
| `docs/notes/volatility/svi_calibration_design.md` | Robust loss / repair behavior | repository policy backed by tests | References normalized; tests remain linked in note. |
| `docs/user_guides/heston*.md`, `docs/user_guides/heston_diagnostics.md` | `robust`, diagnostics reruns, backend disagreement | repository policy / proof-page guidance | No formal bibliography added; these pages link into Heston notes. |
| `docs/user_guides/essvi.md`, `docs/user_guides/surface_workflow.md` | Arbitrage-safe / trustworthy workflow language | repository policy + generated evidence | No formal bibliography added; notes now carry formal references. |
