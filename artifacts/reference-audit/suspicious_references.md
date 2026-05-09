# Suspicious references and marker triage

Raw marker scan: `suspicious_markers.txt`.

## Marker results

No `TODO(evidence)`, `TODO(reference)`, `citation needed`, `source needed`, `TBD`, `FIXME`, or `REVIEW` markers were found in the scanned authored docs.

## Suspicious source patterns resolved

| Pattern | Location | Resolution |
|---|---|---|
| Local PDF filenames used as citations | `docs/notes/**` reference sections | Replaced with bibliographic references. |
| `Options Futures Derivatives (2021).pdf` under folder `Hall` | Foundations/pricing/implied-vol notes | Corrected to Hull, *Options, Futures, and Other Derivatives*. |
| `FastHestonCalib-.pdf` | Heston calibration notes | Replaced with Cui, del Ba?o Rollin, and Germano. |
| `HestonTrap.pdf` | Heston pricing/diagnostics notes | Replaced with Albrecher, Mayer, Schoutens, and Tistaert. |
| `AndersenHestonSimulation.pdf` | Heston MC/QE notes | Replaced with Andersen. |
| `broadie_kaya_exact_sim_or_2006.pdf` | Heston MC/QE notes | Replaced with Broadie and Kaya. |
| `Monotonic Spline.pdf`, `No-Arbitrage Spline.pdf`, `PAV Regression.pdf`, `Tikhonov Regularization.pdf` | Interpolation note | Identified from title pages and replaced with author/title references. |
| `Thomas Notes.pdf` | Not formally cited | Left uncited; finite-difference mechanics use Tavella-Randall/Duffy instead. |

## AAD / ML / trading scan

- AAD / autodiff / Homescu / Baydin / Lopez de Prado did not appear in core pricing or volatility implementation notes.
- Sinclair appears only in `docs/notes/volatility/implied_volatility.md` as practitioner volatility-quoting context, so it was retained.
