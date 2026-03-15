# Invariants for the eSSVI / local-vol / PDE workflow

Use these as review prompts, not as excuses to add unnecessary code.

## Surface and calibration invariants
- Calibration or projection changes should preserve sane parameter domains and produce explainable outputs rather than silent nonsense.
- No-arbitrage related behavior should remain consistent with the repo's existing diagnostics and validation tests.
- Surface changes should not silently change the meaning of public objects or their documented fields.

## Local-vol invariants
- Constant-vol or near-constant-vol cases should remain a high-priority sanity check.
- Invalid regions, masks, or denominator-failure diagnostics should stay explicit rather than being hidden.
- Local-vol changes should be validated with both numerical checks and representative diagnostics-heavy tests.

## PDE invariants
- Black-Scholes baseline consistency should remain intact where the existing tests expect it.
- Convergence or remedy changes for discontinuous payoffs should be explained in terms of stability, bias, or oscillation control.
- Domain or grid-policy changes should be treated as behavior changes, not just implementation details.

## Typing and API invariants
- Avoid widening public surface area unless the task requires it.
- Preserve existing callable shapes unless the task explicitly changes an interface.
- If a signature or return shape changes, ensure API tests, docs, and changelog implications are checked.

## What to write down after validation
- Which invariants were explicitly checked
- Which tests support those checks
- Whether any docs or examples now need to move with the code
