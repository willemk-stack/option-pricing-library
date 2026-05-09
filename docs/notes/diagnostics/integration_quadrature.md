# Fixed-rule quadrature

!!! note "Status"
    This is a numerical-method note plus Heston implementation rationale.
    Gaussian quadrature facts are standard mathematical facts, with the local
    source-library quadrature references listed below. Heston-specific comments are repository
    implementation policy unless they are tied to tests or generated
    diagnostics elsewhere.

## Gaussian quadrature and orthogonal polynomials

Let \(p_{n+1}\) be the degree-\(n+1\) polynomial orthogonal with respect to a
weight \(w(x)\) on \([a,b]\):

\[
\langle p,q\rangle=\int_a^b w(x)p(x)q(x)\,dx.
\]

The \((n+1)\)-point Gaussian quadrature rule uses the zeros
\(x_1,\ldots,x_{n+1}\) of \(p_{n+1}\) as its nodes, with weights
\(\lambda_1,\ldots,\lambda_{n+1}\):

\[
\int_a^b w(x) f(x)\,dx
\approx
\sum_{i=1}^{n+1} \lambda_i f(x_i).
\]

For sufficiently regular \(f\), the rule is exact for every polynomial of
degree at most \(2n+1\). The key structural point is that the nodes are chosen
from orthogonality, not from an equally spaced geometric grid.

## Newton-Cotes versus Gauss-Legendre

Both Newton-Cotes and Gauss-Legendre rules can be written as weighted sums of
function values, but their node choices come from different principles.

For Newton-Cotes rules:

- nodes are fixed by equally spaced geometry;
- weights are obtained by integrating interpolation basis polynomials;
- high-order closed rules can become poorly conditioned and can have negative
  weights.

For Gauss-Legendre rules:

- nodes are not equally spaced;
- nodes are the roots of Legendre polynomials on the transformed interval;
- with \(n\) nodes, the rule is exact for polynomials up to degree \(2n-1\).

That is the mathematical reason Gauss-Legendre is attractive when a fixed
finite interval and a high-order deterministic rule are useful.

## Heston implementation rationale

The Heston Fourier pricer needs more than a one-off integral approximation. In
calibration and diagnostics, the same pricing path is evaluated across strikes,
maturities, seeds, and parameter perturbations. A deterministic fixed-rule
backend makes those repeated evaluations easier to reproduce and inspect.

Repository implementation policy:

- keep the truncation boundary \(u_{\max}\), panel count, nodes per panel, and
  spacing policy explicit;
- reuse deterministic nodes and weights across repeated evaluations where the
  integration rule is unchanged;
- surface panel-level diagnostics so tail, origin, oscillation, and
  cancellation problems can be reviewed locally;
- keep SciPy `quad` as an independent comparison backend rather than the main
  calibration workhorse.

These are implementation and diagnostics choices. They do not prove that one
quadrature rule is universally best across all Heston regimes.

## Panelization

Panelization divides the finite integration interval into subintervals and
applies the same fixed rule on each panel. The composite rule is then the sum of
panel contributions. This gives the implementation one global quadrature
contract while still allowing diagnostics to identify where resolution is
missing.

Typical panel-level signals:

- early panels: possible under-resolution near the origin;
- late panels: possible tail truncation issue;
- alternating or spiky panels: possible oscillation or cancellation problem.

The Heston diagnostics pages use these signals as review evidence, not as
automatic rejection rules.

## API controls

The implementation should keep these controls visible:

- `backend`: for example `gauss_legendre` or `quad`;
- `u_max`: truncation boundary;
- `n_panels`: composite-rule panel count;
- `nodes_per_panel`: fixed-rule order on each panel;
- `panel_spacing`: optional spacing policy.

## Jacobi-matrix construction

Gautschi gives the standard computational route from orthogonal-polynomial
recurrence coefficients to Gaussian quadrature nodes and weights. In that
setup, the nodes are eigenvalues of the finite Jacobi matrix associated with
the recurrence, and the weights are recovered from the normalized eigenvectors.

This repo does not require the Heston docs to rederive that construction in
full. The important implementation point is narrower: if a fixed
Gauss-Legendre rule is selected, the node and weight arrays are deterministic
and can be treated as part of the pricing configuration.

## References

- Davis, P. J., & Rabinowitz, P. (1984). *Methods of Numerical Integration* (2nd ed.). Academic Press.
- Gautschi, W. (2004). *Orthogonal Polynomials: Computation and Approximation*. Oxford University Press.
- Hale, N., & Townsend, A. Fast and accurate computation of Gauss-Legendre and Gauss-Jacobi quadrature nodes and weights.
