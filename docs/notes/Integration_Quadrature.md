# Fixed-Rule-Quadrature

## why Gaussian quadrature is tied to orthogonal polynomial families

Let $p_{n+1}$ be the degree-$n+1$ polynomial orthogonal with respect to the weight $w(x)$ on ([a,b]), meaning

$$
\langle p,q\rangle=\int_a^b w(x)p(x)q(x),dx.
$$

Then the **$(n+1)$-point Gaussian quadrature rule** uses the zeros $x_1,\dots,x_{n+1}$ of $p_{n+1}$ as its nodes, and there exist weights $\lambda_1,\dots,\lambda_{n+1}$ such that

$$
\int_a^b w(x) f(x),dx
\approx
\sum_{i=1}^{n+1} w_i f(x_i),
$$

and this rule is exact for every polynomial $f$ of degree at most $2n+1$

## Why orthogonal polynomials appear

> Gaussian quadrature arises because an ((n+1))-node rule that is exact for all polynomials up to degree (2n+1) must use as its node polynomial the degree-(n+1) orthogonal polynomial for the weight (w(x)); therefore the nodes are the zeros of that orthogonal polynomial.

## what makes Gauss–Legendre structurally different from low-order Newton–Cotes style rules.

Structurally:

$$
\int_a^b f(x),dx \approx \sum_{i=0}^n w_i f(x_i)
$$

for both methods, but the $x_i$ come from very different principles.

For Newton–Cotes:

* $x_i$ are fixed by geometry: equally spaced.
* weights come afterward from integrating Lagrange basis polynomials.
* this simplicity is nice, but high-order versions become unstable and can even produce negative weights.

For Gauss–Legendre:

* $x_i$ are **not** equally spaced.
* they are chosen to satisfy an orthogonality condition tied to Legendre polynomials.
* with (n) nodes, Gauss–Legendre is exact for all polynomials up to degree (2n-1), which is much better than Newton–Cotes with the same number of nodes.

Another way to say it:

* **Newton–Cotes** is an **interpolate-then-integrate on a fixed grid** method.
* **Gauss–Legendre** is an **optimize the nodes for polynomial exactness** method.

> Newton–Cotes uses equally spaced interpolation nodes and integrates the interpolant, whereas Gauss–Legendre chooses nonuniform nodes and weights so that the quadrature is maximally exact; in the Legendre case, the nodes are the roots of Legendre polynomials.

## Why deterministic nodes and weights is a meaningful implementation asset for a Heston Backend

In the Heston pricer, the main numerical problem is not merely to approximate an integral, but to do so stably, reproducibly, and diagnostically across difficult parameter regimes. The Fourier integrand can change character across maturities and parameter sets, and the pricing engine must remain well behaved not only for one price evaluation, but also under calibration loops, regression tests, and parameter perturbations. That is why a deterministic fixed-rule backend is valuable.

The practical advantage of deterministic quadrature is that the integration grid is known in advance: once the truncation boundary $u_{max}$, the panel layout, and the rule order are chosen, the nodes and weights are fixed. This makes the backend easier to reason about than black-box adaptive integration, where the effective evaluation pattern changes from case to case.

clearer convergence diagnostics. With deterministic nodes and weights, refinement is structured: increase $u_{max​}$, increase the number of panels, or increase the nodes per panel, and observe how the price changes.

Other points of interest:

- Calibration friendly
- nodes and weights can be cached and reused across repeated evaluations, vectorized across strikes and held fixed while only the integrand values change.

## why compound rules are the natural bridge between a finite interval and local resolution control

Composite rules and panelization are not conceptually identical, but in implementation they work hand in hand: panelization creates the local subinterval structure, and the composite rule is obtained by applying the chosen fixed quadrature rule on each panel and summing the results. This makes composite quadrature a natural bridge between finite-interval integration and local resolution control. A single global rule gives uniformity and conceptual simplicity, but it can be too rigid when different parts of the domain require different levels of resolution. In Heston this can be especially important. Panelization restores that flexibility by allowing resolution to be concentrated where it is needed, while the composite construction preserves a single consistent integration procedure over the whole interval. In that sense, composite rules combine the coherence of a global quadrature framework with the adaptability of local refinement.

## why panelization makes diagnostics clearer

> Panelization makes diagnostics clearer because it localizes numerical error. Instead of observing only that the global integral converges slowly, we can inspect convergence panel by panel and identify which subintervals contribute most of the residual error. In practice, this lets us say not just that the quadrature is under-resolved, but that specific regions of the integration domain are under-resolved and may require more panels, higher local order, or different spacing.

early panels may indicate under-resolution near the origin,
late panels may indicate tail truncation issues,
irregular behavior across adjacent panels may indicate oscillation or cancellation.

## which settings should become explicit controls in the pricing API

- Panel spacing: uniform, local, ...
- Order/Nodes per panel(s):
- convergence tolerance
- 



## Implementation/Interface choices:

- Composite rule should be clean arbitrary-n Gauss–Legendre interface.

- `Backend` = `gauss_legendre`
- `u_max`
- `n_panels`,
- `nodes_per_panel`,
- `panel_spacing` (optional).

## Gautschi Core

We take  $\tilde \pi _k(t)$, the orthonormal family of Gauss polynomials.

Then from Theorem 1.31 $\Rightarrow$, the zero's $t_\nu^{(n)}$ of the Gauss polynomials are eigenvalues of $\mathbb{J}_n (d,\\lambda)$ and $\tilde \pi ^{(n)} _\nu(t)$ are the corresponding eigenvectors.

<details>
<summary>Why?</summary>

This follows from rewriting the three step recurrance relation as 

$$
t\tilde \pi (t) 
= 
\mathbb{J}_n ^{d\lambda} \tilde\pi (t)
+
\sqrt{\beta_n}\tilde\pi (t)\mathbb{e_n},
\qquad
e_n := (0,0,\dots,0,1)^T
$$

and cleverly applying properties such as $\tilde\pi _0 = \frac{1}{\sqrt{\beta_0}},$
gives the final result.

</details>

The corollary to Theorem 3.1 gives $v_\nu$, the normalised eigenvector of $\mathbb{J}_n$ to eigenvalue $\lambda_\nu$,

i.e satisfying,

$$
J_n = \sum_\nu \lambda_\nu\, U_\nu \otimes U_\nu,
\qquad
U_\mu^T U_\nu = \delta_{\mu\nu}.
$$

Then,

$$
\beta_0\,v_{\nu,1}^2
\le
\frac{1}{\sum_{k=0}^{n-1}\bigl(\tilde \pi_k(\tau_\nu^{(n)})\bigr)^2}
$$

Because

$$
v_{\nu}
=
\frac{\tilde \pi\bigl(\tau_\nu^{(n)}\bigr)}
{\left\|\tilde \pi\bigl(\tau_\nu^{(n)}\bigr)\right\|}
$$

---

Now, in the Gauss formula, take

$$
f(x)=\tilde \pi_k(x), \qquad k \le n-1.
$$

$$
\pi _0 = \frac{1}{\sqrt{\beta_0}}
$$

By orthogonality we get

$$
\sqrt{\beta_0}\delta _{k,0}
=
\sum_{\nu=1}^n \lambda_\nu^{G}\,\tilde \pi\!\bigl(\tau _\nu^{G}\bigr).
$$

Equivalently,

$$
P\,\lambda^{G} = \sqrt{\beta_0^{1/2}} \mathbb{e_1} \qquad (*)
$$

where

$$
P = \bigl[\tilde \pi(\tau_1^{G}),\dots], \qquad \lambda^{G} = [\lambda _i^{G},\cdots,]^T
$$

$$
P^T P = D_{\pi} = \operatorname{diag}(d_0,d_1,\dots,d_{n-1})
$$

and

$$
d_{\nu-1} = \sum_{k=0}^{n-1} \bigl[\tilde \pi _k(\tau _\nu^{G})\bigr]^2.
$$

Multiplyign both sides of (*) with $P^T$ gives

$$\lambda_{\nu}^G = \frac{1}{\sum_{k=0}^{n-1} [\tilde{\pi}_k(\tau_{\nu}^G)]^2}, \quad \nu = 1, 2, \dots, n$$

**Reference**
Walter Gautschi, *Orthogonal Polynomials: Computation and Approximation*, Numerical Mathematics and Scientific Computation, Oxford University Press, Oxford, 2004. ([Oxford University Press])
