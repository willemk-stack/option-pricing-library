# Finite Difference Methods for Black–Scholes (Duffy, 2006) — Sprint 1 Notes (Non-uniform x-grid)

*PDE framing + 1D finite-difference scaffold for European options on a **non-uniform** spatial grid*

---

## 0) Notation (non-uniform grid, keep consistent everywhere)

### Spatial grid (non-uniform)



* **Spatial nodes:** \(x_0 < x_1 < \dots < x_N\)
* **Local spacings:**

  \[
  h_{j-} = x_j - x_{j-1},\qquad h_{j+} = x_{j+1}-x_j
  \]

  (for interior node \(j=1,\dots,N-1\))

### Time grid (often uniform, but can be general)

* **Time nodes:** \(t_0 < t_1 < \dots < t_M\)
* **If uniform:** \(t_n = nk\) with \(k=\Delta t\)
* **If non-uniform:** \(k_n = t_{n+1}-t_n\)

### Grid function

* \(u_j^n \approx u(x_j, t_n)\)
* **Interior nodes:** \(j=1,\dots,N-1\); **boundary nodes:** \(j=0, N\)

> In option pricing you often solve backward in calendar time \(t\) with terminal (payoff) at \(T\). Many implementations use time-to-maturity \(\tau = T-t\) to march forward in \(\tau\).

---

## 1) PDE + IBVP framing (what problem are we solving?)

A broad class of pricing PDEs (after transforms) can be written:

\[
u_t = a(x,t)u_{xx} + b(x,t)u_x + c(x,t)u
\]

You must specify an **IBVP** (initial/terminal condition + two boundary conditions). In option pricing:

\[
V(\cdot,T)=\text{payoff}(\cdot)
\]

---

## 2) “Well-posed” (practical definition)

Well-posed means: 1) solution exists, 2) unique, 3) depends continuously on inputs.

**Practical takeaway:** incompatible IC/BC can break numerics even if the PDE looks fine.

---

## 3) Finite differences on a **non-uniform** x-grid (key stencils)

Let \(h_{j-}=x_j-x_{j-1}\) and \(h_{j+}=x_{j+1}-x_j\).

### First derivative (\(u_x\))

**Non-uniform central (2nd order):**

\[
u_x(x_j,t_n)\approx
-\frac{h_{j+}}{h_{j-}(h_{j-}+h_{j+})}u_{j-1}^n
+\frac{h_{j+}-h_{j-}}{h_{j-}h_{j+}}u_j^n
+\frac{h_{j-}}{h_{j+}(h_{j-}+h_{j+})}u_{j+1}^n
+O(h^2)
\]

**Backward (upwind for \(b>0\), 1st order):**

\[
u_x(x_j,t_n)\approx \frac{u_j^n-u_{j-1}^n}{h_{j-}} + O(h)
\]

**Forward (upwind for \(b<0\), 1st order):**

\[
u_x(x_j,t_n)\approx \frac{u_{j+1}^n-u_j^n}{h_{j+}} + O(h)
\]

> On non-uniform grids, “uniform-grid formulas” like \((u_{j+1}-u_{j-1})/(2h)\) are **not** correct unless \(h_{j-}=h_{j+}\).

### Second derivative (\(u_{xx}\))

**Non-uniform central (2nd order):**

\[
u_{xx}(x_j,t_n)\approx
\frac{2}{h_{j-}(h_{j-}+h_{j+})}u_{j-1}^n
-\frac{2}{h_{j-}h_{j+}}u_j^n
+\frac{2}{h_{j+}(h_{j-}+h_{j+})}u_{j+1}^n
+O(h^2)
\]

### Time derivative (baseline)

* **If uniform:** \(u_t(x_j,t_n) \approx \frac{u_j^{n+1}-u_j^n}{k} + O(k)\)
* **If non-uniform:** \(u_t(x_j,t_n) \approx \frac{u_j^{n+1}-u_j^n}{k_n} + O(k_n)\)

---

## 4) Truncation error, consistency, stability, convergence

* **TE:** from Taylor expansion; leading term gives order.
* **Consistency:** TE \(\to 0\) as \(\max(h_{j-},h_{j+}), k \to 0\).
* **Stability:** perturbations don’t amplify uncontrollably.
* **Convergence:** numerical \(\to\) true solution as mesh refines.

---

## 5) Time stepping: the \(\theta\)-scheme (explicit / implicit / CN)

Let \(L_h\) be the **non-uniform** spatial discretization of \(L\). The \(\theta\)-scheme is:

\[
\mathbf{u}^{n+1} = \mathbf{u}^n + k\Big((1-\theta)L_h\mathbf{u}^n + \theta L_h\mathbf{u}^{n+1}\Big)
\]

**Matrix form:**

\[
\left(I - k\theta L_h\right)\mathbf{u}^{n+1} = \left(I + k(1-\theta)L_h\right)\mathbf{u}^{n} + \mathbf{g}^{n,n+1}
\]

* \(\theta=0\): explicit Euler
* \(\theta=1\): implicit Euler
* \(\theta=0.5\): Crank–Nicolson

In 1D with 3-point stencils, \(L_h\) is still **tridiagonal** even on a non-uniform grid.

---

## 6) CN interior-only system + boundary contributions

Full grid vector: \(\mathbf{u}^n = [u_0^n, u_1^n,\dots,u_{N-1}^n,u_N^n]^T\). For Dirichlet boundaries: \(u_0^n = \alpha(t_n)\) and \(u_N^n = \beta(t_n)\).

\[
A_{II}\mathbf{u}_I^{n+1} = B_{II}\mathbf{u}_I^n + \big(B_{IB}\mathbf{u}_B^n - A_{IB}\mathbf{u}_B^{n+1}\big) + \mathbf{g}_I
\]

**Key implementation fact:** boundary coupling only affects the **first** and **last** entries of the RHS because the stencil is 3-point.

---

## 7) Drift discretization (central vs upwind)

* **Non-uniform central:** 2nd order but can oscillate when advection dominates diffusion.
* **Upwind:** 1st order, adds numerical diffusion, improves monotonicity.

---

## 8) Black–Scholes mapping (finance \(\to\) PDE)

Black–Scholes in \(S\):

\[
V_t + \tfrac12\sigma^2 S^2 V_{SS} + r S V_S - rV = 0,\qquad V(S,T)=\text{payoff}(S)
\]

Log-price (\(x=\log S\)) is still useful; the grid can be **uniform or clustered in \(x\)**.

---

## 9) Implementation scaffold (`pde_fd.py`)

* **Grid:** Coordinate (`S` vs `logS`), nodes (`Nx`), spacing (uniform vs clustered).
* **Time stepping:** `theta`, `Nt`, Rannacher smoothing steps.
* **Linear solver:** 1D Thomas algorithm (still works for non-uniform tridiagonal).

---

## 10) CN issues + mitigations

* CN ringing near kinks (mitigate with Rannacher smoothing).
* Grid design: cluster near strike/spot; ensure strike lies on-grid node.

---

## 11) X-grid clustering (sinh mapping around a center)



Once you use a clustered grid, you must use the **non-uniform stencils** from Section 3.