Temporary Engeneerting notes

# Finite Difference Methods in Financial Engineering (Duffy, 2006) — Sprint 1 Notes

*(PDE approach + 1D finite-difference scaffold for Black–Scholes European options)*

---

## 1) PDE framing: what problem are we solving?

### Canonical parabolic PDE form

A broad class of pricing PDEs (after suitable transforms) can be written as a **second-order parabolic PDE**:

\[
\frac{\partial u}{\partial t} = Lu
\]

with the (linear) spatial differential operator

\[
Lu \equiv \sum_{i, j=1}^{n} a_{i,j}(x, t), \frac{\partial^2 u}{\partial x_i \partial x_j}

* \sum_{j=1}^{n} b_j(x, t), \frac{\partial u}{\partial x_j}
* c(x,t),u
\]

Interpretation (in 1D):

* **Diffusion term** (a(x,t),u_{xx}): smooths the solution.
* **Convection/drift term** (b(x,t),u_x): transports features.
* **Reaction/discount term** (c(x,t),u): growth/decay (often (-ru) in finance).

### Initial/terminal + boundary conditions (IBVP mindset)

A PDE problem is not “just the PDE”. You must also specify:

* **Domain**: (x \in [x_{\min}, x_{\max}]), (t \in [0, T])
* **Initial condition** (or **terminal condition** in finance and then solve backward in time)
* **Boundary conditions** at (x_{\min}) and (x_{\max})

In option pricing:

* “Initial condition” for a forward PDE becomes a **terminal payoff** for backward time stepping:
  
\[
V(S,T)=\text{payoff}(S)
\]

---

## 2) What “well-posed” means (practical definition)

A PDE problem is **well-posed** if:

1. **Existence**: a solution exists
2. **Uniqueness**: the solution is unique
3. **Continuous dependence on data**: small changes in inputs/BC/IC cause small changes in the solution

Practical takeaway:

* If boundary/initial conditions are missing, incompatible, or unstable numerically, you can get non-unique or exploding solutions (even if the PDE looks fine).

---

## 3) Finite differences: operators, grids, truncation error

### Grid + notation

Use a uniform spatial grid and time grid:

* \(x_j = x_{\min} + jh\), for \(j=0,\dots,N\), where \(h=\Delta x\)
* \(t_n = n k\), for \(n=0,\dots,M\), where \(k=\Delta t\)

Let \(u_j^n \approx u(x_j,t_n)\).

### First derivative approximations

Forward difference (1st order):

\[
u_x(x_j,t_n) \approx \frac{u_{j+1}^n-u_j^n}{h} ;+; O(h)
\]

Backward difference (1st order):

\[
u_x(x_j,t_n) \approx \frac{u_j^n-u_{j-1}^n}{h} ;+; O(h)
\]

Central difference (2nd order):

\[
u_x(x_j,t_n) \approx \frac{u_{j+1}^n-u_{j-1}^n}{2h} ;+; O(h^2)
\]

### Second derivative approximation

Central difference (2nd order):

\[
u_{xx}(x_j,t_n) \approx \frac{u_{j+1}^n - 2u_j^n + u_{j-1}^n}{h^2} ;+; O(h^2)
\]

### Time derivative approximation (common)

Forward Euler time derivative (1st order):

\[
u_t(x_j,t_n) \approx \frac{u_j^{n+1}-u_j^n}{k} ;+; O(k)
\]

---

## 4) Truncation error, consistency, stability, convergence

### Truncation error (TE)

* Derived by Taylor expanding the exact solution and seeing what terms are discarded.
* The **leading discarded term** determines the **order**.

Example: central first derivative is (O(h^2)) accurate; forward/backward is (O(h)).

### Richardson extrapolation (concept)

If a method has error (E(h)=C h^p + o(h^p)), then combining solutions at (h) and (h/2) can cancel the leading term and improve accuracy.

### Consistency

A scheme is **consistent** if the truncation error (\to 0) as (h\to 0) and (k\to 0).

### Stability

A scheme is **stable** if errors (rounding + perturbations) do not amplify uncontrollably as steps proceed.

* For many linear schemes, stability is analyzed via **amplification factors** (Fourier/von Neumann analysis).
* For explicit schemes, stability often yields a **CFL-like restriction**, e.g. something like:
  
\[
\text{(typical)}\quad \frac{a k}{h^2}\le \text{const}
\]

(exact constants depend on discretization and coefficients)

### Convergence

A scheme **converges** if the numerical solution approaches the true solution as (h,k\to 0).

**Lax Equivalence Theorem (linear IVP setting)**:

> For a properly-posed linear problem, **consistency + stability ⇒ convergence**.

---

## 5) Explicit Euler baseline scheme (why it’s not your “production” choice)

### Model convection–diffusion–reaction PDE (1D)

A standard template:

\[
u_t = \sigma(x,t) u_{xx} + \mu(x,t) u_x + b(x,t)u
\]

Using central differences in space and forward Euler in time gives (at node (j), time level (n)):

\[
u_j^{n+1}
= \left( 1 + k b_j^n - \frac{2k\sigma_j^n}{h^2} \right) u_j^n

* k \left( \frac{\sigma_j^n}{h^2} + \frac{\mu_j^n}{2h} \right) u_{j+1}^n
* k \left( \frac{\sigma_j^n}{h^2} - \frac{\mu_j^n}{2h} \right) u_{j-1}^n
\]

### Why explicit Euler is conditionally stable

* Stability requires constraints relating (k) and (h) (especially due to the diffusion term).
* In practice for option pricing grids, explicit schemes often require **too many timesteps** to remain stable.

### Why you typically don’t use it for production pricing

* **Computational cost**: tiny (k) for stability.
* **Oscillation sensitivity**: can oscillate near non-smooth payoffs and sharp gradients.
* **Constraint handling** (e.g., no-arbitrage monotonicity) can be harder to maintain.

---

## 6) Implicit Euler + Crank–Nicolson (CN): the workhorses

### θ-scheme (unifies explicit / implicit / CN)

Write the semi-discrete operator as (F(u,t)) (i.e., the spatial discretization). Then:

\[
u^{n+1} = u^n + k\Big( (1-\theta),F(u^n,t_n) + \theta,F(u^{n+1},t_{n+1}) \Big)
\]

* (\theta=0): explicit Euler
* (\theta=1): implicit Euler
* (\theta=\tfrac12): **Crank–Nicolson**

### CN as “half explicit + half implicit”

\[
\mathbf{u}^{n+1} = \mathbf{u}^n + \frac{k}{2}\left(F(\mathbf{u}^n,t_n)+F(\mathbf{u}^{n+1},t_{n+1})\right)
\]

### Matrix form and tridiagonal structure (1D)

After discretizing \(u_{xx}\) and \(u_x\) in 1D, each interior point \(j\) couples only to \((j-1, j, j+1)\).
That yields a linear system each time step:

\[
A,\mathbf{u}^{n+1} = B,\mathbf{u}^{n} + \mathbf{g}
\]

* (A) and (B) are typically **tridiagonal**
* (\mathbf{g}) contains boundary condition contributions
* Solve efficiently with the **Thomas algorithm** (direct tridiagonal solver)

---

## 7) Convection–diffusion discretization choices (drift matters!)

### Drift term (u_x): central vs upwind

* **Central difference**: 2nd order accurate, but can produce oscillations in **advection-dominated** regimes.
* **Upwind**: 1st order accurate, more diffusive, but often more stable/monotone for strong drift.

Rule of thumb mindset:

* If drift dominates diffusion (large “Péclet number” behavior), consider upwinding or stabilization.

### Diffusion term (u_{xx}): central difference

* Central difference is standard and stable-friendly when used with implicit/CN time stepping.

### Boundary conditions entering the discrete system

Boundary values affect:

* the first and last interior equations (through (u_0^n), (u_N^n))
* and/or they define extra equations if using Neumann/Robin BCs (through “ghost nodes” or one-sided stencils)

---

## 8) PDE essentials for your documentation (IBVP checklist + failure modes)

### IBVP checklist (what to state in docs)

* Domain: (x\in[x_{\min},x_{\max}]), (t\in[0,T])
* PDE coefficients: diffusion/drift/reaction terms
* Terminal/initial condition
* Boundary conditions at both ends (type + formula)
* Numerical scheme: stencil + time stepping + solver
* Stability/convergence remarks (what you expect / monitor)

### Common failure modes

* **Boundary bias**: poor far-field approximation contaminates interior values.
* **Oscillations**: CN + non-smooth payoff (discontinuous derivative at strike) is a classic source.
* **Grid misalignment**: if strike not on grid, Greeks near strike can degrade.
* **Advection dominance**: central drift can create nonphysical oscillations.

---

## 9) Moving to Black–Scholes (finance mapping)

### Black–Scholes PDE (in spot (S))

For a European option (V(S,t)):

\[
\frac{\partial V}{\partial t}

* \frac12\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}
* r S \frac{\partial V}{\partial S}

- rV = 0
\]

with terminal condition \(V(S,T)=\text{payoff}(S)\).

### Far-field boundary conditions (typical)

For a **call**:

* As \(S\to 0\): \(V\to 0\)
* As \(S\to\infty\): \(V\sim S - K e^{-r(T-t)}\) (linear asymptotic)

For a **put**:

* As \(S\to 0\): \(V\sim K e^{-r(T-t)}\)
* As \(S\to\infty\): \(V\to 0\)

In practice at a finite computational boundary, you choose:

* **Truncation** (set a boundary far enough out + impose asymptotic)
* **Linear extrapolation / asymptotic Dirichlet** using the known far-field behavior

### Why log-price coordinates help

Let \(x=\log S\). Then:

* \(S\in(0,\infty)\) maps to \(x\in(-\infty,\infty)\)
* A **uniform grid in (x)** corresponds to **exponential spacing in (S=e^x)**, often giving better resolution near the strike/spot region.
* Under BS, \(\log S_T\) is normally distributed, so truncation can be set using “(m) sigmas”:

\[
x_{\min/\max} \approx \log(S_0) + \left(r-\tfrac12\sigma^2\right)T \pm m,\sigma\sqrt{T}
\]

---

## 10) Implementation notes: PDE finite-difference scaffolding (`pde_fd.py`)

### What to parameterize (so the solver is reusable)

**Grid**

* Coordinate: `S` vs `logS`
* Bounds policy: e.g. `m_sigmas` rule for domain width
* Node count: `Nx` (and later `Ny`)
* Spacing: uniform vs clustered (around strike / barrier / spot)

**Time stepping**

* Scheme: θ-scheme (`theta=0.5` CN, `theta=1.0` implicit Euler)
* `Nt` or `dt_max`
* Rannacher steps: `n_rannacher` (common mitigation for CN oscillations at payoff discontinuity)
* Event times list: dividends, Bermudan exercise dates, monitoring dates

**Boundary conditions**

* Type per boundary: `DIRICHLET / NEUMANN / ROBIN`
* Boundary value functions: callable in time (or time-to-maturity)
* “Far-field model”: call asymptotic vs put asymptotic behavior

**Payoff / constraints**

* Payoff function (callable)
* (Later) American/Bermudan: projection / penalty / PSOR choices
* Grid alignment: ensure strike is on-grid (or be explicit about interpolation)

**Linear solver**

* Tridiagonal (Thomas) for 1D
* Sparse solver options and tolerances (for future 2D/3D)

**Convergence runner**

* Refinement schedule (halve (h), adjust (k))
* Error norm (L∞, L2) vs reference (closed-form BS price)

---

## 11) Crank–Nicolson: known shortcomings (Duffy-style cautions) + mitigations

Key issues to remember:

* CN is **second-order** on **uniform** meshes (order can degrade on nonuniform grids without care).
* CN can generate **spurious oscillations/spikes** with **non-smooth payoff** and/or incompatible BC/IC.
* When diffusion is small (advection-dominated), CN can behave closer to **neutrally stable**, making it sensitive to rounding/error propagation.
* CN often gives **poor delta/gamma near strike** unless you treat the payoff discontinuity carefully.

Common mitigations you can plan for (Sprint 2+):

* **Rannacher smoothing**: a few initial implicit Euler steps before switching to CN
* **Upwinding/stabilization** for strong drift
* **Grid design**: place strike exactly on grid; local refinement near strike
* **Smoother payoff initialization** (careful—must not bias price)

---

## References

* Duffy, Daniel J. (2006). *Finite Difference Methods in Financial Engineering: A Partial Differential Equation Approach.*

---
