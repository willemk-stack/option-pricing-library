# Finite Difference Methods for Black–Scholes (Duffy, 2006)

## Sprint 1 — Non-Uniform Spatial Grid (Design + Implementation Notes)

> **Scope:** This document defines the *1D finite-difference solver* for parabolic pricing PDEs on **non-uniform spatial grids**, with an emphasis on Black–Scholes–type operators. It serves as a solver specification for implementation in `pde_fd.py`.

---

## 0) Notation (Non-Uniform Grid)

### Spatial Grid
* **Nodes:**

$$
x_0 < x_1 < \dots < x_N
$$

* **Local Spacings** (defined at interior nodes $j=1, \dots, N-1$):

$$
h_{j-} = x_j - x_{j-1}, \qquad h_{j+} = x_{j+1} - x_j
$$



### Time Grid
* **Time nodes:** $t_0 < t_1 < \dots < t_M$
* **Uniform time step:** $k = \Delta t$
* **Non-uniform time step:** $k_n = t_{n+1} - t_n$

### Grid Function
* **Approximation:**

$$
u_j^n \approx u(x_j, t_n)
$$

* **Interior nodes:** $j=1, \dots, N-1$
* **Boundary nodes:** $j=0, N$

> **Direction of Integration:** In finance, we solve **backward** from maturity $T$. We define time-to-maturity $\tau = T - t$ and march **forward** in $\tau$.

---

## 1) PDE and IBVP Formulation

We consider parabolic PDEs of the form:

$$
u_\tau = a(x, \tau)u_{xx} + b(x, \tau)u_x + c(x, \tau)u
$$

posed as an **Initial–Boundary Value Problem (IBVP)**:
* **Initial Condition (Payoff):** $u(x, 0) = \text{payoff}(x)$
* **Boundary Conditions:** Specified at $x=x_0$ and $x=x_N$.

---

## 2) Well-posedness

A problem is well-posed if a unique solution exists and depends continuously on data. 

**Implementation Insight:** Numerical stability is highly sensitive to the compatibility between the payoff's curvature at the boundaries and the chosen boundary conditions.

---

## 3) Non-Uniform Finite-Difference Stencils

### 3.1 First Derivative ($u_x$)

#### Central (2nd order, non-uniform)

$$
u_x(x_j) \approx \beta^-_j u_{j-1} + \beta^0_j u_j + \beta^+_j u_{j+1}
$$

with coefficients:

$$
\beta^-_j = -\frac{h_{j+}}{h_{j-}(h_{j-} + h_{j+})}, \quad \beta^0_j = \frac{h_{j+} - h_{j-}}{h_{j-} h_{j+}}, \quad \beta^+_j = \frac{h_{j-}}{h_{j+}(h_{j-} + h_{j+})}
$$

#### Upwind (1st order, monotone)
Used when the drift $b$ dominates diffusion $a$.
* **Backward** (if $b > 0$): $u_x \approx \frac{u_j - u_{j-1}}{h_{j-}}$
* **Forward** (if $b < 0$): $u_x \approx \frac{u_{j+1} - u_j}{h_{j+}}$

### 3.2 Second Derivative ($u_{xx}$)

#### Central (2nd order, non-uniform)

$$
u_{xx}(x_j) \approx \alpha^-_j u_{j-1} + \alpha^0_j u_j + \alpha^+_j u_{j+1}
$$

with coefficients:

$$
\alpha^-_j = \frac{2}{h_{j-}(h_{j-} + h_{j+})}, \quad \alpha^0_j = -\frac{2}{h_{j-} h_{j+}}, \quad \alpha^+_j = \frac{2}{h_{j+}(h_{j-} + h_{j+})}
$$

---

## 4) Operator Assembly

The discrete spatial operator $L_h$ is defined as:

$$
(L_h u)_j = A_j u_{j-1} + B_j u_j + C_j u_{j+1}
$$

Using central discretizations for both derivatives:

$$
\begin{aligned}
A_j &= a_j \alpha^-_j + b_j \beta^-_j \\
B_j &= a_j \alpha^0_j + b_j \beta^0_j + c_j \\
C_j &= a_j \alpha^+_j + b_j \beta^+_j
\end{aligned}
$$

**Storage Strategy:** Store as a tridiagonal matrix or three separate vectors for the **Thomas Algorithm**.



---

## 5) Time Discretization — The $\theta$-Scheme

For the interior vector $\mathbf{u}^n$:

$$
\mathbf{u}^{n+1} = \mathbf{u}^n + k \big[ (1-\theta) L_h \mathbf{u}^n + \theta L_h \mathbf{u}^{n+1} \big]
$$

Rearranging into the system $M_1 \mathbf{u}^{n+1} = M_0 \mathbf{u}^n + \mathbf{g}$:

$$
(I - k\theta L_h) \mathbf{u}^{n+1} = (I + k(1-\theta)L_h) \mathbf{u}^n + \mathbf{g}^{n+1}
$$

* **$\theta = 0$:** Explicit Euler.
* **$\theta = 1$:** Implicit Euler (Robust, diffusive).
* **$\theta = 0.5$:** Crank–Nicolson (High accuracy, susceptible to "ringing").

---

## 6) Boundary Conditions (BCs)

### 6.1 Dirichlet Far-field
For $S_{\min} \approx 0$ and $S_{\max} \gg K$:
* **Call:** $V(S_{\min}) = 0, V(S_{\max}) = S_{\max}e^{-q\tau} - Ke^{-r\tau}$
* **Put:** $V(S_{\min}) = Ke^{-r\tau} - S_{\min}e^{-q\tau}, V(S_{\max}) = 0$

### 6.2 Gamma-Zero (Linearity)
Presume $\frac{\partial^2 V}{\partial S^2} = 0$ at the boundary. 
Implementation: Express $u_0$ as a linear extrapolation of $u_1, u_2$ and substitute into the $j=1$ equation to maintain tridiagonality.

---

## 7) Stability and Safeguards

### 7.1 Peclet Control
Check the local Peclet number to detect advection dominance:

$$
\mathrm{Pe}_j = \frac{|b_j| \cdot \text{avg}(h_j)}{a_j}
$$

### 7.2 Rannacher Smoothing
Crucial for Crank–Nicolson. Replace the first few steps (usually 2 or 4) with **Implicit Euler** using sub-steps $k/2$ to damp high-frequency errors from the payoff kink.

---

## 8) Black–Scholes Mapping

The physical PDE:

$$
V_\tau = \frac{1}{2}\sigma^2 S^2 V_{SS} + (r-q)S V_S - rV
$$

Map coefficients for the solver:
* $a(S) = \frac{1}{2}\sigma^2 S^2$
* $b(S) = (r-q)S$
* $c(S) = -r$

---

## 9) Grid Clustering (Closeness to Strike)

Use a non-uniform mapping (e.g., Sinh or power law) to place more nodes near $S=K$.

$$
S_j = K + A \sinh(\xi_j)
$$



---

## 10) Failure Modes

1.  **Strike Mismatch:** $K$ not being a grid node.
2.  **CN Ringing:** Oscillations near $K$ due to $\theta=0.5$ without smoothing.
3.  **Boundary Pollution:** $S_{\max}$ too small, causing asymptotic BCs to bias the price.
4.  **CFL Violation:** Using $\theta=0$ with too large a $k$.

---

## 11) Implementation Checklist

* [ ] **Grid:** Generate $x_j$ (Uniform or Sinh).
* [ ] **Stencil:** Compute $\alpha_j, \beta_j$ vectors.
* [ ] **Matrix:** Build tridiagonal components $(A_j, B_j, C_j)$.
* [ ] **BCs:** Implement Dirichlet/Neumann injection into RHS.
* [ ] **Solver:** Thomas Algorithm implementation.
* [ ] **Tests:** Compare vs. Black-Scholes analytical formula.