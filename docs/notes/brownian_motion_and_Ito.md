## **1: Definition and intuition**
### Formal definition of Brownian motion
In this Notebook, we will be working with a Brownian motion, which is a continuous-time stochastic process satisfying these properties:
Let $(\Omega,\mathcal{F}, \mathbb{P})$ be a probability space. Then for a Brownian motion $W(t)$, $t \geq 0$
1. $W_0 = 0$ Almost surely
2. $0=t_0​<t_1​<\cdots<t_m$
    $$
    W_{t_{1}}-W_{t_{0}}, W_{t_{2}}-W_{t_{1}}, \cdots, W_{t_{m}}-W_{t_{m-1}}\quad
    \text{are independent and normally distributed, satisfying:}
    $$
3. 
    $
    \mathbb{E}[W(t_{i+1})-W(t_i)] = 0, \quad \forall i 
    $
4. 
    $
    \text{Var}[W(t_{i+1})-W(t_i)] = t_{i+1} - t_i,\quad \forall i
    $

Source: Adapted from S. Shreve, *Stochastic Calculus for Finance II: Continuous-Time Models*, Definition 3.3.1.

### Intuition/Visualisation
- Scaled limit of random walk
- Continous but jagged.
    - Though the function is continuous for every t, it is not differentiable as it is not a smooth function.
- The brownian motion has no tendency to go up or down, see def 3, the Exp val of W(s) at time t = s is W(s). This is not to say that Brownian motion will always converge to 0 but if it attains W(s) the Exp val of W(s) at time t = s is W(s)
- 

### Relevance to Finance

Brownian motion is fundamental to quantitative finance. It is the canonical model for continuous-time randomness and is the process on which Itô calculus is first developed. Because Brownian motion is the continuous-time limit of random walks, it presents a natural way to model a stock's random price fluctuations.

In the Black–Scholes–Merton model, the (discounted) asset price is driven by Brownian motion:
$$
dS_t = \mu S_t dt + \sigma S_t dW_t
$$
Here the drift term ($\mu S_t dt$) represents the instantaneous expected growth, while the Brownian term $\sigma S_t dW_t$ models volatitly.

Under the **risk-neutral measure**, we adjust the drift so that the **discounted price** $e^{-rt}S_t$ becomes a martingale. Intuitively, this means that, once you remove the risk-free rate, there is no predictable way to make excess profit: the “fair” expected return of the asset is just the risk-free rate. Brownian motion is still the source of randomness, but the drift changes, and this is what ultimately leads to the Black–Scholes option pricing formula, which we will see in a later notebook _{insert notebook}_.


## Ito example
In this section I will give an example how Ito's calculus is used as a tool. We will return to more of the math heavy stuff in a later notebook to derive the BSM. For now let us give a simple example.

Given $f(t,X)$, where $X=X(t)$ is a random adapted (mb give def) process. Ito's lemma gives the following rule for the total derivative $d(f(t,x))$
$$d(f(t,X)) = f_{x}(t,X)dX + f_t(t,X)dt + \frac{1}{2}f_{xx}(t,X)dXdX \quad(*)$$

For a Brownian motion $(W(t))_{t \geq 0}$ as defined previously, the following multiplication rules apply:
$$
dWdt = dtdW = dtdt = 0,\quad dWdW = dt
$$
Now, let us look at $f(t, W(t)) = (W(t))^2$ and find its total derivative using
Itô's lemma.

For $f(t,x) = x^2$ xe have
$$
f_t(t,x) = 0, \qquad f_x(t,x) = 2x, \qquad f_{xx}(t,x) = 2.
$$

Applying Itô's lemma with $X(t) = W(t)$ gives
\begin{align*}
d(W(t)^2)
&= f_t\,dt + f_w\,dW(t) + \tfrac{1}{2} f_{ww}\,(dW(t))^2 \\
&= 0\cdot dt + 2W(t)\,dW(t) + \tfrac{1}{2}\cdot 2\,(dW(t))^2.
\end{align*}
Using $(dW(t))^2 = dt$, this simplifies to
$$
d(W(t)^2) = 2W(t)\,dW(t) + dt.
$$

Integrating from $0$ to $t$ we obtain
$$
W(t)^2 - W(0)^2 = 2\int_0^t W(s)\,dW(s) + \int_0^t ds,
$$
so
$$
W(t)^2 = W(0)^2 + 2\int_0^t W(s)\,dW(s) + t.
$$

For a standard Brownian motion $W(0) = 0$, hence
$$
W(t)^2 = 2\int_0^t W(s)\,dW(s) + t.
$$

Let us analyze the expectation value of $W(t)^2$. Define
$$
I(t) := \int_0^t W(s)\,dW(s).
$$
Since $I(t)$ is an Itô integral with adapted integrand, it is a martingale and
$I(0)=0$. Therefore
$$
\mathbb{E}[I(t)] = \mathbb{E}[I(0)] = 0.
$$

Taking expectations in the identity for $W(t)^2$ gives
$$
\mathbb{E}[W(t)^2]
= 2\,\mathbb{E}[I(t)] + t
= 2\cdot 0 + t
= t.
$$
So we recover the familiar result $\mathbb{E}[W(t)^2] = t$.

Lets see if our theory holds up by simulating paths for $W(t)^2$, and computing the mean over time.