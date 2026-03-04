# Fritsch-Carlson

Interpolating **total variance** is exactly the place where that paper’s idea is most useful — but mainly **in the maturity (T) direction**, not as a full “surface interpolator”.

### Where interpolating fits really well (and is worth implementing)

For each fixed strike/log-moneyness (k), **calendar no-arbitrage** requires total variance

$$
w(k,T)=\sigma_{\text{impl}}(k,T)^2,T
$$

to be **nondecreasing in (T)**.

A monotone piecewise cubic Hermite interpolant (the paper’s method) is a great building block here because it:

* preserves monotonicity (no overshoot if your input is monotone),
* is **local** (changing one quote only affects nearby intervals),
* is simple and robust as a fallback. 

**Practical pattern for your library**

1. Convert quotes → (w(k,T)).
2. For each (k) (or each pillar strike in your grid), sort by (T).
3. Enforce monotonicity if needed (tiny violations happen from noise): e.g. clamp / isotonic regression so (w(T_{i+1})\ge w(T_i)).
4. Run monotone cubic (PCHIP/Fritsch–Carlson style) in (T).

That gives you a clean, reusable **term-structure interpolator** that respects calendar monotonicity by construction (after step 3).

### Where it’s *not* the right fallback

Across **strike** at fixed (T), the key constraint is **butterfly/convexity** (equivalently nonnegative risk-neutral density). Monotone cubic interpolation doesn’t enforce convexity, and in fact vol smiles are typically **not monotone** in strike anyway.

### Local volatility note (important)

Even if you interpolate (w) nicely in (T), **Dupire local vol is sensitive to second derivatives in strike**. Piecewise cubic Hermite methods are (C^1) but generally not (C^2) at knots, so curvature can jump — which can show up as noisy local vol if you differentiate the interpolant.

**Good compromise**:

* Use this monotone cubic **only for the (T)-direction** (calendar).
* Use **SVI/SSVI** (your main plan) or another convexity-aware method for the smile in (k).
* As a true “oh-no fallback” per maturity slice, a safer arbitrage fallback is often **convex interpolation of option prices** (even piecewise linear after projecting quotes to convexity), rather than a cubic on vols.

### A clean “fallback stack” that works in practice

* **Primary**: fit SVI/SSVI (smile arbitrage handled if you enforce known conditions).
* **Secondary**: if smile fit fails at some maturities, use a convexity-safe price-based interpolation on that slice.
* **Everywhere**: for term structure at fixed (k), interpolate **total variance in (T)** with monotone cubic.


# Alternatives to look into
If you want a **fallback that is arbitrage-safe across strike**, the right mental model is:

> don’t interpolate vol; **fit a call-price curve (C(K))** that is **decreasing and convex** in (K), then convert back to IV if you need it.

That convexity is exactly “no butterfly arbitrage” in 1D.

Below are methods in roughly increasing “takes you to the next level” order.

---

## 1) Convex piecewise-linear call-price interpolation (LP projection)

**What it is:** You take market quotes (convert to undiscounted call prices), then **project** them onto the set of curves that satisfy:

* (C(K)) is **nonincreasing** in (K) (vertical spread ≥ 0)
* (C(K)) is **convex** in (K) (butterfly ≥ 0)
* and the usual bounds like ( \max(F-K,0) \le C(K) \le F )

On a discrete strike grid, these become **linear inequality constraints**, so the projection is a **linear program** (or a QP if you do least-squares).

**Why it’s good as a fallback:** It’s extremely robust, simple to reason about, and will *never* create negative density.

**Tradeoff:** Piecewise-linear (C(K)) implies a density that’s a sum of spikes between knots (because the second derivative is “too discrete”). Great for “safe,” not great for smooth local vol.

---

## 2) Constrained smoothing splines / B-splines on prices (QP)

**What it is:** Represent (C(K)) as a spline basis (often B-splines), then solve a **quadratic program**:

* fit market prices in least squares,
* add a smoothness penalty (e.g., integrated squared curvature),
* enforce **convexity** by requiring the spline’s second derivative ≥ 0 on a grid,
* enforce monotonicity (first derivative ≤ 0).

**Why it “levels up”:** You get a curve that is still arbitrage-safe but **much smoother** than piecewise-linear, which matters if you later differentiate (densities, Dupire inputs, etc.).

**Tradeoff:** More engineering (QP solver, constraint discretization).

A classic reference thread here is the work around arbitrage-free smoothing of the IV surface (e.g., Dietmar Pfaffel / Rüdiger Kiesel style approaches), and especially Matthias Fengler’s treatment in his book on implied volatility surface modeling.

---

## 3) Density-based interpolation (piecewise-constant or piecewise-linear density)

**What it is:** Instead of fitting (C(K)) directly, you fit the **risk-neutral density** (q(K)\ge 0) (or cumulative distribution), then compute prices via integration.

Two practical variants:

* **Piecewise-constant density** between strikes → (C(K)) becomes piecewise-quadratic.
* **Piecewise-linear density** → (C(K)) becomes piecewise-cubic (but with guaranteed (q\ge0)).

**Why it’s strong:** Positivity is built-in, and you can target *exactly* what causes arbitrage (density negativity).

**Tradeoff:** You need careful handling of wings / normalization and sometimes a bit more math (but very doable).

---

## 4) Mohamed Kahalé’s arbitrage-free smile interpolation (standout)

**What it is:** A well-known construction that produces an **arbitrage-free interpolation of the smile** (typically starting from call prices across strikes) with a structure that preserves no-butterfly constraints by design.

**Why it stands out:** It’s “clever-advanced” and purpose-built for *exactly your use case*: a robust smile interpolator that doesn’t invent arbitrage.

**Tradeoff:** Implementation is more specialized than “just solve a QP,” but once you do it, it’s a great reusable library component.

---

## 5) “Surface-level” advanced: Andreasen–Huge style calibration (next-level)

**What it is:** You calibrate a **(piecewise) local volatility** model directly to option prices across strikes and maturities via a PDE/FD solver, often with regularization.

**Why it’s next level:** When done well, you get a surface that is smooth, stable, and consistent in both (K) and (T)—very useful if you ultimately care about local vol.

**Tradeoff:** Heavier machinery (PDE grid, stability, speed). This is less of a “fallback” and more of a “second engine.”

---

# How this ties to your “total variance + SVI/SSVI” plan

A really effective architecture is:

1. **Primary:** SVI/SSVI fit (fast, interpretable).
2. **Backup smile per maturity:** **arbitrage-safe price interpolation** (methods 1–4).
3. **Term structure at fixed (k):** monotone interpolation of **total variance in (T)** (your earlier idea).

That gives you: robust fallback + calendar sanity + good library reuse.

---

## If you want one “best” fallback to implement first

If your goal is “strong and reusable” without going full PDE:

**Constrained B-spline/QP on call prices** (method 2) is the sweet spot:

* arbitrage-safe by constraints,
* smooth enough for downstream numerics,
* general (works for any asset, any quote set).

---

If you tell me how your quotes come in (delta-quoted vs strike) and whether you’re working in (K) or log-moneyness (k=\log(K/F)), I can recommend a concrete constraint set + objective (LP vs QP, which basis, what wing conditions) that fits your library design.
