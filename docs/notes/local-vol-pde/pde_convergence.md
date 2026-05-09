# PDE convergence notes

!!! note "Status: provisional implementation note"
    This note tracks convergence remedies for nonsmooth payoffs in PDE pricing.
    Treat it as implementation guidance until each remedy is linked to a
    maintained validation page or focused test.

These notes cover common methods used to handle discontinuities in payoffs when using a PDE pricer.

# Convergence remedies
When saying convergence remedies I mean methods that we can employ to re-obtain the expected rate of convergence that
is otherwise lost when the payoff is not smooth.

## Averaging cells of Initial Conditions
In this method, nodal values are replaced with averages of surrounding values in the form:

$$
f_i =
\frac{1}{S_{i+\frac12} - S_{i-\frac12}}
\int_{S_{i-\frac12}}^{S_{i+\frac12}} f(y)\,dy
$$

where $f$ denotes an option's payoff

### digital example?

### gauss3 rule (3-point Gauss–Legendre quadrature rule)

## Shifting The Mesh (not (yet?) implemented)
### uniform
### non-uniform / clustered

## Projecting The Initial Conditions


## References

The note relies on the local `Finance-books` source library:

- *Convergence Remedies for Non-Smooth Payoffs in Option Pricing.pdf*
    in `04_Numerical_Methods/01_PDE_Finite_Difference`.
