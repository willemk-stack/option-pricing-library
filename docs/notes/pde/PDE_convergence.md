These notes will cover some common methods used to handle discontinuities in payoffs when using a PDE pricer.

# Convergence remedies
When saying convergence remedies I mean methods that we can employ to re-obtain the expected rate of convergence which before was decreased due to discontinuities in methods that assume smooth funcions

## Averaging cells of Initial Conditions
In this method, nodal values are replaced with averages of surrounding values in the form:

$$f_i = \frac{1}{S_{i + \frac12} - S_{i - \frac12} \int _{S_{i - \frac12}}^{S_{i + \frac12}} f(S_i - y)dy$$

where $f$ denotes an option's payoff

### digital example?

### gauss3 rule (3-point Gauss–Legendre quadrature rule)

## Shifting The Mesh (not (yet?) implemented)
### uniform
### non-uniform / clustered

## Projecting The Initial Conditions


### references:
Pooley, D. M., Vetzal, K. R., & Forsyth, P. A. (2002, June 17). _Convergence remedies for non-smooth payoffs in option pricing_. University of Waterloo.