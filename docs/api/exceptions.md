# Exceptions

<div class="doc-intro" markdown="1">
<p class="doc-intro__kicker">Error types</p>
<p class="doc-intro__lead">These exceptions cover invalid prices, missing brackets, root-finding failures, and convergence breakdowns in the numerical helpers.</p>
<p class="doc-intro__support">Catch these when you want to distinguish input validation failures from numerical solver failures.</p>
</div>

## Exception reference

<p class="doc-section-lead">The module members below separate validation errors from solver and convergence failures so callers can catch the right class at the boundary.</p>

::: option_pricing.exceptions
    options:
      members:
        - InvalidOptionPriceError
        - RootFindingError
        - NoConvergenceError
        - NotBracketedError
        - NoBracketError
        - DerivativeTooSmallError
