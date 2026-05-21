# Heston API

<div class="doc-intro doc-intro--quiet" markdown="1">
<p class="doc-intro__kicker">Namespaced stochastic-volatility stack</p>
<p class="doc-intro__lead">Use this page when you need the Heston model objects, Fourier pricers, Monte Carlo pricers, calibration helpers, simulation primitives, and notebook-facing diagnostics in one reference map.</p>
<p class="doc-intro__support">Heston remains namespaced because it is a model stack with pricing, calibration, simulation, and diagnostics rather than a single everyday scalar pricer.</p>
</div>

## Import map

<p class="doc-section-lead">The package root intentionally stays focused on the everyday public API. Heston callers should import from the responsible namespace for the part of the stack they are using.</p>

- `option_pricing.models.heston` for parameters, Fourier probabilities, and quadrature recommendations.
- `option_pricing.pricers.heston` for semi-analytic vanilla pricing.
- `option_pricing.pricers.heston_mc` for Monte Carlo pricing wrappers.
- `option_pricing.models.heston.simulation` for lower-level path and terminal simulators.
- `option_pricing.models.heston.calibration` for calibrators, quote preflight helpers, bounds, multistart results, and seed helpers.
- `option_pricing.models.heston.calibration.heston_types` for the quote-set container used by calibration and diagnostics.
- `option_pricing.diagnostics.heston` for notebook-facing diagnostics, generated-report helpers, and model comparison.

See the [Heston guide](../user_guides/heston.md) for the workflow narrative and
the [Heston model comparison](../user_guides/heston_model_comparison.md) page
for the reviewer-facing Capstone 3 evidence path.

## Model parameters and Fourier support

<p class="doc-section-lead">These objects define the parameter container, probability integral, and quadrature-policy entrypoint used by the pricing and calibration layers.</p>

::: option_pricing.models.heston
    options:
      members:
        - HestonParams
        - heston_probability
        - recommend_heston_quadrature_config
        - heston_char_fn

## Fourier vanilla pricers

<p class="doc-section-lead">The semi-analytic pricers are the deterministic vanilla-pricing layer used by the guide, calibration objective, and Monte Carlo validation cross-checks.</p>

::: option_pricing.pricers.heston
    options:
      members:
        - heston_price_from_ctx
        - heston_price_call_from_ctx
        - heston_price_put_from_ctx
        - heston_price_instrument
        - heston_price_instrument_from_ctx
        - heston_price_and_param_jac_from_ctx
        - heston_price_call_and_param_jac_from_ctx
        - heston_price_put_and_param_jac_from_ctx

## Monte Carlo pricers

<p class="doc-section-lead">The public Monte Carlo wrappers price vanilla, terminal-payoff, or path-payoff instruments under the Heston simulators while returning the standard <code>MonteCarloResult</code>.</p>

::: option_pricing.pricers.heston_mc
    options:
      members:
        - heston_mc_price_from_ctx
        - heston_mc_price
        - heston_mc_price_call
        - heston_mc_price_put
        - heston_mc_price_instrument
        - heston_mc_price_instrument_from_ctx
        - heston_mc_price_path_payoff_from_ctx
        - heston_mc_price_with_vanilla_control_from_ctx
        - heston_vanilla_control_from_ctx

## Simulation primitives

<p class="doc-section-lead">These lower-level simulators support the Monte Carlo pricers and diagnostics. Use the pricer wrappers first unless you need direct path or terminal samples.</p>

::: option_pricing.models.heston.simulation
    options:
      members:
        - HestonTerminalSimulator
        - HestonPathSimulator
        - HestonSimulationResult
        - simulate_heston_terminal
        - simulate_heston_paths

## Calibration

<p class="doc-section-lead">Calibration is intentionally kept under the Heston model namespace. Bounds and seed helpers are optimizer safeguards and diagnostics inputs, not claims of parameter uniqueness.</p>

::: option_pricing.models.heston.calibration
    options:
      members:
        - HestonCalibrationBounds
        - HestonCalibrationRun
        - HestonMultistartResult
        - HestonQuotePreflight
        - HestonObjectiveType
        - HestonParameterTransform
        - calibrate_heston
        - calibrate_heston_multistart
        - default_heston_seed
        - heston_seed_grid
        - preflight_heston_quotes

### Quote-set container

<p class="doc-section-lead">The quote-set type carries the shared vanilla target used by calibration, fit diagnostics, and model comparison.</p>

::: option_pricing.models.heston.calibration.heston_types
    options:
      members:
        - HestonQuoteSet

## Diagnostics and model comparison

<p class="doc-section-lead">Diagnostics are notebook-facing helpers for reviewing numerical health, calibration fit, Monte Carlo behavior, synthetic benchmark coverage, and the Heston versus eSSVI/local-vol comparison workflow.</p>

::: option_pricing.diagnostics.heston
    options:
      members:
        - run_heston_pricing_diagnostics
        - run_heston_slice_diagnostics
        - compare_backend_slice
        - price_slice_with_diagnostics
        - probability_slice_with_diagnostics
        - run_heston_calibration_fit_diagnostics
        - run_heston_calibration_diagnostics
        - build_synthetic_heston_quote_set
        - run_heston_calibration_benchmark_diagnostics
        - run_heston_mc_comparison_sweep
        - compare_heston_mc_schemes
        - summarize_bias_vs_timestep
        - summarize_runtime_vs_error
        - build_market_like_heston_quote_set
        - run_heston_vs_local_vol_comparison
        - heston_calibration_quote_policy_tables

## Reading order

- Start with the [Heston guide](../user_guides/heston.md) for workflow context.
- Use [Heston diagnostics](../user_guides/heston_diagnostics.md) when interpreting report outputs.
- Use [Heston model comparison](../user_guides/heston_model_comparison.md) for the bounded model-choice story.
- Return to this page when you need exact import paths and signatures.
