from importlib import import_module

from option_pricing.models.heston import HestonParams

DOC_EXPORTED_NAMES: dict[str, tuple[str, ...]] = {
    "option_pricing.models.heston": (
        "HestonParams",
        "heston_probability",
        "recommend_heston_quadrature_config",
    ),
    "option_pricing.pricers.heston": (
        "heston_price_call_from_ctx",
        "heston_price_put_from_ctx",
        "heston_price_from_ctx",
        "heston_price_instrument",
        "heston_price_instrument_from_ctx",
    ),
    "option_pricing.pricers.heston_mc": (
        "heston_mc_price_from_ctx",
        "heston_mc_price",
        "heston_mc_price_instrument",
        "heston_mc_price_instrument_from_ctx",
        "heston_mc_price_path_payoff_from_ctx",
        "heston_mc_price_with_vanilla_control_from_ctx",
    ),
    "option_pricing.models.heston.calibration": (
        "HestonCalibrationBounds",
        "calibrate_heston_multistart",
        "default_heston_seed",
        "heston_seed_grid",
    ),
    "option_pricing.diagnostics.heston": (
        "HestonDiagnosticsReport",
        "build_market_like_heston_quote_set",
        "run_heston_calibration_fit_diagnostics",
        "run_heston_pricing_diagnostics",
        "run_heston_vs_local_vol_comparison",
        "summarize_bias_vs_timestep",
        "summarize_runtime_vs_error",
    ),
}


def test_heston_doc_modules_export_documented_names() -> None:
    for module_name, expected_names in DOC_EXPORTED_NAMES.items():
        module = import_module(module_name)
        missing = [name for name in expected_names if not hasattr(module, name)]
        assert (
            not missing
        ), f"{module_name} is missing documented names: {', '.join(missing)}"


def test_heston_params_support_documented_repo_names() -> None:
    params = HestonParams(kappa=1.5, vbar=0.04, eta=0.5, rho=-0.7, v=0.04)

    assert params.kappa == 1.5
    assert params.vbar == 0.04
    assert params.eta == 0.5
    assert params.rho == -0.7
    assert params.v == 0.04
