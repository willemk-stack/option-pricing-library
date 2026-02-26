"""Package-level exports for the SVI toolbox.

Convenience names are imported here so that users can access the usual
symbols via ``from option_pricing.vol.svi import ...`` even though the
implementation is now split across several submodules.
"""

from __future__ import annotations

# optimizer re-export (used by tests and calibration frontend)
from scipy.optimize import least_squares

# high-level entrypoints
from .calibrate import calibrate_svi

# diagnostics
from .diagnostics import (
    ButterflyCheck,
    GJExample51RepairResult,
    LeeWingCheck,
    SVIDiagnosticsContext,
    SVIFitDiagnostics,
    SVIModelChecks,
    SVISolverInfo,
    build_svi_diagnostics,
    check_butterfly_arbitrage,
    run_gj_example51_repair_sanity_check,
)

# domain
from .domain import DomainCheckConfig, build_domain_grid

# math utilities
from .math import (
    EPS,
    LOG2,
    RHO_MAX,
    compute_lee_wing_check,
    gatheral_g_jac_params,
    gatheral_g_scalar,
    gatheral_g_vec,
    gatheral_g_wing_limits,
    gatheral_gprime_scalar,
    gatheral_gprime_vec,
    jw_to_raw,
    raw_to_jw,
    svi_jac_w1_wrt_params,
    svi_jac_w2_wrt_params,
    svi_jac_wrt_params,
    svi_total_variance,
    svi_total_variance_dy,
    svi_total_variance_dyy,
    svi_total_variance_dyyy,
)

# models
from .models import (
    JWParams,
    SVIFitResult,
    SVIParams,
    SVISmile,
)

# objective / calibration helpers
from .objective import SVIObjective, svi_residual_vector

# regularization helpers
from .regularization import (
    SVIRegConfig,
    _robust_rhoprime,
    apply_reg_override,
    default_reg_from_data,
    soft_hinge_log_ratio,
    soft_hinge_ratio_excess,
)

# repair utilities
from .repair import (
    repair_butterfly_jw_optimal,
    repair_butterfly_raw,
    repair_butterfly_with_fallback,
)

# transforms
from .transforms import SVITransformLeeCap, logit, sigmoid, softplus, softplus_inv

# wings / slope estimation
from .wings import (
    estimate_wing_slopes_one_sided,
    usable_obs_slopes,
)

__all__ = [
    # domain
    "DomainCheckConfig",
    "build_domain_grid",
    # transforms
    "sigmoid",
    "softplus",
    "softplus_inv",
    "logit",
    "SVITransformLeeCap",
    # math
    "RHO_MAX",
    "EPS",
    "LOG2",
    "svi_total_variance",
    "svi_total_variance_dy",
    "svi_total_variance_dyy",
    "svi_total_variance_dyyy",
    "svi_jac_wrt_params",
    "svi_jac_w1_wrt_params",
    "svi_jac_w2_wrt_params",
    "gatheral_g_vec",
    "gatheral_gprime_vec",
    "gatheral_g_scalar",
    "gatheral_gprime_scalar",
    "gatheral_g_wing_limits",
    "gatheral_g_jac_params",
    "compute_lee_wing_check",
    "raw_to_jw",
    "jw_to_raw",
    # regularization
    "SVIRegConfig",
    "default_reg_from_data",
    "apply_reg_override",
    "_robust_rhoprime",
    "soft_hinge_log_ratio",
    "soft_hinge_ratio_excess",
    # wings
    "estimate_wing_slopes_one_sided",
    "usable_obs_slopes",
    # objective
    "SVIObjective",
    "svi_residual_vector",
    # repair
    "repair_butterfly_raw",
    "repair_butterfly_jw_optimal",
    "repair_butterfly_with_fallback",
    "run_gj_example51_repair_sanity_check",
    # diagnostics
    "LeeWingCheck",
    "ButterflyCheck",
    "check_butterfly_arbitrage",
    "SVISolverInfo",
    "SVIModelChecks",
    "SVIFitDiagnostics",
    "SVIDiagnosticsContext",
    "build_svi_diagnostics",
    "GJExample51RepairResult",
    # models
    "SVIParams",
    "JWParams",
    "SVIFitResult",
    "SVISmile",
    # high-level
    "least_squares",
    "calibrate_svi",
]
