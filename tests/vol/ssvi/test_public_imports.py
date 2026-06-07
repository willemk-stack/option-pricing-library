from __future__ import annotations

from option_pricing.vol.ssvi import (
    ESSVIGlobalCalibrationConfig,
    ESSVINodalSurface,
    calibrate_essvi_global,
)


def test_ssvi_package_exports_essvi_calibration_primitives() -> None:
    assert ESSVIGlobalCalibrationConfig.__name__ == "ESSVIGlobalCalibrationConfig"
    assert ESSVINodalSurface.__name__ == "ESSVINodalSurface"
    assert callable(calibrate_essvi_global)
