from dataclasses import dataclass

from ..params import HestonParams
from .heston_types import HestonObjectiveType, HestonParameterTransform


@dataclass(frozen=True, slots=True)
class HestonCalibrationRun:
    seed_index: int
    seed_params: HestonParams
    fitted_params: HestonParams | None
    success: bool
    cost: float
    optimality: float
    nfev: int
    njev: int | None
    message: str


@dataclass(frozen=True, slots=True)
class HestonCalibrationMultiStartResult:
    best_params: HestonParams
    best_run: HestonCalibrationRun
    runs: tuple[HestonCalibrationRun, ...]
    objective_type: HestonObjectiveType
    parameter_transform: HestonParameterTransform
