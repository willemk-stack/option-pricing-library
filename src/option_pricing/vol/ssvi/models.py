from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np

from ...typing import ArrayLike

type TermInput = ArrayLike | Callable[[ArrayLike], ArrayLike]
type InputVariable = Literal["T", "log_T"]

DEFAULT_NUMERICAL_TOL = 1e-12


def _as_float_array(value: ArrayLike) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _broadcast_to_query(query: ArrayLike, raw: ArrayLike) -> np.ndarray:
    query_arr = _as_float_array(query)
    raw_arr = _as_float_array(raw)
    _, broadcast = np.broadcast_arrays(query_arr, raw_arr)
    return np.asarray(broadcast, dtype=np.float64)


def _evaluate_term_input(spec: TermInput, query: np.ndarray) -> np.ndarray:
    raw = spec(query) if callable(spec) else spec
    return _broadcast_to_query(query, raw)


def _validate_positive_maturity(T: ArrayLike) -> np.ndarray:
    T_arr = _as_float_array(T)
    if np.any(~np.isfinite(T_arr)) or np.any(T_arr <= 0.0):
        raise ValueError("T must be finite and > 0.")
    return T_arr


def _safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    name: str,
    eps: float,
) -> np.ndarray:
    if np.any(~np.isfinite(denominator)):
        raise ValueError(f"{name} is undefined because the denominator is not finite.")
    if np.any(np.abs(denominator) <= eps):
        raise ValueError(
            f"{name} is undefined because the denominator is too close to zero "
            f"(abs(denominator) <= {eps:g})."
        )
    with np.errstate(divide="raise", invalid="raise"):
        return np.asarray(numerator / denominator, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class MaturityTermStructure:
    """Scalar term structure with a maturity-based API.

    The canonical external variable is always maturity ``T``. If
    ``input_variable='log_T'`` the stored callables are interpreted as functions
    of ``x = log(T)``, and the returned derivatives are converted back to
    maturity derivatives via the chain rule.
    """

    value: TermInput
    first_derivative: TermInput | None = None
    second_derivative: TermInput | None = None
    input_variable: InputVariable = "T"

    def __post_init__(self) -> None:
        if self.input_variable not in ("T", "log_T"):
            raise ValueError("input_variable must be either 'T' or 'log_T'.")

    @property
    def label(self) -> str:
        return type(self).__name__

    def _prepare(self, T: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        T_arr = _validate_positive_maturity(T)
        if self.input_variable == "log_T":
            return T_arr, np.log(T_arr)
        return T_arr, T_arr

    def __call__(self, T: ArrayLike) -> np.ndarray:
        _, query = self._prepare(T)
        return _evaluate_term_input(self.value, query)

    def dT(self, T: ArrayLike) -> np.ndarray:
        T_arr, query = self._prepare(T)
        if self.first_derivative is None:
            raise NotImplementedError(
                f"{self.label} does not define a first derivative."
            )

        raw = _evaluate_term_input(self.first_derivative, query)
        if self.input_variable == "log_T":
            return np.asarray(raw / T_arr, dtype=np.float64)
        return np.asarray(raw, dtype=np.float64)

    def d2T2(self, T: ArrayLike) -> np.ndarray:
        T_arr, query = self._prepare(T)
        if self.second_derivative is None:
            raise NotImplementedError(
                f"{self.label} does not define a second derivative."
            )

        raw = _evaluate_term_input(self.second_derivative, query)
        if self.input_variable == "log_T":
            if self.first_derivative is None:
                raise NotImplementedError(
                    f"{self.label} needs a first derivative to map d2/dx2 into d2/dT2."
                )
            first = _evaluate_term_input(self.first_derivative, query)
            return np.asarray((raw - first) / (T_arr * T_arr), dtype=np.float64)
        return np.asarray(raw, dtype=np.float64)

    @classmethod
    def constant(cls, value: float) -> Self:
        def const(arg: ArrayLike) -> np.ndarray:
            arg_arr = np.asarray(arg, dtype=np.float64)
            return np.full_like(arg_arr, float(value), dtype=np.float64)

        def zeros(arg: ArrayLike) -> np.ndarray:
            arg_arr = np.asarray(arg, dtype=np.float64)
            return np.zeros_like(arg_arr, dtype=np.float64)

        return cls(
            value=const,
            first_derivative=zeros,
            second_derivative=zeros,
            input_variable="T",
        )


class ThetaTermStructure(MaturityTermStructure):
    """Primary maturity term structure for theta(T)."""


class PsiTermStructure(MaturityTermStructure):
    """Primary maturity term structure for psi(T)."""


class EtaTermStructure(MaturityTermStructure):
    """Primary maturity term structure for eta(T)."""


@dataclass(frozen=True, slots=True)
class ESSVITermStructures:
    """Primary eSSVI maturity parametrization in theta(T), psi(T), eta(T)."""

    theta_term: ThetaTermStructure
    psi_term: PsiTermStructure
    eta_term: EtaTermStructure
    eps: float = DEFAULT_NUMERICAL_TOL

    def __post_init__(self) -> None:
        if self.eps <= 0.0:
            raise ValueError("eps must be > 0.")

    def validate(self, T: ArrayLike) -> None:
        _validate_positive_maturity(T)
        theta = self.theta(T)
        psi = self.psi(T)
        eta = self.eta(T)
        abs_eta = np.abs(eta)

        if np.any(~np.isfinite(theta)) or np.any(theta <= 0.0):
            raise ValueError("theta(T) must be finite and > 0.")
        if np.any(~np.isfinite(psi)) or np.any(psi <= 0.0):
            raise ValueError("psi(T) must be finite and > 0.")
        if np.any(~np.isfinite(eta)):
            raise ValueError("eta(T) must be finite.")
        if np.any(abs_eta >= psi):
            raise ValueError(
                "eSSVI requires |eta(T)| < psi(T), equivalently |rho(T)| < 1."
            )
        if np.any(psi + abs_eta >= 4.0 - self.eps):
            raise ValueError("eSSVI requires psi(T) + |eta(T)| < 4.")
        if np.any(psi * (psi + abs_eta) > 4.0 * theta - self.eps):
            raise ValueError(
                "eSSVI requires psi(T) * (psi(T) + |eta(T)|) <= 4 * theta(T)."
            )

    def theta(self, T: ArrayLike) -> np.ndarray:
        return np.asarray(self.theta_term(T), dtype=np.float64)

    def psi(self, T: ArrayLike) -> np.ndarray:
        return np.asarray(self.psi_term(T), dtype=np.float64)

    def eta(self, T: ArrayLike) -> np.ndarray:
        return np.asarray(self.eta_term(T), dtype=np.float64)

    def dtheta_dT(self, T: ArrayLike) -> np.ndarray:
        return np.asarray(self.theta_term.dT(T), dtype=np.float64)

    def dpsi_dT(self, T: ArrayLike) -> np.ndarray:
        return np.asarray(self.psi_term.dT(T), dtype=np.float64)

    def deta_dT(self, T: ArrayLike) -> np.ndarray:
        return np.asarray(self.eta_term.dT(T), dtype=np.float64)

    def rho(self, T: ArrayLike) -> np.ndarray:
        return _safe_divide(self.eta(T), self.psi(T), name="rho(T)", eps=self.eps)

    def phi(self, T: ArrayLike) -> np.ndarray:
        return _safe_divide(self.psi(T), self.theta(T), name="phi(T)", eps=self.eps)

    def drho_dT(self, T: ArrayLike) -> np.ndarray:
        psi = self.psi(T)
        eta = self.eta(T)
        dpsi = self.dpsi_dT(T)
        deta = self.deta_dT(T)
        numerator = deta * psi - eta * dpsi
        denominator = psi * psi
        return _safe_divide(numerator, denominator, name="drho_dT(T)", eps=self.eps)

    def dphi_dT(self, T: ArrayLike) -> np.ndarray:
        theta = self.theta(T)
        psi = self.psi(T)
        dtheta = self.dtheta_dT(T)
        dpsi = self.dpsi_dT(T)
        numerator = dpsi * theta - psi * dtheta
        denominator = theta * theta
        return _safe_divide(numerator, denominator, name="dphi_dT(T)", eps=self.eps)


@dataclass(frozen=True, slots=True)
class ESSVINodeSet:
    """Discrete eSSVI node representation on calibrated expiries."""

    expiries: np.ndarray
    theta: np.ndarray
    psi: np.ndarray
    rho: np.ndarray
    eps: float = DEFAULT_NUMERICAL_TOL

    def __post_init__(self) -> None:
        expiries = _as_float_array(self.expiries).reshape(-1)
        theta = _as_float_array(self.theta).reshape(-1)
        psi = _as_float_array(self.psi).reshape(-1)
        rho = _as_float_array(self.rho).reshape(-1)

        if expiries.size == 0:
            raise ValueError("ESSVINodeSet requires at least one expiry.")
        if not (expiries.size == theta.size == psi.size == rho.size):
            raise ValueError("expiries, theta, psi, and rho must have the same size.")
        if np.any(~np.isfinite(expiries)) or np.any(expiries <= 0.0):
            raise ValueError("expiries must be finite and > 0.")
        if np.any(np.diff(expiries) <= 0.0):
            raise ValueError("expiries must be strictly increasing.")
        if np.any(~np.isfinite(theta)) or np.any(theta <= 0.0):
            raise ValueError("theta nodes must be finite and > 0.")
        if np.any(~np.isfinite(psi)) or np.any(psi <= 0.0):
            raise ValueError("psi nodes must be finite and > 0.")
        if np.any(~np.isfinite(rho)):
            raise ValueError("rho nodes must be finite.")
        if np.any(np.abs(rho) >= 1.0 - float(self.eps)):
            raise ValueError("|rho| must be strictly less than 1 at every node.")
        if self.eps <= 0.0:
            raise ValueError("eps must be > 0.")

        object.__setattr__(self, "expiries", expiries)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "psi", psi)
        object.__setattr__(self, "rho", rho)

    @property
    def eta(self) -> np.ndarray:
        return np.asarray(self.psi * self.rho, dtype=np.float64)

    @property
    def g_plus(self) -> np.ndarray:
        return np.asarray(self.psi * (1.0 + self.rho), dtype=np.float64)

    @property
    def g_minus(self) -> np.ndarray:
        return np.asarray(self.psi * (1.0 - self.rho), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class MingoneGlobalParams:
    """Unconstrained global optimizer variables for Mingone eSSVI calibration."""

    expiries: np.ndarray
    rho_raw: np.ndarray
    theta1_raw: float
    a_raw: np.ndarray
    c_raw: np.ndarray
    rho_cap: float = 0.95
    eps: float = DEFAULT_NUMERICAL_TOL

    def __post_init__(self) -> None:
        expiries = _as_float_array(self.expiries).reshape(-1)
        rho_raw = _as_float_array(self.rho_raw).reshape(-1)
        a_raw = _as_float_array(self.a_raw).reshape(-1)
        c_raw = _as_float_array(self.c_raw).reshape(-1)

        if expiries.size == 0:
            raise ValueError("MingoneGlobalParams requires at least one expiry.")
        if np.any(~np.isfinite(expiries)) or np.any(expiries <= 0.0):
            raise ValueError("expiries must be finite and > 0.")
        if np.any(np.diff(expiries) <= 0.0):
            raise ValueError("expiries must be strictly increasing.")
        if rho_raw.size != expiries.size:
            raise ValueError("rho_raw must have the same length as expiries.")
        if c_raw.size != expiries.size:
            raise ValueError("c_raw must have the same length as expiries.")
        if a_raw.size != max(expiries.size - 1, 0):
            raise ValueError("a_raw must have length len(expiries) - 1.")
        if not np.isfinite(float(self.theta1_raw)):
            raise ValueError("theta1_raw must be finite.")
        if not np.isfinite(float(self.rho_cap)) or not (0.0 < self.rho_cap < 1.0):
            raise ValueError("rho_cap must be finite and lie in (0, 1).")
        if self.eps <= 0.0:
            raise ValueError("eps must be > 0.")

        object.__setattr__(self, "expiries", expiries)
        object.__setattr__(self, "rho_raw", rho_raw)
        object.__setattr__(self, "a_raw", a_raw)
        object.__setattr__(self, "c_raw", c_raw)

    @property
    def size(self) -> int:
        return int(self.expiries.size)
