from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

DomainMode = Literal["fixed_logmoneyness", "quantile_pad", "explicit"]


@dataclass(frozen=True, slots=True)
class DomainCheckConfig:
    """
    Defines the y-domain used for post-fit safety checks (and diagnostic reporting).

    Modes:
      - "fixed_logmoneyness": use [y_min, y_max] (stable, recommended default if y=log(K/F)).
      - "quantile_pad": robust range from data quantiles + padding (adaptive fallback).
      - "explicit": use [y_min, y_max] as an explicit domain (same as fixed, but semantically “user chose this”).

    w_floor:
      Minimum allowed total variance over the domain. Use 0.0 for strict nonnegativity.
      Use a small positive floor (e.g. 1e-10) if downstream does logs/sqrts and you want a buffer.

    tol:
      Numerical tolerance for violations: consider w < w_floor - tol a violation.

    n_grid:
      Sampling density for "domain" checks and reports.
    """

    mode: DomainMode = "fixed_logmoneyness"

    # For fixed/explicit
    y_min: float = -1.25
    y_max: float = 1.25

    # For quantile_pad
    q_lo: float = 0.01
    q_hi: float = 0.99
    pad_frac: float = 0.15
    pad_abs: float = 0.05

    # Sampling density
    n_grid: int = 41

    # Acceptance threshold
    w_floor: float = 0.0
    tol: float = 1e-12


def build_domain_grid(
    y: NDArray[np.float64], cfg: DomainCheckConfig
) -> tuple[tuple[float, float], NDArray[np.float64]]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(max(cfg.n_grid, 5))

    if cfg.mode in ("fixed_logmoneyness", "explicit"):
        y_lo = float(min(cfg.y_min, cfg.y_max))
        y_hi = float(max(cfg.y_min, cfg.y_max))
        y_chk = np.linspace(y_lo, y_hi, n, dtype=np.float64)
        return (y_lo, y_hi), y_chk

    if cfg.mode == "quantile_pad":
        if y.size == 0:
            y_lo, y_hi = 0.0, 0.0
            return (y_lo, y_hi), np.linspace(y_lo, y_hi, n, dtype=np.float64)

        # robust domain from quantiles
        if y.size >= 20:
            q_lo = float(np.clip(cfg.q_lo, 0.0, 0.49))
            q_hi = float(np.clip(cfg.q_hi, 0.51, 1.0))
            q1, q2 = np.quantile(y, [q_lo, q_hi])
        else:
            q1, q2 = float(np.min(y)), float(np.max(y))

        span = float(max(q2 - q1, 1e-6))
        pad = float(cfg.pad_frac) * span + float(cfg.pad_abs)

        y_lo = float(q1 - pad)
        y_hi = float(q2 + pad)
        y_chk = np.linspace(y_lo, y_hi, n, dtype=np.float64)
        return (y_lo, y_hi), y_chk

    raise ValueError(f"Unknown DomainCheckConfig.mode: {cfg.mode}")
