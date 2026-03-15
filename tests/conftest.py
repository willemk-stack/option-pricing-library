"""Pytest helpers for the option_pricing library."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from option_pricing.types import MarketData, OptionSpec, OptionType, PricingInputs

# Ensure the src-layout package is importable when running tests from a source checkout.
# (CI / local runs often do `pytest` without `pip install -e .`.)
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))


@pytest.fixture
def base_params() -> dict:
    """A small set of canonical parameters used across tests."""
    return {
        "S": 100.0,
        "K": 100.0,
        "r": 0.05,
        "q": 0.0,
        "sigma": 0.2,
        "T": 1.0,
        "t": 0.0,
    }


@pytest.fixture
def make_inputs():
    """Factory fixture for constructing the library's PricingInputs."""

    def _make(
        *,
        S: float,
        K: float,
        r: float,
        q: float = 0.0,
        sigma: float,
        T: float,
        t: float = 0.0,
        kind: OptionType = OptionType.CALL,
    ) -> PricingInputs:
        spec = OptionSpec(kind=kind, strike=K, expiry=T)
        market = MarketData(spot=S, rate=r, dividend_yield=q)
        return PricingInputs(spec=spec, market=market, sigma=sigma, t=t)

    return _make


@pytest.fixture
def rng():
    """Seeded RNG factory."""

    def _rng(seed: int) -> np.random.Generator:
        return np.random.default_rng(seed)

    return _rng


@pytest.fixture(scope="session")
def quick_demo_workflow_overrides() -> dict[str, object]:
    return {
        "RUN_*": {
            "RUN_EXPLICIT_SVI_REPAIR_DEMO": False,
            "RUN_GJ_PAPER_SANITY_CHECK": False,
            "RUN_DIGITAL_PDE_BASELINE": False,
            "RUN_LOCALVOL_REPRICING": False,
            "RUN_LOCALVOL_CONVERGENCE_SWEEP": False,
            "RUN_DUPIRE_VS_GATHERAL_COMPARE": False,
        }
    }


@pytest.fixture(scope="session")
def quick_demo_scenario(quick_demo_workflow_overrides: dict[str, object]):
    from option_pricing.demos import build_shared_demo_scenario

    return build_shared_demo_scenario(
        profile="quick",
        seed=7,
        overrides=quick_demo_workflow_overrides,
    )


@pytest.fixture(scope="session")
def quick_demo_surface_artifacts(
    quick_demo_scenario,
    quick_demo_workflow_overrides: dict[str, object],
):
    from option_pricing.demos import build_surface_demo_artifacts

    return build_surface_demo_artifacts(
        profile="quick",
        seed=7,
        overrides=quick_demo_workflow_overrides,
        scenario=quick_demo_scenario,
    )


@pytest.fixture(scope="session")
def quick_demo_bridge_artifacts(
    quick_demo_scenario,
    quick_demo_surface_artifacts,
    quick_demo_workflow_overrides: dict[str, object],
):
    from option_pricing.demos import build_essvi_bridge_artifacts

    return build_essvi_bridge_artifacts(
        profile="quick",
        seed=7,
        overrides=quick_demo_workflow_overrides,
        scenario=quick_demo_scenario,
        surface_artifacts=quick_demo_surface_artifacts,
    )


@pytest.fixture(scope="session")
def quick_demo08_case(quick_demo_scenario, quick_demo_bridge_artifacts):
    strikes = np.asarray(quick_demo_scenario.cfg["SHARED_STRIKES"], dtype=np.float64)[
        4:-4:4
    ]
    expiries = np.asarray(quick_demo_scenario.cfg["SHARED_EXPIRIES"], dtype=np.float64)[
        1::2
    ]
    return quick_demo_scenario, quick_demo_bridge_artifacts, strikes, expiries


@pytest.fixture(scope="session")
def quick_smoothed_localvol_workflow(quick_demo_scenario, quick_demo_bridge_artifacts):
    return SimpleNamespace(
        shared=quick_demo_scenario,
        smoothed_surface=quick_demo_bridge_artifacts.smoothed_surface,
        localvol=quick_demo_bridge_artifacts.localvol,
        solver_cfg=quick_demo_scenario.cfg["LV_PDE_SOLVER_CFG"],
    )
