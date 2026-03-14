"""Demo recipes and helpers."""

from .capstone2 import Capstone2Artifacts, run_capstone2
from .capstone2_demo_helpers import DemoDashboard
from .essvi_bridge import ESSVIBridgeArtifacts, build_essvi_bridge_artifacts
from .integration import (
    SurfaceToLocalVolPDEIntegrationArtifacts,
    run_surface_to_localvol_pde_integration,
)
from .localvol_pde_flagship import (
    LocalVolPDEDemoArtifacts,
    build_localvol_pde_demo_artifacts,
)
from .scenario import SharedDemoScenario, build_shared_demo_scenario
from .surface_flagship import SurfaceDemoArtifacts, build_surface_demo_artifacts

__all__ = [
    "Capstone2Artifacts",
    "DemoDashboard",
    "ESSVIBridgeArtifacts",
    "LocalVolPDEDemoArtifacts",
    "SharedDemoScenario",
    "SurfaceDemoArtifacts",
    "SurfaceToLocalVolPDEIntegrationArtifacts",
    "build_essvi_bridge_artifacts",
    "build_localvol_pde_demo_artifacts",
    "build_shared_demo_scenario",
    "build_surface_demo_artifacts",
    "run_capstone2",
    "run_surface_to_localvol_pde_integration",
]
