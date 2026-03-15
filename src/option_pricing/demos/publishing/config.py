from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

BuildProfileName = Literal["ci", "publish"]


@dataclass(frozen=True, slots=True)
class VisualBuildConfig:
    profile: BuildProfileName
    workflow_profile: str
    surface_nt: int
    surface_ny: int
    localvol_nt: int
    localvol_ny: int
    dpi: int
    include_3d_extras: bool
    surface_y_min: float = -0.50
    surface_y_max: float = 0.50
    localvol_y_min: float = -0.30
    localvol_y_max: float = 0.30
    readme_assets_dir: Path = Path("docs") / "assets" / "generated"
    bundle_root: Path = Path("out") / "visual_bundles"

    @property
    def overrides(self) -> dict[str, Any]:
        return {
            "RUN_*": {
                # These are useful notebook demonstrations, but they are not
                # part of the canonical publishing bundle.
                "RUN_GJ_PAPER_SANITY_CHECK": False,
                "RUN_EXPLICIT_SVI_REPAIR_DEMO": False,
                "RUN_DIGITAL_PDE_BASELINE": False,
                "RUN_DUPIRE_VS_GATHERAL_COMPARE": True,
                "RUN_LOCALVOL_REPRICING": True,
                "RUN_LOCALVOL_CONVERGENCE_SWEEP": True,
                "RUN_ESSVI_TIME_SMOOTHNESS_COMPARE": True,
            }
        }


def get_visual_build_config(profile: BuildProfileName | str) -> VisualBuildConfig:
    name = str(profile).strip().lower()
    if name == "ci":
        return VisualBuildConfig(
            profile="ci",
            workflow_profile="quick",
            surface_nt=21,
            surface_ny=41,
            localvol_nt=21,
            localvol_ny=41,
            dpi=140,
            include_3d_extras=False,
        )
    if name == "publish":
        return VisualBuildConfig(
            profile="publish",
            workflow_profile="full",
            surface_nt=41,
            surface_ny=81,
            localvol_nt=31,
            localvol_ny=61,
            dpi=240,
            include_3d_extras=False,
        )
    raise ValueError("profile must be 'ci' or 'publish'")
