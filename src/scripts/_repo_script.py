from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def load_repo_script(module_name: str, script_stem: str) -> None:
    """Execute a repo-root script module into the current import target."""
    script_path = _REPO_SCRIPTS_DIR / f"{script_stem}.py"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load repo script module from {script_path}")

    module = sys.modules.get(module_name)
    if module is None:
        raise ImportError(f"Import target {module_name!r} is not registered")

    module.__file__ = str(script_path)
    module.__package__ = module_name.rpartition(".")[0]
    module.__loader__ = spec.loader
    module.__spec__ = spec
    spec.loader.exec_module(module)
