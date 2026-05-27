"""Deterministic local market snapshot fixture loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import pandas as pd

from option_pricing.marketdata.validation import coerce_frame, order_columns

LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1: Final = "local_snapshot_synth_schema_v1"
_REQUIRED_FILES: Final = ("manifest.json", "market_inputs.csv", "option_chain.csv")


@dataclass(frozen=True, slots=True)
class LocalSnapshotResult:
    """Loaded schema-only local market snapshot fixture."""

    name: str
    manifest: dict[str, Any]
    market_inputs: pd.DataFrame
    option_chain: pd.DataFrame
    warnings: tuple[str, ...] = ()


class LocalSnapshotProvider:
    """Load provider-neutral synthetic local market snapshot fixtures."""

    def __init__(self, fixture_root: Path | None = None) -> None:
        self.fixture_root = (
            Path(fixture_root).expanduser().resolve()
            if fixture_root is not None
            else _default_fixture_root()
        )

    def load_snapshot(self, name: str) -> LocalSnapshotResult:
        """Load a named local snapshot fixture and validate its A1 schemas."""

        fixture_name = _validate_fixture_name(name)
        fixture_dir = self._resolve_fixture_dir(fixture_name)
        files = self._required_paths(fixture_dir, fixture_name)

        manifest = _read_manifest(files["manifest.json"])
        market_inputs = _read_schema_csv(files["market_inputs.csv"], "market_inputs")
        option_chain = _read_schema_csv(files["option_chain.csv"], "option_chain")

        return LocalSnapshotResult(
            name=fixture_name,
            manifest=manifest,
            market_inputs=market_inputs,
            option_chain=option_chain,
        )

    def _resolve_fixture_dir(self, name: str) -> Path:
        fixture_dir = (self.fixture_root / name).resolve()
        if not fixture_dir.is_dir():
            raise FileNotFoundError(
                f"Unknown local snapshot fixture {name!r} under {self.fixture_root}"
            )
        return fixture_dir

    def _required_paths(self, fixture_dir: Path, name: str) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for filename in _REQUIRED_FILES:
            path = fixture_dir / filename
            if not path.is_file():
                raise FileNotFoundError(
                    f"Local snapshot fixture {name!r} is missing required file: "
                    f"{filename}"
                )
            paths[filename] = path
        return paths


def _default_fixture_root() -> Path:
    return Path(__file__).resolve().parents[4] / "tests" / "marketdata" / "fixtures"


def _validate_fixture_name(name: str) -> str:
    cleaned = name.strip()
    if (
        not cleaned
        or cleaned in {".", ".."}
        or "/" in cleaned
        or "\\" in cleaned
        or Path(cleaned).is_absolute()
    ):
        raise ValueError("fixture name must be a simple directory name")
    return cleaned


def _read_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Local snapshot manifest at {path} must be a JSON object")

    return cast(dict[str, Any], payload)


def _read_schema_csv(path: Path, dataset_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    coerced = coerce_frame(frame, dataset_name, allow_extra=False)
    return order_columns(coerced, dataset_name)


__all__ = [
    "LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1",
    "LocalSnapshotProvider",
    "LocalSnapshotResult",
]
