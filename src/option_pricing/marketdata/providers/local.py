"""Deterministic local market snapshot fixture loader."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import pandas as pd

from option_pricing.marketdata.schemas import (
    MARKET_INPUTS_SCHEMA_VERSION,
    OPTION_CHAIN_SCHEMA_VERSION,
)
from option_pricing.marketdata.validation import coerce_frame, order_columns

LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1: Final = "local_snapshot_synth_schema_v1"
_REQUIRED_FILES: Final = ("manifest.json", "market_inputs.csv", "option_chain.csv")
_REQUIRED_MANIFEST_KEYS: Final = (
    "asof",
    "datasets",
    "files",
    "fixture_name",
    "fixture_schema_version",
    "provider_neutral",
    "scope",
    "source",
    "synthetic",
    "underlying",
)
_EXPECTED_MANIFEST_FILES: Final = {
    "market_inputs": "market_inputs.csv",
    "option_chain": "option_chain.csv",
}
_EXPECTED_DATASET_VERSIONS: Final = {
    "market_inputs": MARKET_INPUTS_SCHEMA_VERSION,
    "option_chain": OPTION_CHAIN_SCHEMA_VERSION,
}
_OPTION_CHAIN_SORT_COLUMNS: Final = (
    "expiry",
    "strike",
    "right",
    "contract_symbol",
)


@dataclass(frozen=True, slots=True)
class LocalSnapshotConfig:
    """Configuration for loading one local snapshot fixture."""

    fixture_root: Path | None = None
    fixture_name: str | None = None
    fixture_path: Path | None = None
    expected_underlying: str | None = None
    run_id: str | None = None


@dataclass(frozen=True, slots=True, eq=False)
class LocalSnapshotResult:
    """Loaded schema-only local market snapshot fixture."""

    fixture_name: str
    snapshot_id: str
    run_id: str | None
    underlying: str
    asof: pd.Timestamp
    manifest: dict[str, Any]
    market_inputs_raw: pd.DataFrame
    option_chain_raw: pd.DataFrame
    metadata: dict[str, Any]
    row_counts: dict[str, int]
    warnings: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        """Backward-compatible alias for ``fixture_name``."""

        return self.fixture_name

    @property
    def market_inputs(self) -> pd.DataFrame:
        """Backward-compatible alias for ``market_inputs_raw``."""

        return self.market_inputs_raw

    @property
    def option_chain(self) -> pd.DataFrame:
        """Backward-compatible alias for ``option_chain_raw``."""

        return self.option_chain_raw

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LocalSnapshotResult):
            return False

        return (
            self.fixture_name == other.fixture_name
            and self.snapshot_id == other.snapshot_id
            and self.run_id == other.run_id
            and self.underlying == other.underlying
            and self.asof == other.asof
            and self.manifest == other.manifest
            and self.metadata == other.metadata
            and self.row_counts == other.row_counts
            and self.warnings == other.warnings
            and self.market_inputs_raw.equals(other.market_inputs_raw)
            and self.option_chain_raw.equals(other.option_chain_raw)
        )


class LocalSnapshotProvider:
    """Load provider-neutral synthetic local market snapshot fixtures."""

    def __init__(self, config: LocalSnapshotConfig | Path | None = None) -> None:
        self.config = _coerce_config(config)
        self.fixture_root = (
            Path(self.config.fixture_root).expanduser().resolve()
            if self.config.fixture_root is not None
            else _default_fixture_root()
        )

    def load_snapshot(self, name: str | None = None) -> LocalSnapshotResult:
        """Load a named local snapshot fixture and validate its A1 schemas."""

        fixture_dir, expected_fixture_name = self._resolve_fixture_dir(name)
        fixture_label = expected_fixture_name or fixture_dir.name
        files = self._required_paths(fixture_dir, fixture_label)

        manifest = _read_manifest(files["manifest.json"])
        _validate_manifest(
            manifest,
            path=files["manifest.json"],
            expected_fixture_name=expected_fixture_name,
        )

        market_inputs = _read_schema_csv(
            files["market_inputs.csv"],
            "market_inputs",
        )
        option_chain = _read_schema_csv(
            files["option_chain.csv"],
            "option_chain",
            sort_by=_OPTION_CHAIN_SORT_COLUMNS,
        )
        fixture_name = _manifest_text(manifest, "fixture_name")
        underlying = _validate_underlying(
            manifest=manifest,
            market_inputs=market_inputs,
            option_chain=option_chain,
            expected_underlying=self.config.expected_underlying,
            fixture_name=fixture_name,
        )
        _validate_market_inputs_row_count(market_inputs, fixture_name=fixture_name)
        _validate_unique_contract_symbols(option_chain, fixture_name=fixture_name)
        asof = _snapshot_asof(market_inputs)
        snapshot_id = _snapshot_id(
            fixture_name=fixture_name,
            underlying=underlying,
            asof=asof,
        )
        run_id = _clean_optional_text(self.config.run_id, "run_id")

        return LocalSnapshotResult(
            fixture_name=fixture_name,
            snapshot_id=snapshot_id,
            run_id=run_id,
            underlying=underlying,
            asof=asof,
            manifest=manifest,
            market_inputs_raw=market_inputs,
            option_chain_raw=option_chain,
            metadata={
                "fixture_path": fixture_dir.as_posix(),
                "source": _manifest_text(manifest, "source"),
                "scope": _manifest_text(manifest, "scope"),
                "synthetic": bool(manifest["synthetic"]),
                "provider_neutral": bool(manifest["provider_neutral"]),
                "files": {
                    "manifest": "manifest.json",
                    **_EXPECTED_MANIFEST_FILES,
                },
            },
            row_counts={
                "market_inputs": int(len(market_inputs)),
                "option_chain": int(len(option_chain)),
            },
        )

    def _resolve_fixture_dir(self, name: str | None) -> tuple[Path, str | None]:
        if name is not None:
            fixture_name = _validate_fixture_name(name)
            return self._resolve_named_fixture_dir(fixture_name), fixture_name

        if self.config.fixture_path is not None:
            fixture_dir = Path(self.config.fixture_path).expanduser().resolve()
            if not fixture_dir.is_dir():
                raise FileNotFoundError(
                    "Local snapshot fixture path does not exist or is not a "
                    f"directory: {fixture_dir}"
                )
            expected_name = (
                _validate_fixture_name(self.config.fixture_name)
                if self.config.fixture_name is not None
                else None
            )
            return fixture_dir, expected_name

        fixture_name = _validate_fixture_name(
            self.config.fixture_name or LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1
        )
        return self._resolve_named_fixture_dir(fixture_name), fixture_name

    def _resolve_named_fixture_dir(self, name: str) -> Path:
        fixture_dir = (self.fixture_root / name).resolve()
        _ensure_relative_to(fixture_dir, self.fixture_root)
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


def _coerce_config(config: LocalSnapshotConfig | Path | None) -> LocalSnapshotConfig:
    if config is None:
        return LocalSnapshotConfig()
    if isinstance(config, LocalSnapshotConfig):
        return config
    return LocalSnapshotConfig(fixture_root=Path(config))


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


def _ensure_relative_to(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("fixture name must stay under fixture_root") from exc


def _read_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Local snapshot manifest at {path} must be a JSON object")

    return cast(dict[str, Any], payload)


def _read_schema_csv(
    path: Path,
    dataset_name: str,
    *,
    sort_by: tuple[str, ...] = (),
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    coerced = coerce_frame(frame, dataset_name, allow_extra=False)
    ordered = order_columns(coerced, dataset_name)
    if sort_by:
        ordered = ordered.sort_values(list(sort_by), kind="mergesort")
    return ordered.reset_index(drop=True)


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    path: Path,
    expected_fixture_name: str | None,
) -> None:
    missing = [key for key in _REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise ValueError(
            f"Local snapshot manifest at {path} is missing required keys: {missing}"
        )

    fixture_name = _manifest_text(manifest, "fixture_name")
    if expected_fixture_name is not None and fixture_name != expected_fixture_name:
        raise ValueError(
            f"Local snapshot manifest fixture_name {fixture_name!r} does not match "
            f"requested fixture {expected_fixture_name!r}"
        )

    files_section = _manifest_mapping(manifest, "files")
    actual_files = {str(key): str(value) for key, value in files_section.items()}
    if actual_files != _EXPECTED_MANIFEST_FILES:
        raise ValueError(
            "Local snapshot manifest files section must match actual fixture files: "
            f"expected {_EXPECTED_MANIFEST_FILES!r}, got {actual_files!r}"
        )

    datasets_section = _manifest_mapping(manifest, "datasets")
    actual_datasets = {str(key): str(value) for key, value in datasets_section.items()}
    if actual_datasets != _EXPECTED_DATASET_VERSIONS:
        raise ValueError(
            "Local snapshot manifest datasets section must match schema versions: "
            f"expected {_EXPECTED_DATASET_VERSIONS!r}, got {actual_datasets!r}"
        )

    _manifest_text(manifest, "underlying")
    pd.Timestamp(_manifest_text(manifest, "asof"))


def _manifest_mapping(
    manifest: dict[str, Any],
    key: str,
) -> Mapping[str, object]:
    value = manifest[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"Local snapshot manifest field {key!r} must be an object")
    return value


def _manifest_text(manifest: dict[str, Any], key: str) -> str:
    value = manifest[key]
    if not isinstance(value, str):
        raise ValueError(f"Local snapshot manifest field {key!r} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(
            f"Local snapshot manifest field {key!r} must be a non-empty string"
        )
    return cleaned


def _clean_optional_text(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{name} must be a non-empty string when provided")
    return cleaned


def _validate_underlying(
    *,
    manifest: dict[str, Any],
    market_inputs: pd.DataFrame,
    option_chain: pd.DataFrame,
    expected_underlying: str | None,
    fixture_name: str,
) -> str:
    manifest_underlying = _manifest_text(manifest, "underlying")
    expected = _clean_optional_text(expected_underlying, "expected_underlying")
    if expected is not None and manifest_underlying != expected:
        raise ValueError(
            f"Local snapshot fixture {fixture_name!r} has manifest underlying "
            f"{manifest_underlying!r}; expected {expected!r}"
        )

    market_underlyings = _unique_text_values(market_inputs, "underlying")
    chain_underlyings = _unique_text_values(option_chain, "underlying")

    if market_underlyings != {manifest_underlying}:
        raise ValueError(
            f"Local snapshot fixture {fixture_name!r} market_inputs underlying "
            f"{sorted(market_underlyings)!r} does not match manifest underlying "
            f"{manifest_underlying!r}"
        )

    if chain_underlyings != {manifest_underlying}:
        raise ValueError(
            f"Local snapshot fixture {fixture_name!r} option_chain underlying "
            f"{sorted(chain_underlyings)!r} does not match manifest underlying "
            f"{manifest_underlying!r}"
        )

    return manifest_underlying


def _unique_text_values(frame: pd.DataFrame, column: str) -> set[str]:
    return set(frame[column].astype("string").dropna().astype(str).unique().tolist())


def _validate_market_inputs_row_count(
    market_inputs: pd.DataFrame,
    *,
    fixture_name: str,
) -> None:
    if len(market_inputs) != 1:
        raise ValueError(
            f"Local snapshot fixture {fixture_name!r} must contain exactly one "
            f"market_inputs row; found {len(market_inputs)}"
        )


def _validate_unique_contract_symbols(
    option_chain: pd.DataFrame,
    *,
    fixture_name: str,
) -> None:
    duplicated = option_chain["contract_symbol"].duplicated(keep=False)
    if not bool(duplicated.any()):
        return

    symbols = sorted(
        option_chain.loc[duplicated, "contract_symbol"]
        .astype("string")
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    raise ValueError(
        f"Local snapshot fixture {fixture_name!r} has duplicate contract_symbol "
        f"values: {symbols}"
    )


def _snapshot_asof(market_inputs: pd.DataFrame) -> pd.Timestamp:
    return cast(pd.Timestamp, market_inputs["asof"].iloc[0])


def _snapshot_id(
    *,
    fixture_name: str,
    underlying: str,
    asof: pd.Timestamp,
) -> str:
    return f"{fixture_name}:{underlying}:{asof.isoformat()}"


__all__ = [
    "LOCAL_SNAPSHOT_SYNTH_SCHEMA_V1",
    "LocalSnapshotConfig",
    "LocalSnapshotProvider",
    "LocalSnapshotResult",
]
