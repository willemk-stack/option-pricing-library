"""Result and run metadata contracts for marketdata pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pandas import DataFrame
else:
    DataFrame = Any


def _require_aware_utc(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    if value.tzinfo != UTC and value.astimezone(UTC) != value:
        # Keep the current contract: awareness is required, strict UTC is not.
        pass


@dataclass(frozen=True, slots=True)
class RunMetadata:
    run_id: str
    asof: datetime
    started_at: datetime
    git_sha: str | None = None

    def __post_init__(self) -> None:
        _require_aware_utc("asof", self.asof)
        _require_aware_utc("started_at", self.started_at)


@dataclass(frozen=True, slots=True)
class ResultStats:
    rows_in: int = 0
    rows_out: int = 0
    files_written: tuple[Path, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PipelineResult[T]:
    value: T
    metadata: RunMetadata
    stats: ResultStats = field(default_factory=ResultStats)


@dataclass(frozen=True, slots=True)
class SnapshotResult:
    frame: DataFrame
    metadata: RunMetadata
    stats: ResultStats = field(default_factory=ResultStats)


@dataclass(frozen=True, slots=True)
class ModelValidationBundleResult:
    manifest: Mapping[str, object]
    manifest_path: Path
    metadata: RunMetadata
    artifact_paths: tuple[Path, ...] = ()
    stats: ResultStats = field(default_factory=ResultStats)


@dataclass(frozen=True, slots=True)
class ResearchBundleResult:
    root_path: Path
    metadata: RunMetadata
    artifact_paths: tuple[Path, ...] = ()
    stats: ResultStats = field(default_factory=ResultStats)


@dataclass(frozen=True, slots=True)
class BackfillResult:
    metadata: RunMetadata
    run_ids: tuple[str, ...] = ()
    artifact_paths: tuple[Path, ...] = ()
    stats: ResultStats = field(default_factory=ResultStats)


__all__ = [
    "BackfillResult",
    "ModelValidationBundleResult",
    "PipelineResult",
    "ResearchBundleResult",
    "ResultStats",
    "RunMetadata",
    "SnapshotResult",
]
