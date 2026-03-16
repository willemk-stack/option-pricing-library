"""
POSSIBLE OUTLINE HERE MAYBE FOR DIFF MODULE

from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ResultStats:
    rows_in: int = 0
    rows_out: int = 0
    files_written: tuple[Path, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PipelineResult(Generic[T]):
    value: T
    metadata: RunMetadata
    stats: ResultStats = field(default_factory=ResultStats)
"""
