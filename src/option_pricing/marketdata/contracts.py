from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ResultStats:
    rows_in: int = 0
    rows_out: int = 0
    files_written: tuple[Path, ...] = ()
    warnings: tuple[str, ...] = ()
