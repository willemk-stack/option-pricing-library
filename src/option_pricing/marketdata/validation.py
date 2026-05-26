"""Validation and coercion helpers for marketdata DataFrame contracts."""

from __future__ import annotations

import pandas as pd

from .schemas import (
    DATASET_COLUMNS,
    DATASET_DTYPES,
    DatasetName,
    PandasSchemaDtype,
    parse_dataset_name,
)


def dataset_columns(dataset_name: DatasetName | str) -> tuple[str, ...]:
    """Return required canonical columns for a marketdata dataset."""

    parsed_name = parse_dataset_name(dataset_name)
    return DATASET_COLUMNS[parsed_name]


def dataset_dtypes(dataset_name: DatasetName | str) -> dict[str, PandasSchemaDtype]:
    """Return expected pandas dtypes for a marketdata dataset."""

    parsed_name = parse_dataset_name(dataset_name)
    return DATASET_DTYPES[parsed_name]


def validate_columns(
    frame: pd.DataFrame,
    dataset_name: DatasetName | str,
    *,
    allow_extra: bool = True,
) -> None:
    """Validate that a DataFrame has the required columns for a dataset."""

    parsed_name = parse_dataset_name(dataset_name)
    required = dataset_columns(parsed_name)
    actual = tuple(frame.columns)

    missing = [column for column in required if column not in actual]

    if missing:
        raise ValueError(f"{parsed_name.value} is missing required columns: {missing}")

    if not allow_extra:
        extra = [column for column in actual if column not in required]

        if extra:
            raise ValueError(
                f"{parsed_name.value} has unexpected extra columns: {extra}"
            )


def validate_dtypes(
    frame: pd.DataFrame,
    dataset_name: DatasetName | str,
    *,
    allow_extra: bool = True,
) -> None:
    """Validate that a DataFrame has the expected pandas dtypes for a dataset."""

    parsed_name = parse_dataset_name(dataset_name)

    validate_columns(frame, parsed_name, allow_extra=allow_extra)

    expected_dtypes = dataset_dtypes(parsed_name)
    mismatches: dict[str, tuple[str, str]] = {}

    for column, expected_dtype in expected_dtypes.items():
        actual_dtype = str(frame[column].dtype)

        if actual_dtype != expected_dtype:
            mismatches[column] = (actual_dtype, expected_dtype)

    if mismatches:
        details = ", ".join(
            f"{column}: actual={actual!r}, expected={expected!r}"
            for column, (actual, expected) in mismatches.items()
        )

        raise TypeError(f"{parsed_name.value} has dtype mismatches: {details}")


def order_columns(frame: pd.DataFrame, dataset_name: DatasetName | str) -> pd.DataFrame:
    """Return frame with canonical columns first and extras after."""

    required = dataset_columns(dataset_name)
    extra = [column for column in frame.columns if column not in required]
    return frame.loc[:, [*required, *extra]]


def coerce_frame(
    frame: pd.DataFrame,
    dataset_name: DatasetName | str,
    *,
    allow_extra: bool = True,
) -> pd.DataFrame:
    """Return a copy of frame coerced to the expected pandas dtypes."""

    parsed_name = parse_dataset_name(dataset_name)
    validate_columns(frame, parsed_name, allow_extra=allow_extra)

    out = frame.copy()
    expected_dtypes = dataset_dtypes(parsed_name)

    for column, dtype in expected_dtypes.items():
        try:
            if dtype == "datetime64[ns, UTC]":
                out[column] = pd.to_datetime(out[column], utc=True)

            elif dtype == "datetime64[ns]":
                out[column] = pd.to_datetime(out[column]).dt.tz_localize(None)

            elif dtype == "Float64":
                out[column] = pd.to_numeric(out[column], errors="raise").astype(
                    pd.Float64Dtype()
                )

            elif dtype == "Int64":
                out[column] = pd.to_numeric(out[column], errors="raise").astype(
                    pd.Int64Dtype()
                )

            else:
                out[column] = out[column].astype(dtype)

        except Exception as exc:
            raise TypeError(
                f"Could not coerce column {column!r} in dataset "
                f"{parsed_name.value!r} to dtype {dtype!r}"
            ) from exc

    validate_dtypes(out, parsed_name, allow_extra=allow_extra)

    return out


__all__ = [
    "coerce_frame",
    "dataset_columns",
    "dataset_dtypes",
    "order_columns",
    "parse_dataset_name",
    "validate_columns",
    "validate_dtypes",
]
