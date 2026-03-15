from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .bundle import load_bundle_dataframe
from .types import BundleManifest


def _as_manifest(manifest_or_path: BundleManifest | str | Path) -> BundleManifest:
    if isinstance(manifest_or_path, BundleManifest):
        return manifest_or_path
    return BundleManifest.load(Path(manifest_or_path))


def _ordered_grid(
    df: pd.DataFrame,
    *,
    i_col: str,
    j_col: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    i_vals = np.sort(pd.to_numeric(df[i_col], errors="coerce").unique())
    j_vals = np.sort(pd.to_numeric(df[j_col], errors="coerce").unique())
    expected = len(i_vals) * len(j_vals)
    if len(df) != expected:
        raise ValueError(
            f"Dataset is not a complete rectangular grid: rows={len(df)} expected={expected}"
        )
    grid = df.set_index([j_col, i_col], drop=False).sort_index()
    if not grid.index.is_unique:
        raise ValueError(f"Dataset has duplicate ({j_col}, {i_col}) points")
    return grid, i_vals, j_vals


def _numeric_fields(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number, bool]).columns.tolist()


def _write_vts(
    *,
    grid: pd.DataFrame,
    i_vals: np.ndarray,
    j_vals: np.ndarray,
    i_col: str,
    j_col: str,
    x_col: str,
    y_col: str,
    z_col: str,
    point_fields: list[str],
    out_path: Path,
    flat: bool,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ni = len(i_vals)
    nj = len(j_vals)
    extent = f"0 {ni - 1} 0 {nj - 1} 0 0"

    with out_path.open("w", encoding="utf-8") as handle:
        handle.write('<?xml version="1.0"?>\n')
        handle.write(
            '<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">\n'
        )
        handle.write(f'  <StructuredGrid WholeExtent="{extent}">\n')
        handle.write(f'    <Piece Extent="{extent}">\n')
        handle.write("      <PointData>\n")
        for field in point_fields:
            handle.write(
                f'        <DataArray type="Float32" Name="{field}" NumberOfComponents="1" format="ascii">\n'
            )
            for j in j_vals:
                for i in i_vals:
                    row = grid.loc[(j, i)]
                    value = float(row[field])
                    handle.write(f"          {value:.8f}\n")
            handle.write("        </DataArray>\n")
        handle.write("      </PointData>\n")

        handle.write("      <Points>\n")
        handle.write(
            '        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n'
        )
        for j in j_vals:
            for i in i_vals:
                row = grid.loc[(j, i)]
                x = float(row[x_col])
                y = float(row[y_col])
                z = 0.0 if flat else float(row[z_col])
                handle.write(f"          {x:.8f} {y:.8f} {z:.8f}\n")
        handle.write("        </DataArray>\n")
        handle.write("      </Points>\n")
        handle.write("    </Piece>\n")
        handle.write("  </StructuredGrid>\n")
        handle.write("</VTKFile>\n")
    return out_path


def export_dataset_vts(
    manifest_or_path: BundleManifest | str | Path,
    *,
    dataset_name: str,
    out_dir: str | Path,
    flat: bool = False,
    z_col: str | None = None,
    point_fields: list[str] | None = None,
) -> Path:
    manifest = _as_manifest(manifest_or_path)
    dataset = manifest.resolve_dataset(dataset_name)
    if not dataset.rectangular_grid:
        raise ValueError(f"{dataset.name} is not a rectangular grid dataset.")

    axes = dataset.axis_columns
    i_col = axes["i"]
    j_col = axes["j"]
    x_col = axes.get("x", i_col)
    y_col = axes.get("y", j_col)
    z_name = z_col or axes.get("z") or dataset.default_scalar_fields[0]

    df = load_bundle_dataframe(manifest, dataset.name)
    grid, i_vals, j_vals = _ordered_grid(df, i_col=i_col, j_col=j_col)

    fields = list(point_fields or dataset.default_scalar_fields)
    available = set(_numeric_fields(df))
    fields = [field for field in fields if field in available]
    for required in (i_col, j_col, x_col, y_col, z_name):
        if required in available and required not in fields:
            fields.append(required)

    out_path = Path(out_dir) / f"{Path(dataset.relative_path).stem}.vts"
    return _write_vts(
        grid=grid,
        i_vals=i_vals,
        j_vals=j_vals,
        i_col=i_col,
        j_col=j_col,
        x_col=x_col,
        y_col=y_col,
        z_col=z_name,
        point_fields=fields,
        out_path=out_path,
        flat=flat,
    )
