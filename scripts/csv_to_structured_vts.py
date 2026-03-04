#!/usr/bin/env python3
import argparse
import re

import numpy as np
import pandas as pd


def sanitize_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip())
    if not s:
        s = "array"
    if s[0].isdigit():
        s = "_" + s
    return s


def parse_csv_list(s: str | None) -> list[str] | None:
    if s is None:
        return None
    items = [x.strip() for x in s.split(",") if x.strip()]
    return items if items else None


def select_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def build_ordered_rows(
    df: pd.DataFrame, i_col: str, j_col: str
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Returns:
      grid: df indexed by (j,i) and sorted, drop=False so columns remain
      i_vals, j_vals: sorted unique axis values
    """
    i_vals = np.sort(df[i_col].unique())
    j_vals = np.sort(df[j_col].unique())
    ni, nj = len(i_vals), len(j_vals)

    if len(df) != ni * nj:
        raise ValueError(
            f"CSV is not a complete grid: rows={len(df)} expected={ni*nj} "
            f"(ni={ni}, nj={nj})."
        )

    grid = df.set_index([j_col, i_col], drop=False).sort_index()
    if not grid.index.is_unique:
        raise ValueError("Index (j_col, i_col) is not unique (duplicate grid points).")

    expected = pd.MultiIndex.from_product([j_vals, i_vals], names=[j_col, i_col])
    if not expected.isin(grid.index).all():
        missing = expected[~expected.isin(grid.index)]
        raise ValueError(
            f"Missing {len(missing)} grid points (showing up to 10): {list(missing[:10])}"
        )

    return grid, i_vals, j_vals


def write_vts(
    grid: pd.DataFrame,
    out_path: str,
    i_vals: np.ndarray,
    j_vals: np.ndarray,
    i_col: str,
    j_col: str,
    x_col: str,
    y_col: str,
    z_col: str,
    point_arrays: list[str],
    flat: bool,
    title: str,
    float_type: str = "Float32",
):
    ni, nj = len(i_vals), len(j_vals)
    nk = 1  # 2D grid

    # Build points in VTK order: i fastest, then j, then k
    points: list[tuple[float, float, float]] = []
    for j in j_vals:
        for i in i_vals:
            row = grid.loc[(j, i)]
            if isinstance(row, pd.DataFrame):
                raise ValueError(f"Duplicate rows for ({j_col}={j}, {i_col}={i}).")
            x = float(row[x_col])
            y = float(row[y_col])
            z = 0.0 if flat else float(row[z_col])
            points.append((x, y, z))

    # Collect point data arrays in the same order
    pdata: dict[str, list[float]] = {}
    for c in point_arrays:
        vals: list[float] = []
        for j in j_vals:
            for i in i_vals:
                row = grid.loc[(j, i)]
                vals.append(float(row[c]))
        pdata[c] = vals

    # Sanitize and avoid collisions
    name_map: dict[str, str] = {}
    used = set()
    for c in point_arrays:
        s = sanitize_name(c)
        base = s
        k = 1
        while s in used:
            k += 1
            s = f"{base}_{k}"
        used.add(s)
        name_map[c] = s

    # Extents are inclusive indices for points
    whole_extent = f"0 {ni-1} 0 {nj-1} 0 {nk-1}"
    npts = ni * nj * nk

    with open(out_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write(f"<!-- {title} -->\n")
        f.write(
            '<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">\n'
        )
        f.write(f'  <StructuredGrid WholeExtent="{whole_extent}">\n')
        f.write(f'    <Piece Extent="{whole_extent}">\n')

        # PointData
        f.write("      <PointData>\n")
        for orig_col in point_arrays:
            arr_name = name_map[orig_col]
            f.write(
                f'        <DataArray type="{float_type}" Name="{arr_name}" NumberOfComponents="1" format="ascii">\n'
            )
            # write values, one per line (ParaView-friendly)
            for v in pdata[orig_col]:
                f.write(f"          {v:.8f}\n")
            f.write("        </DataArray>\n")
        f.write("      </PointData>\n")

        # Points
        f.write("      <Points>\n")
        f.write(
            f'        <DataArray type="{float_type}" NumberOfComponents="3" format="ascii">\n'
        )
        for x, y, z in points:
            f.write(f"          {x:.8f} {y:.8f} {z:.8f}\n")
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")

        f.write("    </Piece>\n")
        f.write("  </StructuredGrid>\n")
        f.write("</VTKFile>\n")

    print(f"Wrote {out_path} ({npts} points, {len(point_arrays)} arrays)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV file")
    ap.add_argument("--out", required=True, help="Output VTS (.vts) file")

    ap.add_argument("--i-col", required=True, help="Grid fast axis (e.g., y or K)")
    ap.add_argument("--j-col", required=True, help="Grid slow axis (e.g., T)")

    ap.add_argument("--x-col", required=True, help="X coordinate column")
    ap.add_argument("--y-col", required=True, help="Y coordinate column")
    ap.add_argument("--z-col", required=True, help="Z coordinate column (height)")

    ap.add_argument(
        "--write-cols",
        default=None,
        help="Comma-separated columns to export as PointData (default: all numeric).",
    )
    ap.add_argument(
        "--exclude-cols",
        default=None,
        help="Comma-separated columns to exclude from PointData.",
    )
    ap.add_argument(
        "--flat",
        action="store_true",
        help="Write geometry with Z=0 so you can Warp By Scalar in ParaView.",
    )
    ap.add_argument(
        "--title", default="Structured grid from CSV", help="Title comment in the file"
    )

    args = ap.parse_args()
    df = pd.read_csv(args.csv)

    # Require axes/coords first (z_col may be derived below)
    for col in [args.i_col, args.j_col, args.x_col, args.y_col]:
        if col not in df.columns:
            raise SystemExit(f"Missing required column '{col}' in CSV.")

    # --- NEW: derive missing iv_svi / w_svi if possible ---
    if "T" in df.columns:
        T = pd.to_numeric(df["T"], errors="coerce").to_numpy(dtype=float)

        if "w_svi" not in df.columns and "iv_svi" in df.columns:
            iv = pd.to_numeric(df["iv_svi"], errors="coerce").to_numpy(dtype=float)
            w = np.where(
                (T > 0) & np.isfinite(T) & np.isfinite(iv), T * iv * iv, np.nan
            )
            df["w_svi"] = w

        if "iv_svi" not in df.columns and "w_svi" in df.columns:
            w = pd.to_numeric(df["w_svi"], errors="coerce").to_numpy(dtype=float)
            iv = np.where(
                (T > 0) & np.isfinite(T) & np.isfinite(w) & (w >= 0),
                np.sqrt(w / T),
                np.nan,
            )
            df["iv_svi"] = iv

    # Now ensure z-col exists (it might have been derived)
    if args.z_col not in df.columns:
        raise SystemExit(
            f"Missing required column '{args.z_col}' in CSV (after derivations)."
        )

    # Choose arrays
    write_cols = parse_csv_list(args.write_cols)
    exclude_cols = set(parse_csv_list(args.exclude_cols) or [])

    if write_cols is None:
        arrays = select_numeric_columns(df)
    else:
        missing = [c for c in write_cols if c not in df.columns]
        if missing:
            raise SystemExit(f"--write-cols includes missing columns: {missing}")
        arrays = write_cols

    arrays = [c for c in arrays if c not in exclude_cols]

    # Always include these as arrays (handy in ParaView), if numeric
    must = [args.i_col, args.j_col, args.x_col, args.y_col, args.z_col]
    for c in must:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and c not in arrays:
            arrays.append(c)

    # --- NEW: always expose both if present & not excluded ---
    for c in ("iv_svi", "w_svi"):
        if (
            c in df.columns
            and pd.api.types.is_numeric_dtype(df[c])
            and c not in arrays
            and c not in exclude_cols
        ):
            arrays.append(c)

    grid, i_vals, j_vals = build_ordered_rows(df, args.i_col, args.j_col)

    write_vts(
        grid=grid,
        out_path=args.out,
        i_vals=i_vals,
        j_vals=j_vals,
        i_col=args.i_col,
        j_col=args.j_col,
        x_col=args.x_col,
        y_col=args.y_col,
        z_col=args.z_col,
        point_arrays=arrays,
        flat=args.flat,
        title=args.title,
    )


if __name__ == "__main__":
    main()
