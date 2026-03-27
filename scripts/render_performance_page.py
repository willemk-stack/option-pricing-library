from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "scripts" / "templates" / "performance.md.template"
OUTPUT_PATH = ROOT / "docs" / "performance.md"
SUMMARY_PATH = ROOT / "benchmarks" / "artifacts" / "performance_summary.json"
HEADER = (
    "<!-- Generated from scripts/templates/performance.md.template and "
    "benchmarks/artifacts/*. Do not edit docs/performance.md directly. -->\n\n"
)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_summary() -> dict[str, Any]:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def _to_float(value: str | float | int) -> float:
    return float(value)


def _fmt_runtime(value_ms: float) -> str:
    if value_ms >= 10_000:
        return f"{value_ms / 1000:.1f} s"
    if value_ms >= 1000:
        return f"{value_ms / 1000:.2f} s"
    return f"{value_ms:.2f} ms"


def _fmt_table_runtime(value_ms: float) -> str:
    if value_ms >= 1000:
        return f"{value_ms / 1000:.1f} s"
    return f"{value_ms:.1f} ms"


def _fmt_sci(value: float) -> str:
    return f"{value:.2e}"


def _fmt_speedup(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}x"
    return f"{value:.1f}x"


def _find_row(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    raise ValueError(f"Could not find row matching {criteria}")


def _build_snapshot_table() -> str:
    iv_speedup_rows = _read_csv_rows(
        ROOT / "benchmarks" / "artifacts" / "iv_speedup.csv"
    )
    iv_scalar_rows = _read_csv_rows(
        ROOT / "benchmarks" / "artifacts" / "iv_scalar_scenarios.csv"
    )
    pde_rows = _read_csv_rows(
        ROOT / "benchmarks" / "artifacts" / "pde_runtime_error_tradeoff.csv"
    )
    digital_rows = _read_csv_rows(
        ROOT / "benchmarks" / "artifacts" / "digital_remedies_stability.csv"
    )
    macro_rows = _read_csv_rows(
        ROOT / "benchmarks" / "artifacts" / "macro_pipeline_summary.csv"
    )

    iv_last = max(iv_speedup_rows, key=lambda row: int(row["n_strikes"]))
    scalar_runtimes = [_to_float(row["runtime_ms"]) for row in iv_scalar_rows]

    pde_rannacher = [
        row
        for row in pde_rows
        if row["family"] == "black_scholes_vanilla" and row["method"] == "rannacher"
    ]
    pde_low = min(pde_rannacher, key=lambda row: int(row["Nx"]))
    pde_high = max(pde_rannacher, key=lambda row: int(row["Nx"]))

    digital_none = _find_row(digital_rows, config="Rannacher + none")
    digital_cell_avg = _find_row(digital_rows, config="Rannacher + cell average")
    digital_l2 = _find_row(digital_rows, config="Rannacher + L2 projection")

    macro_surface_fit = {
        row["scenario"]: row for row in macro_rows if row["stage"] == "surface_fit_ms"
    }
    macro_pde = {
        row["scenario"]: row for row in macro_rows if row["stage"] == "pde_reprice_ms"
    }

    rows = [
        [
            "Implied-vol inversion",
            "Scalar BS inversion and Black-76 slice inversion on 21-801 strikes",
            "Scalar loop baseline",
            (
                f"The vectorized slice path reaches {_fmt_speedup(_to_float(iv_last['vectorized_speedup_x']))} speedup "
                f"at {iv_last['n_strikes']} strikes while preserving zero measured vol difference versus the scalar loop. "
                f"Scalar single-option inversions stay around {_fmt_runtime(min(scalar_runtimes))} to {_fmt_runtime(max(scalar_runtimes))}."
            ),
        ],
        [
            "Vanilla PDE",
            "Black-Scholes European call, Nx=Nt from 101 to 801",
            "Analytic Black-Scholes price",
            (
                f"Absolute error falls from about {_fmt_sci(_to_float(pde_low['abs_error']))} at {pde_low['Nx']}x{pde_low['Nt']} "
                f"to about {_fmt_sci(_to_float(pde_high['abs_error']))} at {pde_high['Nx']}x{pde_high['Nt']}. "
                f"Runtime rises from about {_fmt_table_runtime(_to_float(pde_low['runtime_ms']))} to about {_fmt_table_runtime(_to_float(pde_high['runtime_ms']))}."
            ),
        ],
        [
            "Digital PDE remedies",
            "Digital call, Nx=Nt in {101, 201, 401}",
            "Analytic digital price",
            (
                f"The untreated path keeps an ~{_fmt_sci(_to_float(digital_none['refinement_spread']))} refinement spread, "
                f"while Rannacher + cell average cuts it to {_fmt_sci(_to_float(digital_cell_avg['refinement_spread']))} "
                f"and Rannacher + L2 projection cuts it to {_fmt_sci(_to_float(digital_l2['refinement_spread']))}."
            ),
        ],
        [
            "End-to-end macro pipeline",
            "Synthetic SVI quote mesh -> fitted surface -> handoff probe -> local-vol surface -> representative local-vol PDE",
            "Stage-level timing only",
            (
                "Surface fitting dominates the measured stage budget "
                f"({_fmt_table_runtime(_to_float(macro_surface_fit['small']['runtime_ms']))}, "
                f"{_fmt_table_runtime(_to_float(macro_surface_fit['medium']['runtime_ms']))}, "
                f"{_fmt_table_runtime(_to_float(macro_surface_fit['large']['runtime_ms']))} for small, medium, large), "
                "with PDE repricing second "
                f"({_fmt_table_runtime(_to_float(macro_pde['small']['runtime_ms']))}, "
                f"{_fmt_table_runtime(_to_float(macro_pde['medium']['runtime_ms']))}, "
                f"{_fmt_table_runtime(_to_float(macro_pde['large']['runtime_ms']))})."
            ),
        ],
    ]

    lines = [
        "| Family | Conditions | Reference | What the committed snapshot shows |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(f"| {' | '.join(row)} |" for row in rows)
    return "\n".join(lines)


def _build_key_signals() -> str:
    iv_rows = _read_csv_rows(ROOT / "benchmarks" / "artifacts" / "iv_speedup.csv")
    digital_rows = _read_csv_rows(
        ROOT / "benchmarks" / "artifacts" / "digital_remedies_stability.csv"
    )
    macro_rows = _read_csv_rows(
        ROOT / "benchmarks" / "artifacts" / "macro_pipeline_summary.csv"
    )

    iv_last = max(iv_rows, key=lambda row: int(row["n_strikes"]))
    digital_none = _find_row(digital_rows, config="Rannacher + none")
    digital_l2 = _find_row(digital_rows, config="Rannacher + L2 projection")

    macro_grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in macro_rows:
        macro_grouped.setdefault(row["scenario"], {})[row["stage"]] = row

    surface_fit_small = _to_float(
        macro_grouped["small"]["surface_fit_ms"]["runtime_ms"]
    )
    surface_fit_large = _to_float(
        macro_grouped["large"]["surface_fit_ms"]["runtime_ms"]
    )
    pde_small = _to_float(macro_grouped["small"]["pde_reprice_ms"]["runtime_ms"])
    pde_large = _to_float(macro_grouped["large"]["pde_reprice_ms"]["runtime_ms"])
    handoff_small = _to_float(
        macro_grouped["small"]["surface_handoff_ms"]["runtime_ms"]
    )
    handoff_large = _to_float(
        macro_grouped["large"]["surface_handoff_ms"]["runtime_ms"]
    )

    return "\n".join(
        [
            '<div class="doc-card-grid doc-card-grid--quiet" markdown="1">',
            '<div class="doc-card doc-card--quiet" markdown="1">',
            '<p class="doc-card__eyebrow">IV scaling</p>',
            f"<p class=\"doc-card__title\">`{_fmt_speedup(_to_float(iv_last['vectorized_speedup_x']))} at {iv_last['n_strikes']} strikes`</p>",
            "Vectorized Black-76 slice inversion keeps zero measured vol difference versus the scalar-loop baseline.",
            "</div>",
            '<div class="doc-card doc-card--quiet" markdown="1">',
            '<p class="doc-card__eyebrow">Digital remedies</p>',
            f"<p class=\"doc-card__title\">`{_fmt_sci(_to_float(digital_none['refinement_spread']))} -> {_fmt_sci(_to_float(digital_l2['refinement_spread']))}`</p>",
            "Refinement spread collapses when the discontinuity treatment moves from no remedy to Rannacher + L2 projection.",
            "</div>",
            '<div class="doc-card doc-card--quiet" markdown="1">',
            '<p class="doc-card__eyebrow">Stage budget</p>',
            f'<p class="doc-card__title">`{_fmt_runtime(handoff_small)} -> {_fmt_runtime(handoff_large)}` handoff</p>',
            f"Surface fit ({_fmt_table_runtime(surface_fit_small)} to {_fmt_table_runtime(surface_fit_large)}) and PDE repricing ({_fmt_table_runtime(pde_small)} to {_fmt_table_runtime(pde_large)}) dominate the integrated workflow instead.",
            "</div>",
            "</div>",
        ]
    )


def _build_iv_summary() -> str:
    iv_rows = _read_csv_rows(ROOT / "benchmarks" / "artifacts" / "iv_speedup.csv")
    iv_last = max(iv_rows, key=lambda row: int(row["n_strikes"]))
    return (
        "Use the vectorized slice inverter whenever the workload is a smile or surface rather than an isolated option. "
        f"At {iv_last['n_strikes']} strikes the committed snapshot records about {_fmt_runtime(_to_float(iv_last['vectorized_slice']))} "
        f"for the vectorized slice path versus about {_fmt_runtime(_to_float(iv_last['scalar_loop']))} for the scalar-loop baseline."
    )


def _build_pde_summary() -> str:
    pde_rows = _read_csv_rows(
        ROOT / "benchmarks" / "artifacts" / "pde_runtime_error_tradeoff.csv"
    )
    localvol_rows = [row for row in pde_rows if row["family"] == "local_vol_digital"]
    localvol_ref = max(localvol_rows, key=lambda row: int(row["Nx"]))
    return (
        "The vanilla PDE curve shows the expected refinement tradeoff against an analytic benchmark. "
        "The local-vol curve uses a finer-grid local-vol PDE solve as its reference because there is no closed-form target for that path. "
        "That is the right comparison for review: the figure shows what extra grid density buys rather than just how long one solve took. "
        f"In the current snapshot, the finest local-vol reference grid is {localvol_ref['Nx']}x{localvol_ref['Nt']}."
    )


def _build_macro_summary() -> str:
    rows = _read_csv_rows(
        ROOT / "benchmarks" / "artifacts" / "macro_pipeline_summary.csv"
    )
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["scenario"], {})[row["stage"]] = row

    return (
        "For the integrated benchmark, the key observation is where time actually goes. "
        f"In this snapshot the handoff itself stays cheap at {_fmt_runtime(_to_float(grouped['small']['surface_handoff_ms']['runtime_ms']))} to {_fmt_runtime(_to_float(grouped['large']['surface_handoff_ms']['runtime_ms']))}, "
        f"the local-vol extraction step stays around {_fmt_runtime(_to_float(grouped['small']['local_vol_ms']['runtime_ms']))} to {_fmt_runtime(_to_float(grouped['large']['local_vol_ms']['runtime_ms']))}, "
        "and the expensive parts are fitting the surface and running the PDE. That is useful when discussing optimization priorities because it argues against spending engineering effort on the wrong stage."
    )


def _build_localvol_table() -> str:
    rows = _read_csv_rows(ROOT / "benchmarks" / "artifacts" / "localvol_scaling.csv")
    largest = {
        coordinate: max(
            [row for row in rows if row["strike_coordinate"] == coordinate],
            key=lambda row: int(row["grid_points"]),
        )
        for coordinate in ("K", "logK")
    }
    lines = [
        "| Coordinate | Largest tested grid | Median runtime | Interior invalid share |",
        "| --- | --- | --- | --- |",
    ]
    for coordinate in ("K", "logK"):
        row = largest[coordinate]
        lines.append(
            "| "
            f"`{coordinate}` | `{row['nT']} x {row['nK']}` | {_fmt_runtime(_to_float(row['runtime_ms']))} | {100 * _to_float(row['interior_invalid_share']):.1f}% |"
        )
    return "\n".join(lines)


def _build_tree_table() -> str:
    rows = _read_csv_rows(ROOT / "benchmarks" / "artifacts" / "tree_scaling.csv")
    selected_steps = {"200", "1000", "5000"}
    lines = [
        "| `n_steps` | Median runtime | Absolute error vs Black-Scholes |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        if row["n_steps"] not in selected_steps:
            continue
        lines.append(
            "| "
            f"`{row['n_steps']}` | {_fmt_table_runtime(_to_float(row['runtime_ms']))} | {_fmt_sci(_to_float(row['abs_error']))} |"
        )
    return "\n".join(lines)


def _build_environment_summary(summary: dict[str, Any]) -> str:
    environment = summary["environment"]
    versions = environment["versions"]
    cpu_count = environment.get("cpu_count")
    cpu_fragment = (
        f", {cpu_count} logical CPUs"
        if isinstance(cpu_count, int) and cpu_count > 0
        else ""
    )
    return (
        f"- Snapshot environment: Python `{versions['python']}`, NumPy `{versions['numpy']}`, "
        f"SciPy `{versions['scipy']}`, pandas `{versions['pandas']}`{cpu_fragment}.\n"
        "- Treat the absolute timings as machine-specific. The defensible signal is the relative slope of each curve, the runtime/error tradeoff, and which workflow dominates the stage budget.\n"
        "- The machine-readable snapshot used for this page is committed under `benchmarks/artifacts/`, with stable summary metadata in `benchmarks/artifacts/performance_summary.json`."
    )


def render(*, check: bool) -> int:
    summary = _load_summary()
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    rendered = template
    replacements = {
        "{{ SNAPSHOT_TABLE }}": _build_snapshot_table(),
        "{{ KEY_SIGNALS }}": _build_key_signals(),
        "{{ IV_SUMMARY }}": _build_iv_summary(),
        "{{ PDE_SUMMARY }}": _build_pde_summary(),
        "{{ MACRO_SUMMARY }}": _build_macro_summary(),
        "{{ LOCALVOL_TABLE }}": _build_localvol_table(),
        "{{ TREE_TABLE }}": _build_tree_table(),
        "{{ ENVIRONMENT_SUMMARY }}": _build_environment_summary(summary),
    }
    for placeholder, replacement in replacements.items():
        rendered = rendered.replace(placeholder, replacement)

    rendered = HEADER + rendered.rstrip() + "\n"
    if check:
        current = (
            OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.exists() else ""
        )
        if current != rendered:
            sys.stderr.write(
                "docs/performance.md is out of date. Run: python scripts/render_performance_page.py\n"
            )
            return 1
        return 0

    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render docs/performance.md from the committed benchmark artifacts."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if docs/performance.md is not in sync with the template and artifacts.",
    )
    args = parser.parse_args()
    return render(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
