from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from build_benchmark_artifacts import build_benchmark_overview_asset  # noqa: E402

from option_pricing.demos.publishing import (  # noqa: E402
    build_visual_bundle,
    export_dataset_vts,
    load_bundle_manifest,
    render_plot_presets,
)


def _bundle_dir(profile: str, seed: int, bundle_dir: str | None) -> Path:
    if bundle_dir:
        return Path(bundle_dir)
    return ROOT / "out" / "visual_bundles" / f"profile_{profile}_seed_{int(seed)}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_file_map(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}

    files: dict[str, str] = {}
    for path in sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    ):
        files[path.relative_to(root).as_posix()] = _sha256(path)
    return files


def _check_rendered_tree(current_root: Path, rendered_root: Path) -> int:
    current = _relative_file_map(current_root)
    rendered = _relative_file_map(rendered_root)

    missing = sorted(path for path in rendered if path not in current)
    changed = sorted(
        path for path in rendered if path in current and rendered[path] != current[path]
    )
    orphaned = sorted(path for path in current if path not in rendered)

    if not missing and not changed and not orphaned:
        print("Generated visual assets are up to date.")
        return 0

    print("Generated visual assets are out of date. Rebuild with:")
    print("  python scripts/build_visual_artifacts.py all --profile ci\n")
    for label, items in (
        ("missing", missing),
        ("changed", changed),
        ("orphaned", orphaned),
    ):
        if not items:
            continue
        for item in items:
            print(f" - {label}: {item}")
    return 1


def _seed_existing_benchmark_assets(current_root: Path, rendered_root: Path) -> None:
    current_benchmarks = current_root / "benchmarks"
    rendered_benchmarks = rendered_root / "benchmarks"
    if not current_benchmarks.exists():
        return

    rendered_benchmarks.mkdir(parents=True, exist_ok=True)
    for path in sorted(
        candidate for candidate in current_benchmarks.rglob("*") if candidate.is_file()
    ):
        target = rendered_benchmarks / path.relative_to(current_benchmarks)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build visual publishing bundles, plots, and VTS exports."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_shared(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--profile", choices=("ci", "publish"), default="ci")
        subparser.add_argument("--seed", type=int, default=7)
        subparser.add_argument(
            "--check",
            action="store_true",
            help="Do not write docs assets. Fail if rendered outputs would differ.",
        )
        subparser.add_argument(
            "--bundle-dir",
            type=str,
            default="",
            help="Optional override for the visual bundle directory.",
        )

    p_data = sub.add_parser("data", help="Build the canonical data bundle.")
    add_shared(p_data)

    p_plots = sub.add_parser("plots", help="Render plot presets from a data bundle.")
    add_shared(p_plots)
    p_plots.add_argument(
        "--preset",
        dest="presets",
        action="append",
        default=[],
        choices=("static", "dupire", "poster", "docs", "numerics", "showcase"),
        help="Plot preset to render. Repeat to render multiple presets.",
    )
    p_plots.add_argument(
        "--out-root",
        type=str,
        default=str(ROOT / "docs" / "assets" / "generated"),
        help="Root output directory for rendered plot presets.",
    )

    p_vts = sub.add_parser("vts", help="Export one or more datasets to VTS.")
    add_shared(p_vts)
    p_vts.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        required=True,
        help="Dataset name or alias, for example essvi_smoothed_grid or localvol/gatheral_grid.",
    )
    p_vts.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "out" / "visual_vts"),
        help="Output directory for VTS files.",
    )
    p_vts.add_argument(
        "--flat",
        action="store_true",
        help="Write geometry with z=0 so the VTS can be warped by scalar in ParaView.",
    )

    p_all = sub.add_parser(
        "all",
        help="Build the canonical data bundle and render plot presets.",
    )
    add_shared(p_all)
    p_all.add_argument(
        "--preset",
        dest="presets",
        action="append",
        default=[],
        choices=("static", "dupire", "poster", "docs", "numerics", "showcase"),
        help="Plot preset to render. Repeat to render multiple presets.",
    )
    p_all.add_argument(
        "--out-root",
        type=str,
        default=str(ROOT / "docs" / "assets" / "generated"),
        help="Root output directory for rendered plot presets.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    bundle_dir = _bundle_dir(args.profile, args.seed, args.bundle_dir or None)

    if args.command in {"data", "vts"} and args.check:
        raise SystemExit(
            "--check is only supported for the `plots` and `all` commands."
        )

    if args.command == "data":
        result = build_visual_bundle(
            profile=args.profile,
            seed=args.seed,
            bundle_dir=bundle_dir,
        )
        print(result.manifest.manifest_path())
        return 0

    if args.command == "plots":
        manifest = load_bundle_manifest(bundle_dir)
        if args.check:
            with tempfile.TemporaryDirectory(
                prefix=".tmp-visual-artifacts-check-",
                dir=ROOT,
            ) as tmp_dir:
                rendered_root = Path(tmp_dir) / "generated"
                _seed_existing_benchmark_assets(Path(args.out_root), rendered_root)
                render_plot_presets(
                    manifest,
                    presets=args.presets or None,
                    out_root=rendered_root,
                )
                build_benchmark_overview_asset(
                    artifacts_dir=ROOT / "benchmarks" / "artifacts",
                    plot_dir=rendered_root / "benchmarks",
                )
                return _check_rendered_tree(Path(args.out_root), rendered_root)

        written = render_plot_presets(
            manifest,
            presets=args.presets or None,
            out_root=args.out_root,
        )
        overview_path = build_benchmark_overview_asset(
            artifacts_dir=ROOT / "benchmarks" / "artifacts",
            plot_dir=Path(args.out_root) / "benchmarks",
        )
        written.append(ROOT / overview_path)
        for path in written:
            print(path)
        return 0

    if args.command == "vts":
        manifest = load_bundle_manifest(bundle_dir)
        for dataset in args.datasets:
            path = export_dataset_vts(
                manifest,
                dataset_name=dataset,
                out_dir=args.out_dir,
                flat=bool(args.flat),
            )
            print(path)
        return 0

    if args.command == "all":
        if args.check:
            with tempfile.TemporaryDirectory(
                prefix=".tmp-visual-artifacts-check-",
                dir=ROOT,
            ) as tmp_dir:
                temp_root = Path(tmp_dir)
                temp_bundle_dir = temp_root / "bundle"
                rendered_root = temp_root / "generated"
                _seed_existing_benchmark_assets(Path(args.out_root), rendered_root)
                result = build_visual_bundle(
                    profile=args.profile,
                    seed=args.seed,
                    bundle_dir=temp_bundle_dir,
                )
                render_plot_presets(
                    result.manifest,
                    presets=args.presets or None,
                    out_root=rendered_root,
                )
                build_benchmark_overview_asset(
                    artifacts_dir=ROOT / "benchmarks" / "artifacts",
                    plot_dir=rendered_root / "benchmarks",
                )
                return _check_rendered_tree(Path(args.out_root), rendered_root)

        result = build_visual_bundle(
            profile=args.profile,
            seed=args.seed,
            bundle_dir=bundle_dir,
        )
        written = render_plot_presets(
            result.manifest,
            presets=args.presets or None,
            out_root=args.out_root,
        )
        overview_path = build_benchmark_overview_asset(
            artifacts_dir=ROOT / "benchmarks" / "artifacts",
            plot_dir=Path(args.out_root) / "benchmarks",
        )
        written.append(ROOT / overview_path)
        print(result.manifest.manifest_path())
        for path in written:
            print(path)
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
