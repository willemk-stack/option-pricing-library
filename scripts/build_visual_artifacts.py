from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build visual publishing bundles, plots, and VTS exports."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_shared(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--profile", choices=("ci", "publish"), default="ci")
        subparser.add_argument("--seed", type=int, default=7)
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
        choices=("static", "dupire", "poster", "docs", "numerics"),
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
        choices=("static", "dupire", "poster", "docs", "numerics"),
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
        written = render_plot_presets(
            manifest,
            presets=args.presets or None,
            out_root=args.out_root,
        )
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
        print(result.manifest.manifest_path())
        for path in written:
            print(path)
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
