from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from . import build_heston_docs_artifacts
except ImportError:
    import build_heston_docs_artifacts

ROOT = Path(__file__).resolve().parents[1]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail when the committed Heston comparison artifact bundle is stale "
            "relative to its tracked source inputs."
        )
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ROOT / "docs" / "assets" / "generated" / "heston",
        help="Directory containing the committed Heston docs artifact bundle.",
    )
    return parser.parse_args(argv)


def _freshness_manifest_path(artifacts_dir: Path) -> Path:
    return (
        artifacts_dir
        / "data"
        / build_heston_docs_artifacts.HESTON_ARTIFACT_FRESHNESS_MANIFEST
    )


def _print_rebuild_command() -> None:
    print("Regenerate the committed Heston comparison artifacts with:")
    print(f"  {build_heston_docs_artifacts.HESTON_ARTIFACT_REBUILD_COMMAND}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    artifacts_dir = args.artifacts_dir.resolve()
    manifest_path = _freshness_manifest_path(artifacts_dir)

    if not manifest_path.exists():
        print("Heston artifact freshness manifest is missing.")
        _print_rebuild_command()
        return 1

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_version = payload.get("version")
    if (
        recorded_version
        != build_heston_docs_artifacts.HESTON_ARTIFACT_FRESHNESS_VERSION
    ):
        print("Heston artifact freshness manifest format is out of date.")
        print(
            " - manifest version: "
            f"recorded={recorded_version!r}, "
            "expected="
            f"{build_heston_docs_artifacts.HESTON_ARTIFACT_FRESHNESS_VERSION}"
        )
        _print_rebuild_command()
        return 1

    formats = tuple(str(item) for item in payload.get("formats", ("svg", "png")))
    recorded_source_inputs = dict(payload.get("source_inputs", {}))
    current_source_inputs = (
        build_heston_docs_artifacts._heston_artifact_source_payload()
    )

    missing_sources = sorted(
        path for path in current_source_inputs if path not in recorded_source_inputs
    )
    changed_sources = sorted(
        path
        for path, digest in current_source_inputs.items()
        if path in recorded_source_inputs and recorded_source_inputs[path] != digest
    )
    orphaned_sources = sorted(
        path for path in recorded_source_inputs if path not in current_source_inputs
    )

    expected_generated_paths = {
        build_heston_docs_artifacts._relative_to_dir(path, artifacts_dir)
        for path in build_heston_docs_artifacts._freshness_tracked_paths(
            artifacts_dir,
            formats=formats,
        )
    }
    recorded_generated_files = list(payload.get("generated_files", []))
    recorded_generated_paths = {
        str(entry.get("path", ""))
        for entry in recorded_generated_files
        if entry.get("path")
    }
    missing_generated_entries = sorted(
        expected_generated_paths - recorded_generated_paths
    )
    orphaned_generated_entries = sorted(
        recorded_generated_paths - expected_generated_paths
    )

    missing_files_on_disk: list[str] = []
    changed_generated_files: list[str] = []
    for entry in recorded_generated_files:
        relative_path = str(entry.get("path", "")).strip()
        if not relative_path:
            continue
        artifact_path = artifacts_dir / relative_path
        if not artifact_path.exists():
            missing_files_on_disk.append(relative_path)
            continue
        recorded_digest = str(entry.get("sha256", "")).strip()
        current_digest = build_heston_docs_artifacts._file_sha256(artifact_path)
        if recorded_digest != current_digest:
            changed_generated_files.append(relative_path)

    if not any(
        (
            missing_sources,
            changed_sources,
            orphaned_sources,
            missing_generated_entries,
            orphaned_generated_entries,
            missing_files_on_disk,
            changed_generated_files,
        )
    ):
        print("Heston docs artifacts are up to date.")
        return 0

    print("Heston comparison artifacts are stale relative to tracked sources.")
    for label, items in (
        ("missing source input", missing_sources),
        ("changed source input", changed_sources),
        ("orphaned source input", orphaned_sources),
        ("missing generated manifest entry", missing_generated_entries),
        ("orphaned generated manifest entry", orphaned_generated_entries),
        ("missing generated file", missing_files_on_disk),
        ("changed generated file", changed_generated_files),
    ):
        for item in items:
            print(f" - {label}: {item}")
    _print_rebuild_command()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
