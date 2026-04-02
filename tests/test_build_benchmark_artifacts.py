from __future__ import annotations

import json
from pathlib import Path

import scripts.benchmark_source_scope as benchmark_scope
import scripts.build_benchmark_artifacts as benchmark_artifacts


def _write_sample_source(path: Path, *, newline: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(newline.join((b"def price():", b"    return 1.0", b"")))


def test_benchmark_source_manifest_payload_normalizes_checkout_newlines(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "benchmarks" / "sample.py"
    monkeypatch.setattr(benchmark_artifacts, "ROOT", tmp_path)
    monkeypatch.setattr(
        benchmark_artifacts,
        "_iter_benchmark_source_files",
        lambda: [source_path],
    )

    _write_sample_source(source_path, newline=b"\n")
    lf_payload = benchmark_artifacts._benchmark_source_manifest_payload()

    _write_sample_source(source_path, newline=b"\r\n")
    crlf_payload = benchmark_artifacts._benchmark_source_manifest_payload()

    assert lf_payload == crlf_payload


def test_check_benchmark_source_manifest_ignores_checkout_newlines(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    source_path = tmp_path / "benchmarks" / "sample.py"
    artifacts_dir = tmp_path / "benchmarks" / "artifacts"
    monkeypatch.setattr(benchmark_artifacts, "ROOT", tmp_path)
    monkeypatch.setattr(
        benchmark_artifacts,
        "_iter_benchmark_source_files",
        lambda: [source_path],
    )

    _write_sample_source(source_path, newline=b"\n")
    manifest_path = benchmark_artifacts._write_benchmark_source_manifest(artifacts_dir)

    _write_sample_source(source_path, newline=b"\r\n")
    assert benchmark_artifacts._check_benchmark_source_manifest(artifacts_dir) == 0
    assert "up to date" in capsys.readouterr().out

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == benchmark_artifacts.BENCHMARK_SOURCE_MANIFEST_VERSION


def test_check_benchmark_source_manifest_requires_current_version(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    source_path = tmp_path / "benchmarks" / "sample.py"
    artifacts_dir = tmp_path / "benchmarks" / "artifacts"
    manifest_path = artifacts_dir / benchmark_artifacts.BENCHMARK_SOURCE_MANIFEST
    monkeypatch.setattr(benchmark_artifacts, "ROOT", tmp_path)
    monkeypatch.setattr(
        benchmark_artifacts,
        "_iter_benchmark_source_files",
        lambda: [source_path],
    )

    _write_sample_source(source_path, newline=b"\n")
    benchmark_artifacts._write_benchmark_source_manifest(artifacts_dir)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = benchmark_artifacts.BENCHMARK_SOURCE_MANIFEST_VERSION - 1
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    assert benchmark_artifacts._check_benchmark_source_manifest(artifacts_dir) == 1
    output = capsys.readouterr().out
    assert "Benchmark source manifest format is out of date." in output


def test_benchmark_source_scope_tracks_expected_repo_files() -> None:
    paths = set(benchmark_scope.benchmark_source_paths())

    assert "src/option_pricing/instruments/digital.py" in paths
    assert "src/option_pricing/viz/publishing.py" in paths
    assert "src/option_pricing/vol/svi/__init__.py" in paths
    assert "src/option_pricing/pricers/heston.py" not in paths
    assert "src/option_pricing/models/heston/__init__.py" not in paths


def test_benchmark_source_scope_ignores_untracked_files(
    monkeypatch,
    tmp_path: Path,
) -> None:
    builder_path = tmp_path / "scripts" / "build_benchmark_artifacts.py"
    benchmark_path = tmp_path / "benchmarks" / "test_bench_tmp.py"
    package_root = tmp_path / "src" / "option_pricing"
    tracked_paths = frozenset(
        {
            "benchmarks/test_bench_tmp.py",
            "scripts/build_benchmark_artifacts.py",
            "src/option_pricing/__init__.py",
            "src/option_pricing/pricers/__init__.py",
            "src/option_pricing/pricers/core.py",
        }
    )

    (package_root / "pricers").mkdir(parents=True, exist_ok=True)
    builder_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)

    builder_path.write_text(
        "from option_pricing.pricers.core import price\n",
        encoding="utf-8",
    )
    benchmark_path.write_text(
        "from option_pricing.pricers.core import price\n",
        encoding="utf-8",
    )
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "pricers" / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "pricers" / "core.py").write_text(
        "def price() -> float:\n    return 1.0\n",
        encoding="utf-8",
    )
    (package_root / "pricers" / "heston.py").write_text(
        "def unused() -> float:\n    return 0.0\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        benchmark_scope,
        "_git_tracked_paths",
        lambda root_str: tracked_paths,
    )

    paths = set(benchmark_scope.benchmark_source_paths(tmp_path))

    assert "src/option_pricing/pricers/core.py" in paths
    assert "src/option_pricing/pricers/heston.py" not in paths
