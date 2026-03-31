from __future__ import annotations

import json
from pathlib import Path

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
        "BENCHMARK_SOURCE_INPUTS",
        ("benchmarks/sample.py",),
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
        "BENCHMARK_SOURCE_INPUTS",
        ("benchmarks/sample.py",),
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
        "BENCHMARK_SOURCE_INPUTS",
        ("benchmarks/sample.py",),
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
