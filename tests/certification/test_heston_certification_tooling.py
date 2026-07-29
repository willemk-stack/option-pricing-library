"""Contract tests for the Heston certification generator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_generator() -> ModuleType:
    path = ROOT / "scripts" / "generate_heston_certification.py"
    spec = importlib.util.spec_from_file_location("heston_certificate_generator", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    return _load_generator()


def _inputs(generator: ModuleType) -> dict[str, object]:
    scope = json.loads(
        (ROOT / "certification" / "heston" / "certification_scope.v2.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "scope": scope,
        "policy_sha256": generator.EXPECTED_POLICY_SHA256,
        "worktree_evidence": {
            "implementation_origin": (
                "https://github.com/willemk-stack/option-pricing-library.git"
            ),
            "implementation_head": generator.CERTIFIED_IMPLEMENTATION_COMMIT,
        },
        "environment": {
            "python_version": "3.12.0",
            "platform": "test-platform",
            "architecture": "AMD64",
        },
        "dependencies": {
            "numpy": "1.26.4",
            "scipy": "1.14.1",
            "pytest": "9.0.2",
            "certification_tooling_commit_sha": "1" * 40,
        },
        "test_summary": {
            "passed": 10,
            "failed": 0,
            "skipped": 1,
            "warning_test_case_identifiers": ["suite.test_warning_policy"],
        },
        "grid_summary": {
            "status": "PASS",
            "certified_grid": {"case_count": 384},
            "summaries": {
                "backend_comparison": {
                    "status": "PASS",
                    "case_count": 384,
                    "value": 0.0004,
                }
            },
        },
        "evidence_artifacts": [{"path": "evidence.json", "sha256": "2" * 64}],
        "created_at_utc": "2026-07-29T12:00:00Z",
    }


def test_normalized_policy_matches_frozen_omrr_digest(
    generator: ModuleType,
) -> None:
    policy = json.loads(
        (
            ROOT / "certification" / "heston" / "omrr_policy.normalized.v2.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        generator.verify_policy(policy)
        == "dc115c2e4d67e8bd8c9937f62e4a1a1f239d6cf62cf6574dd70b1b401d538af1"
    )


def test_certificate_builder_emits_exact_omrr_v2_fields(
    generator: ModuleType,
) -> None:
    payload = generator.build_certificate(**_inputs(generator))

    assert set(payload) == generator.CERTIFICATE_FIELDS
    assert payload["schema_version"] == "omrr_opl_heston_certification.v2"
    assert payload["opl_commit_sha"] == "41c01d886aeddf87d6837927be63d5041cfc2f89"
    assert len(payload["certified_parameter_regimes"]) == 6
    assert payload["opl_worktree_dirty"] is False
    assert (
        payload["environment_metadata"]["dependencies"][
            "certification_tooling_commit_sha"
        ]
        == "1" * 40
    )
    assert "certification_tooling_commit_sha" not in payload


def test_certificate_builder_requires_warning_evidence(
    generator: ModuleType,
) -> None:
    inputs = _inputs(generator)
    inputs["test_summary"]["warning_test_case_identifiers"] = []

    with pytest.raises(
        generator.CertificationError,
        match="no warning-behavior tests",
    ):
        generator.build_certificate(**inputs)


@pytest.mark.parametrize(
    "value",
    [
        "2026-07-29T12:00:00+01:00",
        "not-a-timestampZ",
    ],
)
def test_creation_time_must_be_valid_utc(
    generator: ModuleType,
    value: str,
) -> None:
    with pytest.raises(generator.CertificationError):
        generator.validate_created_at_utc(value)


def test_junit_parser_collects_structured_counts(
    generator: ModuleType,
    tmp_path: Path,
) -> None:
    path = tmp_path / "results.xml"
    path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" errors="0" failures="1" skipped="1" tests="4">
    <testcase classname="suite" name="test_pass" />
    <testcase classname="suite" name="test_warning_policy" />
    <testcase classname="suite" name="test_skip"><skipped /></testcase>
    <testcase classname="suite" name="test_fail"><failure /></testcase>
  </testsuite>
</testsuites>
""",
        encoding="utf-8",
    )

    summary = generator.parse_junit(path)

    assert summary["total"] == 4
    assert summary["passed"] == 2
    assert summary["failed"] == 1
    assert summary["skipped"] == 1
    assert summary["warning_test_case_identifiers"] == ["suite.test_warning_policy"]


def test_hash_bound_writers_use_platform_independent_lf(
    generator: ModuleType,
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "evidence.json"
    text_path = tmp_path / "evidence.txt"

    generator._write_json(json_path, {"lines": ["one", "two"]})
    generator._write_text_lf(text_path, "one\r\ntwo\rthree\n")

    assert b"\r" not in json_path.read_bytes()
    assert text_path.read_bytes() == b"one\ntwo\nthree\n"
