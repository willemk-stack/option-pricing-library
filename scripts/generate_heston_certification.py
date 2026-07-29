"""Generate OMRR Heston certification from tests on a frozen OPL checkout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CERTIFIED_IMPLEMENTATION_COMMIT = "6daf359c2c5c534fc95991f6b37a332258b52099"
CERTIFICATE_SCHEMA_VERSION = "omrr_opl_heston_certification.v1"
CERTIFICATION_POLICY_VERSION = "omrr_opl_heston_certification_policy.v1"
EXPECTED_POLICY_SHA256 = (
    "7836958535e1f66a5c936952d75632f7dd4382b768c2db43c169929d95c48f9f"
)
SCRIPT_PATH = Path(__file__).resolve()
TOOLING_ROOT = SCRIPT_PATH.parents[1]
POLICY_DIR = TOOLING_ROOT / "certification" / "heston"
PROBE_PATH = SCRIPT_PATH.with_name("heston_certification_probe.py")

REQUIRED_IMPLEMENTATION_TESTS = (
    "tests/models/heston/test_heston_fourier_backends.py",
    "tests/models/heston/test_heston_reference_prices.py",
    "tests/models/heston/test_heston_numerical_smoke_regimes.py",
    "tests/models/heston/test_heston_numerical_robustness.py",
    "tests/models/heston/test_heston_quadrature_recommender.py",
    "tests/models/heston/test_fourier_diagnostics.py",
    "tests/models/heston/test_heston_params.py",
    "tests/pricers/test_heston_pricer_vectorization.py",
    "tests/market/test_market_parity.py",
    "tests/diagnostics/heston/test_heston_integration.py",
    "tests/diagnostics/heston/test_heston_pricing.py",
)

CERTIFICATE_FIELDS = {
    "schema_version",
    "certification_status",
    "opl_repository_identity",
    "opl_commit_sha",
    "opl_worktree_dirty",
    "certification_policy_version",
    "certification_policy_sha256",
    "created_at_utc",
    "pricer_identity",
    "supported_pricing_api",
    "supported_production_backend",
    "supported_production_quadrature_tier",
    "supported_option_rights",
    "certified_parameter_envelope",
    "certified_market_envelope",
    "numerical_tolerances",
    "test_suite_result_summary",
    "backend_comparison_evidence_summary",
    "stress_regime_evidence_summary",
    "warning_behavior_evidence_summary",
    "environment_metadata",
    "evidence_artifacts",
}


class CertificationError(RuntimeError):
    """Raised when mandatory certification evidence cannot be established."""


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and result.returncode != 0:
        raise CertificationError(
            f"command failed with exit status {result.returncode}: "
            f"{subprocess.list2cmdline(command)}\n{result.stdout}"
        )
    return result


def _git(path: Path, *args: str) -> str:
    return _run(["git", *args], cwd=path).stdout.strip()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    )


def _write_text_lf(path: Path, value: str) -> None:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(normalized.encode("utf-8"))


def _copy_text_lf(source: Path, destination: Path) -> None:
    _write_text_lf(destination, source.read_text(encoding="utf-8"))


def _normalize_text_file_lf(path: Path) -> None:
    _write_text_lf(path, path.read_text(encoding="utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: object) -> str:
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def validate_created_at_utc(value: str) -> str:
    if not value.endswith("Z"):
        raise CertificationError("created_at_utc must end in Z")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CertificationError("created_at_utc must be valid ISO-8601") from exc
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise CertificationError("created_at_utc must be UTC")
    return value


def verify_policy(policy: dict[str, Any]) -> str:
    digest = canonical_json_sha256(policy)
    if digest != EXPECTED_POLICY_SHA256:
        raise CertificationError(
            "normalized policy does not match OMRR's frozen policy SHA-256: "
            f"{digest}"
        )
    return digest


def verify_worktrees(
    *,
    implementation_worktree: Path,
    expected_implementation_commit: str,
) -> dict[str, str]:
    if not re.fullmatch(r"[0-9a-f]{40}", expected_implementation_commit):
        raise CertificationError("expected implementation commit must be a full SHA")
    implementation_root = Path(
        _git(implementation_worktree, "rev-parse", "--show-toplevel")
    ).resolve()
    if implementation_root != implementation_worktree.resolve():
        raise CertificationError(
            "implementation path must identify the root of its isolated worktree"
        )
    implementation_head = _git(implementation_root, "rev-parse", "HEAD")
    if implementation_head != expected_implementation_commit:
        raise CertificationError(
            "implementation worktree HEAD does not match the certified commit: "
            f"{implementation_head}"
        )
    implementation_status = _git(
        implementation_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    if implementation_status:
        raise CertificationError("implementation worktree is not clean")

    tooling_head = _git(TOOLING_ROOT, "rev-parse", "HEAD")
    tooling_status = _git(
        TOOLING_ROOT,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    if tooling_status:
        raise CertificationError(
            "certification tooling worktree must be committed and clean before "
            "production evidence is generated"
        )
    return {
        "implementation_root": str(implementation_root),
        "implementation_head": implementation_head,
        "implementation_origin": _git(
            implementation_root, "remote", "get-url", "origin"
        ),
        "tooling_root": str(TOOLING_ROOT),
        "tooling_head": tooling_head,
        "tooling_origin": _git(TOOLING_ROOT, "remote", "get-url", "origin"),
    }


def implementation_environment(
    implementation_root: Path,
    *,
    tooling_head: str,
) -> tuple[dict[str, Any], dict[str, str], str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(implementation_root / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    script = (
        "import json, platform, sys; "
        "import numpy, scipy, option_pricing; "
        "print(json.dumps({"
        "'python_executable': sys.executable,"
        "'python_version': platform.python_version(),"
        "'platform': platform.platform(),"
        "'architecture': platform.machine(),"
        "'numpy_version': numpy.__version__,"
        "'scipy_version': scipy.__version__,"
        "'option_pricing_module_path': option_pricing.__file__"
        "}, sort_keys=True))"
    )
    result = _run([sys.executable, "-c", script], cwd=implementation_root, env=env)
    metadata = json.loads(result.stdout)
    imported_path = Path(metadata["option_pricing_module_path"]).resolve()
    expected_source = (implementation_root / "src" / "option_pricing").resolve()
    if expected_source not in imported_path.parents:
        raise CertificationError(
            "option_pricing did not resolve from the certified implementation "
            f"checkout: {imported_path}"
        )
    dependencies = {
        "numpy": str(metadata["numpy_version"]),
        "scipy": str(metadata["scipy_version"]),
        "pytest": importlib.metadata.version("pytest"),
        "certification_tooling_commit_sha": tooling_head,
    }
    pip_freeze = _run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        cwd=implementation_root,
        env=env,
    ).stdout
    return metadata, dependencies, pip_freeze


def parse_junit(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    test_cases: list[str] = []
    warning_cases: list[str] = []
    for suite in suites:
        for name in totals:
            totals[name] += int(suite.attrib.get(name, "0"))
        for case in suite.iter("testcase"):
            identifier = (
                f"{case.attrib.get('classname', '')}.{case.attrib.get('name', '')}"
            )
            test_cases.append(identifier)
            if "warning" in identifier.lower():
                warning_cases.append(identifier)
    failed = totals["failures"] + totals["errors"]
    return {
        "passed": totals["tests"] - failed - totals["skipped"],
        "failed": failed,
        "skipped": totals["skipped"],
        "total": totals["tests"],
        "test_case_identifiers": sorted(test_cases),
        "warning_test_case_identifiers": sorted(warning_cases),
    }


def run_required_tests(
    implementation_root: Path,
    *,
    output_dir: Path,
    env: dict[str, str],
) -> tuple[dict[str, Any], list[str]]:
    junit_path = output_dir / "heston_certification_pytest.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        *REQUIRED_IMPLEMENTATION_TESTS,
        "-q",
        f"--junitxml={junit_path}",
    ]
    result = _run(command, cwd=implementation_root, env=env, check=False)
    _write_text_lf(
        output_dir / "heston_certification_pytest.log",
        result.stdout,
    )
    if junit_path.is_file():
        _normalize_text_file_lf(junit_path)
    if result.returncode != 0:
        raise CertificationError(
            "required Heston certification tests failed; no PASS certificate "
            f"was written (exit {result.returncode})"
        )
    summary = parse_junit(junit_path)
    summary["command"] = command
    summary["exit_status"] = result.returncode
    _write_json(output_dir / "heston_certification_test_summary.json", summary)
    return summary, command


def run_grid_probe(
    implementation_root: Path,
    *,
    output_dir: Path,
    scope_path: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    output_path = output_dir / "heston_numerical_grid_summary.json"
    command = [
        sys.executable,
        str(PROBE_PATH),
        "--scope",
        str(scope_path),
        "--output",
        str(output_path),
    ]
    result = _run(command, cwd=implementation_root, env=env, check=False)
    _write_text_lf(
        output_dir / "heston_numerical_grid.log",
        result.stdout,
    )
    if not output_path.is_file():
        raise CertificationError(
            "numerical grid probe did not produce structured evidence"
        )
    _normalize_text_file_lf(output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if result.returncode != 0 or payload.get("status") != "PASS":
        raise CertificationError(
            "numerical grid certification failed; inspect "
            f"{output_path} (exit {result.returncode})"
        )
    return payload


def build_certificate(
    *,
    scope: dict[str, Any],
    policy_sha256: str,
    worktree_evidence: dict[str, str],
    environment: dict[str, Any],
    dependencies: dict[str, str],
    test_summary: dict[str, Any],
    grid_summary: dict[str, Any],
    evidence_artifacts: list[dict[str, str]],
    created_at_utc: str,
) -> dict[str, Any]:
    backend = grid_summary["summaries"]["backend_comparison"]
    warning_identifiers = test_summary["warning_test_case_identifiers"]
    if not warning_identifiers:
        raise CertificationError("required suite produced no warning-behavior tests")
    payload = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "certification_status": "PASS",
        "opl_repository_identity": worktree_evidence["implementation_origin"],
        "opl_commit_sha": worktree_evidence["implementation_head"],
        "opl_worktree_dirty": False,
        "certification_policy_version": CERTIFICATION_POLICY_VERSION,
        "certification_policy_sha256": policy_sha256,
        "created_at_utc": created_at_utc,
        "pricer_identity": scope["pricer_identity"],
        "supported_pricing_api": scope["supported_pricing_api"],
        "supported_production_backend": scope["production_backend"],
        "supported_production_quadrature_tier": scope["production_quadrature_tier"],
        "supported_option_rights": scope["supported_option_rights"],
        "certified_parameter_envelope": scope["certified_parameter_envelope"],
        "certified_market_envelope": scope["certified_market_envelope"],
        "numerical_tolerances": scope["numerical_tolerances"],
        "test_suite_result_summary": {
            "identifiers": list(REQUIRED_IMPLEMENTATION_TESTS),
            "passed": test_summary["passed"],
            "failed": test_summary["failed"],
            "skipped": test_summary["skipped"],
        },
        "backend_comparison_evidence_summary": {
            "status": backend["status"],
            "test_identifiers": [
                "certification/heston/full_envelope_grid:robust_vs_adaptive",
                "tests/models/heston/test_heston_fourier_backends.py",
            ],
            "case_count": backend["case_count"],
            "maximum_absolute_difference": backend["value"],
            "absolute_tolerance": scope["numerical_tolerances"][
                "backend_absolute_price"
            ],
            "relative_tolerance": scope["numerical_tolerances"][
                "backend_relative_to_discounted_forward"
            ],
        },
        "stress_regime_evidence_summary": {
            "status": grid_summary["status"],
            "test_identifiers": [
                "certification/heston/full_envelope_grid",
                "tests/models/heston/test_heston_numerical_smoke_regimes.py",
                "tests/models/heston/test_heston_numerical_robustness.py",
            ],
            "case_count": grid_summary["certified_grid"]["case_count"],
        },
        "warning_behavior_evidence_summary": {
            "status": "PASS",
            "test_identifiers": warning_identifiers,
            "case_count": len(warning_identifiers),
            "warning_classes": [
                "HestonIntegralWarning",
                "RuntimeWarning promoted to structured numerical failure",
                "scipy.integrate.IntegrationWarning",
            ],
        },
        "environment_metadata": {
            "python_version": environment["python_version"],
            "platform": (
                f"{environment['platform']}; architecture={environment['architecture']}"
            ),
            "dependencies": dependencies,
        },
        "evidence_artifacts": evidence_artifacts,
    }
    if set(payload) != CERTIFICATE_FIELDS:
        raise CertificationError("internal error: certificate field set is not v1")
    if payload["opl_commit_sha"] != CERTIFIED_IMPLEMENTATION_COMMIT:
        raise CertificationError("internal error: wrong implementation identity")
    return payload


def generate(
    *,
    implementation_worktree: Path,
    output_dir: Path,
    expected_implementation_commit: str,
    created_at_utc: str | None,
) -> Path:
    worktrees = verify_worktrees(
        implementation_worktree=implementation_worktree,
        expected_implementation_commit=expected_implementation_commit,
    )
    if expected_implementation_commit != CERTIFIED_IMPLEMENTATION_COMMIT:
        raise CertificationError(
            "this policy certifies only the frozen OPL implementation commit"
        )
    if output_dir.exists():
        raise CertificationError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    policy_source = POLICY_DIR / "omrr_policy.normalized.json"
    scope_source = POLICY_DIR / "certification_scope.v1.json"
    catalog_source = POLICY_DIR / "coverage_catalog.v1.json"
    policy = json.loads(policy_source.read_text(encoding="utf-8"))
    scope = json.loads(scope_source.read_text(encoding="utf-8"))
    policy_sha = verify_policy(policy)
    if scope["certified_implementation_commit_sha"] != expected_implementation_commit:
        raise CertificationError("scope policy does not identify the expected commit")

    copied_policy = output_dir / policy_source.name
    copied_scope = output_dir / scope_source.name
    copied_catalog = output_dir / catalog_source.name
    _copy_text_lf(policy_source, copied_policy)
    _copy_text_lf(scope_source, copied_scope)
    _copy_text_lf(catalog_source, copied_catalog)

    implementation_root = Path(worktrees["implementation_root"])
    environment, dependencies, pip_freeze = implementation_environment(
        implementation_root,
        tooling_head=worktrees["tooling_head"],
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(implementation_root / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    test_summary, test_command = run_required_tests(
        implementation_root,
        output_dir=output_dir,
        env=env,
    )
    grid_summary = run_grid_probe(
        implementation_root,
        output_dir=output_dir,
        scope_path=copied_scope,
        env=env,
    )
    final_implementation_head = _git(implementation_root, "rev-parse", "HEAD")
    final_implementation_status = _git(
        implementation_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    if (
        final_implementation_head != expected_implementation_commit
        or final_implementation_status
    ):
        raise CertificationError(
            "certified implementation worktree changed while evidence was produced"
        )

    environment_evidence = {
        "schema_version": "opl_heston_certification_environment.v1",
        "certified_implementation_worktree": str(implementation_root),
        "certified_implementation_commit_sha": worktrees["implementation_head"],
        "certified_implementation_dirty": False,
        "certification_tooling_worktree": worktrees["tooling_root"],
        "certification_tooling_commit_sha": worktrees["tooling_head"],
        "python_executable": environment["python_executable"],
        "python_version": environment["python_version"],
        "platform": environment["platform"],
        "architecture": environment["architecture"],
        "numpy_version": environment["numpy_version"],
        "scipy_version": environment["scipy_version"],
        "option_pricing_module_path": environment["option_pricing_module_path"],
        "test_command": test_command,
    }
    environment_path = output_dir / "heston_certification_environment.json"
    _write_json(environment_path, environment_evidence)
    freeze_path = output_dir / "heston_certification_dependencies.txt"
    _write_text_lf(freeze_path, pip_freeze)

    evidence_paths = sorted(
        [
            copied_policy,
            copied_scope,
            copied_catalog,
            output_dir / "heston_certification_pytest.xml",
            output_dir / "heston_certification_pytest.log",
            output_dir / "heston_certification_test_summary.json",
            output_dir / "heston_numerical_grid_summary.json",
            output_dir / "heston_numerical_grid.log",
            environment_path,
            freeze_path,
        ],
        key=lambda path: path.name,
    )
    evidence_artifacts = [
        {"path": path.name, "sha256": sha256_file(path)} for path in evidence_paths
    ]
    if created_at_utc is None:
        created_at_utc = (
            datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        )
    created_at_utc = validate_created_at_utc(created_at_utc)
    certificate = build_certificate(
        scope=scope,
        policy_sha256=policy_sha,
        worktree_evidence=worktrees,
        environment=environment,
        dependencies=dependencies,
        test_summary=test_summary,
        grid_summary=grid_summary,
        evidence_artifacts=evidence_artifacts,
        created_at_utc=created_at_utc,
    )
    certificate_path = output_dir / "omrr_opl_heston_certification.v1.json"
    _write_json(certificate_path, certificate)
    print(f"certificate_path={certificate_path.resolve()}")
    print(f"certificate_sha256={sha256_file(certificate_path)}")
    return certificate_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Certify the frozen OPL Heston implementation for OMRR using a "
            "separate clean implementation worktree."
        )
    )
    parser.add_argument("--implementation-worktree", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-implementation-commit",
        default=CERTIFIED_IMPLEMENTATION_COMMIT,
    )
    parser.add_argument(
        "--created-at-utc",
        help="Optional reproducible ISO-8601 UTC creation time ending in Z.",
    )
    args = parser.parse_args()
    try:
        generate(
            implementation_worktree=args.implementation_worktree.resolve(),
            output_dir=args.output_dir.resolve(),
            expected_implementation_commit=args.expected_implementation_commit,
            created_at_utc=args.created_at_utc,
        )
    except CertificationError as exc:
        print(f"certification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
