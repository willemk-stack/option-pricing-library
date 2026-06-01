from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "demo_local_market_validation.py"
MKDOCS_PATH = ROOT / "mkdocs.yml"
GUIDE_INDEX_PATH = ROOT / "docs" / "user_guides" / "index.md"
MARKET_SNAPSHOT_GUIDE_PATH = (
    ROOT / "docs" / "user_guides" / "market_snapshot_validation.md"
)
README_TEMPLATE_PATH = ROOT / "README.template.md"
README_PATH = ROOT / "README.md"
README_RENDER_SCRIPT = ROOT / "scripts" / "render_readme.py"
EXPECTED_BUNDLE_FILES = {
    "manifest.json",
    "market_data.json",
    "cleaned_quotes.parquet",
    "rejected_quotes.parquet",
    "heston_quotes.parquet",
    "surface_inputs.parquet",
    "heston_fit_summary.csv",
    "warnings.json",
}


def _load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_parquet(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_to_parquet(
        self: pd.DataFrame,
        path: str,
        compression: str | None = None,
        index: bool = False,
    ) -> None:
        del compression
        payload = self if index else self.reset_index(drop=True)
        payload.to_pickle(path)

    def _fake_read_parquet(
        path: str,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        frame = cast(pd.DataFrame, pd.read_pickle(path))
        if columns is None:
            return frame
        return cast(pd.DataFrame, frame.loc[:, columns])

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _fake_to_parquet)
    monkeypatch.setattr(pd, "read_parquet", _fake_read_parquet)


def _load_demo_module() -> ModuleType:
    return _load_module(SCRIPT_PATH, "demo_local_market_validation")


def _bundle_root(root: Path, run_id: str = "demo-run") -> Path:
    return (
        root
        / "gold"
        / "model_validation_bundle"
        / "underlying=SYNTH"
        / "date=2026-05-22"
        / f"run_id={run_id}"
    )


def _stub_pipeline_result(root: Path, run_id: str) -> SimpleNamespace:
    bundle_root = _bundle_root(root, run_id=run_id)
    return SimpleNamespace(
        local_snapshot=SimpleNamespace(
            underlying="SYNTH",
            run_id=run_id,
            asof=pd.Timestamp("2026-05-22T00:00:00Z"),
        ),
        bronze_paths=SimpleNamespace(manifest=root / "bronze" / "manifest.json"),
        silver_paths=SimpleNamespace(manifest=root / "silver" / "manifest.json"),
        model_validation_bundle=SimpleNamespace(
            manifest_path=bundle_root / "manifest.json",
            metadata=SimpleNamespace(run_id=run_id),
        ),
    )


def test_demo_runs_local_pipeline_and_prints_reviewer_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fake_parquet: None,
) -> None:
    demo = _load_demo_module()
    output_dir = tmp_path / "marketdata-demo"

    exit_code = demo.main(
        [
            "--output-dir",
            str(output_dir),
            "--run-id",
            "demo-run",
        ]
    )

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Local market validation demo completed." in stdout
    assert "underlying: SYNTH" in stdout
    assert "run_id: demo-run" in stdout
    assert "valuation_date: 2026-05-22" in stdout
    assert "bronze_manifest:" in stdout
    assert "silver_manifest:" in stdout
    assert "market_data:" in stdout
    assert "heston_quotes:" in stdout
    assert "bundle_manifest:" in stdout
    assert "warnings:" in stdout
    assert "heston_fit_summary:" in stdout
    assert "No live providers or credentials were used." in stdout

    bundle_root = _bundle_root(output_dir)
    assert {path.name for path in bundle_root.iterdir()} == EXPECTED_BUNDLE_FILES
    for filename in EXPECTED_BUNDLE_FILES:
        assert (bundle_root / filename).exists()

    manifest = json.loads((bundle_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["heston_smoke"]["status"] == "skipped"

    rerun_exit_code = demo.main(
        [
            "--output-dir",
            str(output_dir),
            "--run-id",
            "demo-run",
            "--overwrite",
        ]
    )
    assert rerun_exit_code == 0


def test_heston_smoke_flags_map_to_bundle_config_without_changing_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = _load_demo_module()
    requested_smoke_modes: list[bool] = []

    def _fake_pipeline(**kwargs: object) -> SimpleNamespace:
        bundle_config = kwargs["bundle_config"]
        requested_smoke_modes.append(bundle_config.run_heston_smoke)
        return _stub_pipeline_result(
            Path(cast(Path, kwargs["storage"])),
            cast(str, kwargs["run_id"]),
        )

    monkeypatch.setattr(demo, "run_local_model_validation_pipeline", _fake_pipeline)

    assert demo.main(["--output-dir", str(tmp_path / "default")]) == 0
    assert (
        demo.main(["--output-dir", str(tmp_path / "skip"), "--skip-heston-smoke"]) == 0
    )
    assert demo.main(["--output-dir", str(tmp_path / "run"), "--run-heston-smoke"]) == 0

    assert requested_smoke_modes == [False, False, True]


def test_demo_script_imports_are_local_only() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    imported_names: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imported_names.append(node.module)
            imported_names.extend(alias.name for alias in node.names)

    forbidden_fragments = (
        "alpaca",
        "fred",
        "yahoo",
        "yfinance",
        "requests",
        "fredapi",
        "duckdb",
        "research_exports",
        "refresh",
        "option_pricing.marketdata.providers",
    )
    lowered_imports = [name.lower() for name in imported_names]

    for imported_name in lowered_imports:
        assert not any(fragment in imported_name for fragment in forbidden_fragments)


def _nav_section(nav: list[object], section_name: str) -> list[dict[str, str]]:
    for item in nav:
        if isinstance(item, dict) and section_name in item:
            section = item[section_name]
            assert isinstance(section, list)
            return cast(list[dict[str, str]], section)
    raise AssertionError(f"Missing MkDocs nav section: {section_name}")


def _nav_paths(section: list[dict[str, str]]) -> list[str]:
    paths: list[str] = []
    for item in section:
        assert isinstance(item, dict)
        assert len(item) == 1
        value = next(iter(item.values()))
        assert isinstance(value, str)
        paths.append(value)
    return paths


def test_market_snapshot_page_is_on_proof_path_not_quickstart() -> None:
    mkdocs = yaml.safe_load(MKDOCS_PATH.read_text(encoding="utf-8"))
    nav = mkdocs["nav"]
    proof_path = _nav_section(nav, "Proof path")
    quickstart = _nav_section(nav, "Quickstart")

    proof_paths = _nav_paths(proof_path)
    quickstart_paths = _nav_paths(quickstart)

    assert "user_guides/market_snapshot_validation.md" in proof_paths
    assert "user_guides/market_snapshot_validation.md" not in quickstart_paths
    assert {
        "Market snapshot validation": "user_guides/market_snapshot_validation.md"
    } in proof_path


def test_a6_docs_are_discoverable_from_index_and_readme() -> None:
    guide_index = GUIDE_INDEX_PATH.read_text(encoding="utf-8")
    readme_template = README_TEMPLATE_PATH.read_text(encoding="utf-8")
    readme = README_PATH.read_text(encoding="utf-8")

    index_line = (
        "- [Market snapshot validation](market_snapshot_validation.md) - "
        "local fixture-to-artifact validation for reviewer reproducibility "
        "with no live providers or credentials."
    )
    readme_line = (
        "- [Market snapshot validation]"
        "(https://willemk-stack.github.io/option-pricing-library/"
        "user_guides/market_snapshot_validation/) for the local "
        "fixture-to-artifact reviewer workflow with no live providers or "
        "credentials"
    )

    assert index_line in guide_index
    assert readme_line in readme_template
    assert readme_line in readme


def test_readme_is_rendered_from_template() -> None:
    render_readme = _load_module(README_RENDER_SCRIPT, "render_readme_for_a6")

    assert (
        render_readme.render(
            template_path=README_TEMPLATE_PATH,
            out_path=README_PATH,
            check=True,
        )
        == 0
    )


def test_a6_docs_keep_local_only_scope_guardrails() -> None:
    guide = MARKET_SNAPSHOT_GUIDE_PATH.read_text(encoding="utf-8")
    guide_index = GUIDE_INDEX_PATH.read_text(encoding="utf-8")
    readme_template = README_TEMPLATE_PATH.read_text(encoding="utf-8")
    combined_lines = (guide + "\n" + guide_index + "\n" + readme_template).splitlines()
    combined = "\n".join(combined_lines).lower()

    assert "no live providers" in combined
    assert "no provider credentials" in combined or "no credentials" in combined
    assert "provider refresh cli or production cli" in combined
    assert "no research exports" in combined

    phrases_requiring_non_goal_context = (
        "production-ready market data",
        "production data quality",
        "live alpaca",
        "live fred",
        "trading signal",
        "trading performance",
        "research conclusion",
        "empirical finding",
        "real-time market data",
    )
    non_goal_markers = (
        "does not",
        "not ",
        "no ",
        "without",
        "non-goal",
        "outside",
        "future-phase",
    )

    lowered_lines = [line.lower() for line in combined_lines]
    for phrase in phrases_requiring_non_goal_context:
        for index, line in enumerate(lowered_lines):
            if phrase not in line:
                continue
            context = " ".join(
                lowered_lines[max(index - 1, 0) : min(index + 2, len(lowered_lines))]
            )
            assert any(
                marker in context for marker in non_goal_markers
            ), f"{phrase!r} must be framed as a non-goal or boundary"
