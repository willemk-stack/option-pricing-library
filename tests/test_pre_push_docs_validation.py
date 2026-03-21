from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pre_push_docs_validation as hook


class _FakeProcess:
    def __init__(self, lines: list[str], return_code: int) -> None:
        self.stdout = lines
        self._return_code = return_code

    def wait(self) -> int:
        return self._return_code


def test_run_failure_includes_output_tail_and_snapshots_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_log = tmp_path / "docs_pre_push_last_run.log"
    failure_log = tmp_path / "docs_pre_push_last_failure.log"
    total_lines = hook.FAILURE_SUMMARY_TAIL_LINES + 5
    output_lines = [f"output line {index}\n" for index in range(total_lines)]

    monkeypatch.setattr(hook, "PRE_PUSH_LOG_DIR", tmp_path)
    monkeypatch.setattr(hook, "PRE_PUSH_RUN_LOG", run_log)
    monkeypatch.setattr(hook, "PRE_PUSH_FAILURE_LOG", failure_log)
    monkeypatch.setattr(
        hook.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakeProcess(output_lines, return_code=7),
    )

    hook.prepare_run_log()
    stage = hook.ValidationStage(
        label="Synthetic stage",
        failure_class="synthetic-failure",
        likely_layer="unit-test harness",
        next_step="Inspect the captured tail.",
    )

    with pytest.raises(hook.HookFailure) as exc_info:
        hook.run(["python", "-m", "synthetic"], stage=stage)

    failure = exc_info.value

    assert failure.exit_code == 7
    assert failure.log_path == failure_log
    assert "Synthetic stage failed." in str(failure)
    assert "Output tail:" in str(failure)
    assert "output line 0" not in str(failure)
    assert f"output line {total_lines - 1}" in str(failure)

    run_log_text = run_log.read_text(encoding="utf8")
    failure_log_text = failure_log.read_text(encoding="utf8")

    assert "[Synthetic stage]" in run_log_text
    assert "> python -m synthetic" in run_log_text
    assert "output line 0" in run_log_text
    assert f"output line {total_lines - 1}" in run_log_text
    assert failure_log_text == run_log_text


def test_prepare_run_log_replaces_stale_failure_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_log = tmp_path / "docs_pre_push_last_run.log"
    failure_log = tmp_path / "docs_pre_push_last_failure.log"

    monkeypatch.setattr(hook, "PRE_PUSH_LOG_DIR", tmp_path)
    monkeypatch.setattr(hook, "PRE_PUSH_RUN_LOG", run_log)
    monkeypatch.setattr(hook, "PRE_PUSH_FAILURE_LOG", failure_log)

    failure_log.write_text("stale failure\n", encoding="utf8")

    hook.prepare_run_log()

    assert run_log.read_text(encoding="utf8") == "Docs pre-push guard log\n"
    assert failure_log.read_text(encoding="utf8") == "Docs pre-push guard log\n"


def test_ensure_clean_generated_diff_detects_untracked_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args, **kwargs):
        return hook.subprocess.CompletedProcess(
            args[0],
            0,
            stdout="?? docs/assets/generated/new-figure.svg\n",
            stderr="",
        )

    monkeypatch.setattr(hook.subprocess, "run", fake_run)

    with pytest.raises(
        hook.HookFailure, match="tracked or untracked generated-file drift"
    ):
        hook.ensure_clean_generated_diff(
            ["docs/assets/generated"],
            label="Generated docs asset tree",
        )


def test_main_readme_only_fast_mode_skips_playwright_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    monkeypatch.setattr(
        hook,
        "parse_args",
        lambda: argparse.Namespace(mode="fast", files=["README.template.md"]),
    )
    monkeypatch.setattr(hook, "prepare_run_log", lambda: None)
    monkeypatch.setattr(hook, "clear_failure_log", lambda: None)
    monkeypatch.setattr(hook, "resolve_python_command", lambda: "python")
    monkeypatch.setattr(
        hook,
        "ensure_playwright_dependencies",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected Playwright check")),
    )
    monkeypatch.setattr(
        hook,
        "docker_available_detail",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected Docker check")),
    )
    monkeypatch.setattr(
        hook,
        "run",
        lambda command, **kwargs: commands.append(command),
    )
    monkeypatch.setattr(
        hook, "ensure_clean_generated_diff", lambda *args, **kwargs: None
    )

    assert hook.main() == 0
    assert commands == [["python", "scripts/render_readme.py", "--check"]]


def test_main_docs_site_fast_mode_skips_browser_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    monkeypatch.setattr(
        hook,
        "parse_args",
        lambda: argparse.Namespace(mode="fast", files=["docs/index.md"]),
    )
    monkeypatch.setattr(hook, "prepare_run_log", lambda: None)
    monkeypatch.setattr(hook, "clear_failure_log", lambda: None)
    monkeypatch.setattr(hook, "resolve_python_command", lambda: "python")
    monkeypatch.setattr(
        hook,
        "ensure_playwright_dependencies",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected Playwright check")),
    )
    monkeypatch.setattr(
        hook,
        "docker_available_detail",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected Docker check")),
    )
    monkeypatch.setattr(
        hook,
        "run",
        lambda command, **kwargs: commands.append(command),
    )
    monkeypatch.setattr(
        hook, "ensure_clean_generated_diff", lambda *args, **kwargs: None
    )

    assert hook.main() == 0
    assert [command[1:] for command in commands] == [
        ["scripts/check_docs_source_links.py"],
        ["-m", "mkdocs", "build", "--strict"],
        ["scripts/visual_audit/check_svg_assets.py"],
    ]


def test_main_manual_mode_checks_playwright_for_docs_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        hook,
        "parse_args",
        lambda: argparse.Namespace(mode="manual", files=["docs/index.md"]),
    )
    monkeypatch.setattr(hook, "prepare_run_log", lambda: None)
    monkeypatch.setattr(hook, "clear_failure_log", lambda: None)
    monkeypatch.setattr(hook, "resolve_python_command", lambda: "python")
    monkeypatch.setattr(
        hook,
        "ensure_playwright_dependencies",
        lambda: calls.append("playwright"),
    )
    monkeypatch.setattr(hook, "docker_available_detail", lambda: (True, None))
    monkeypatch.setattr(hook, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        hook, "ensure_clean_generated_diff", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(hook, "playwright_command", lambda *args: ["playwright", *args])

    assert hook.main() == 0
    assert calls == ["playwright"]


def test_main_benchmark_sensitive_src_change_checks_snapshot_freshness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    monkeypatch.setattr(
        hook,
        "parse_args",
        lambda: argparse.Namespace(
            mode="fast",
            files=["src/option_pricing/models/black_scholes/bs.py"],
        ),
    )
    monkeypatch.setattr(hook, "prepare_run_log", lambda: None)
    monkeypatch.setattr(hook, "clear_failure_log", lambda: None)
    monkeypatch.setattr(hook, "resolve_python_command", lambda: "python")
    monkeypatch.setattr(
        hook,
        "ensure_playwright_dependencies",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected Playwright check")),
    )
    monkeypatch.setattr(
        hook,
        "docker_available_detail",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected Docker check")),
    )
    monkeypatch.setattr(
        hook,
        "run",
        lambda command, **kwargs: commands.append(command),
    )
    monkeypatch.setattr(
        hook, "ensure_clean_generated_diff", lambda *args, **kwargs: None
    )

    assert hook.main() == 0
    assert ["python", "scripts/build_benchmark_artifacts.py", "--check"] in commands
