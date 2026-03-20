from __future__ import annotations

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
