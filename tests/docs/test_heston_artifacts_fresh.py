from __future__ import annotations

import scripts.check_heston_artifacts_fresh as check_heston_artifacts_fresh


def test_committed_heston_artifacts_are_fresh(capsys) -> None:
    assert check_heston_artifacts_fresh.main([]) == 0

    output = capsys.readouterr().out
    assert "Heston docs artifacts are up to date." in output
