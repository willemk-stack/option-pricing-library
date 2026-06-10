from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import option_pricing.vol.svi.repair as repair


def _check(
    ok: bool, min_g: float = 0.0, failure_reason: str | None = None
) -> SimpleNamespace:
    return SimpleNamespace(ok=ok, min_g=min_g, failure_reason=failure_reason)


def test_gj_section51_targets_clips_tiny_negative_c_and_rejects_invalid_values() -> (
    None
):
    c_ast, vtilde_ast = repair._gj_section51_targets(
        SimpleNamespace(p=1.0, psi=-0.5 - 1.0e-15, v=0.2)
    )
    assert c_ast == 0.0
    assert vtilde_ast == 0.0

    with pytest.raises(ValueError, match="Invalid c_ast"):
        repair._gj_section51_targets(SimpleNamespace(p=1.0, psi=-1.0, v=0.2))

    with pytest.raises(ValueError, match="p \\+ c_ast"):
        repair._gj_section51_targets(SimpleNamespace(p=0.0, psi=0.0, v=0.2))


def test_normalized_black_call_handles_zero_variance_intrinsic_and_clips() -> None:
    y = np.array([-1.0, 0.0, 1.0])
    w = np.array([0.0, 0.04, 0.04])

    prices = repair._normalized_black_call_from_total_variance(y, w)

    assert prices[0] == pytest.approx(1.0 - np.exp(-1.0))
    assert np.all((prices >= 0.0) & (prices <= 1.0))


def test_repair_with_fallback_returns_already_ok_without_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = object()
    monkeypatch.setattr(
        repair, "check_butterfly_arbitrage", lambda *a, **k: _check(True, 0.1)
    )

    fixed, check, attempts = repair.repair_butterfly_with_fallback(
        original,
        T=1.0,
        y_domain_hint=(-1.0, 1.0),
    )

    assert fixed is original
    assert check.ok
    assert attempts == [
        {
            "method": "already_ok",
            "ok": True,
            "min_g": 0.1,
            "failure_reason": "",
            "error": "",
        }
    ]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"T": 0.0, "y_domain_hint": (-1.0, 1.0)}, "T must be > 0"),
        ({"T": 1.0, "y_domain_hint": (1.0, 1.0)}, "finite ordered pair"),
        ({"T": 1.0, "y_domain_hint": (np.nan, 1.0)}, "finite ordered pair"),
    ],
)
def test_repair_with_fallback_validates_outer_inputs(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        repair.repair_butterfly_with_fallback(object(), **kwargs)


def test_repair_with_fallback_logs_jw_failure_then_raw_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = object()
    raw_fixed = object()
    calls = {"check": 0}

    def fake_check(candidate: object, **_kwargs: object) -> SimpleNamespace:
        calls["check"] += 1
        if candidate is original:
            return _check(False, -1.0, "pre")
        return _check(True, 0.01, None)

    def fake_jw(*_args: object, **_kwargs: object) -> object:
        raise ValueError("jw failed")

    def fake_raw(*_args: object, **_kwargs: object) -> object:
        return raw_fixed

    monkeypatch.setattr(repair, "check_butterfly_arbitrage", fake_check)
    monkeypatch.setattr(repair, "repair_butterfly_jw_optimal", fake_jw)
    monkeypatch.setattr(repair, "repair_butterfly_raw", fake_raw)

    fixed, check, attempts = repair.repair_butterfly_with_fallback(
        original,
        T=1.0,
        y_domain_hint=(-1.0, 1.0),
        raw_methods=("line_search",),
    )

    assert fixed is raw_fixed
    assert check.ok
    assert [entry["method"] for entry in attempts] == ["jw_optimal", "line_search"]
    assert attempts[0]["ok"] is False
    assert "ValueError: jw failed" in str(attempts[0]["error"])


def test_repair_with_fallback_raises_after_all_methods_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        repair, "check_butterfly_arbitrage", lambda *a, **k: _check(False)
    )
    monkeypatch.setattr(
        repair,
        "repair_butterfly_raw",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("raw failed")),
    )

    with pytest.raises(RuntimeError, match="all configured methods"):
        repair.repair_butterfly_with_fallback(
            object(),
            T=1.0,
            y_domain_hint=(-1.0, 1.0),
            try_jw_optimal=False,
            raw_methods=("line_search", "project"),
        )
