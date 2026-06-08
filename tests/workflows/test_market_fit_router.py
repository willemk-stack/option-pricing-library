from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import option_pricing.workflows.router as market_router
from option_pricing.workflows import fit_market_model


@pytest.mark.parametrize(
    ("model", "delegate_name"),
    (
        ("heston", "fit_heston_from_bundle"),
        ("svi", "fit_svi_from_bundle"),
        ("essvi", "fit_essvi_from_bundle"),
    ),
)
def test_fit_market_model_delegates_to_explicit_one_shot_helper(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    delegate_name: str,
) -> None:
    bundle = SimpleNamespace(name="bundle")
    sentinel = SimpleNamespace(model=model)
    calls: list[tuple[object, dict[str, object]]] = []

    def fake_delegate(path_or_bundle: object, **kwargs: Any) -> object:
        calls.append((path_or_bundle, dict(kwargs)))
        return sentinel

    monkeypatch.setattr(market_router, delegate_name, fake_delegate)

    result = fit_market_model(
        model,  # type: ignore[arg-type]
        bundle,
        raise_on_failure=True,
        custom_option=3,
    )

    assert result is sentinel
    assert calls == [
        (
            bundle,
            {
                "raise_on_failure": True,
                "custom_option": 3,
            },
        )
    ]


def test_fit_market_model_rejects_non_market_fit_workflows() -> None:
    with pytest.raises(ValueError) as exc_info:
        fit_market_model("black_scholes", object())  # type: ignore[arg-type]

    message = str(exc_info.value)
    assert "Unsupported market-fit model 'black_scholes'" in message
    assert "'heston', 'svi', and 'essvi'" in message
    assert "Black-Scholes" in message
    assert "trees" in message
    assert "Monte Carlo" in message
    assert "PDE" in message
    assert "direct local-vol" in message
