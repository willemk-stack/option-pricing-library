"""Configuration contracts for marketdata providers, pipeline, and storage."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, cast

DIVIDEND_POLICY_ZERO_ASSUMPTION = "zero_assumption"
DIVIDEND_POLICY_MANUAL_STATIC = "manual_static"
DIVIDEND_POLICY_PROVIDER_TRAILING_YIELD = "provider_trailing_yield"
DIVIDEND_POLICY_IMPLIED_CARRY = "implied_carry"
SUPPORTED_DIVIDEND_POLICIES = frozenset(
    {
        DIVIDEND_POLICY_ZERO_ASSUMPTION,
        DIVIDEND_POLICY_MANUAL_STATIC,
        DIVIDEND_POLICY_PROVIDER_TRAILING_YIELD,
        DIVIDEND_POLICY_IMPLIED_CARRY,
    }
)


@dataclass(frozen=True, slots=True, init=False)
class AlpacaConfig:
    api_key_env: str = "ALPACA_API_KEY"
    secret_key_env: str = "ALPACA_SECRET_KEY"
    equity_feed: str = "iex"
    option_feed: str = "indicative"
    sandbox: bool = False

    def __init__(
        self,
        api_key_env: str = "ALPACA_API_KEY",
        secret_key_env: str = "ALPACA_SECRET_KEY",
        feed: str | None = None,
        sandbox: bool = False,
        *,
        equity_feed: str | None = None,
        option_feed: str | None = None,
    ) -> None:
        resolved_equity_feed = "iex" if equity_feed is None else equity_feed
        resolved_option_feed = (
            feed
            if option_feed is None and feed is not None
            else ("indicative" if option_feed is None else option_feed)
        )

        object.__setattr__(self, "api_key_env", api_key_env)
        object.__setattr__(self, "secret_key_env", secret_key_env)
        object.__setattr__(self, "equity_feed", resolved_equity_feed)
        object.__setattr__(self, "option_feed", resolved_option_feed)
        object.__setattr__(self, "sandbox", sandbox)
        self._validate(legacy_feed_provided=feed is not None)

    def __post_init__(self) -> None:
        self._validate(legacy_feed_provided=False)

    def _validate(self, *, legacy_feed_provided: bool) -> None:
        _validate_non_empty_string(self.api_key_env, "alpaca.api_key_env")
        _validate_non_empty_string(self.secret_key_env, "alpaca.secret_key_env")
        _validate_non_empty_string(self.equity_feed, "alpaca.equity_feed")
        _validate_non_empty_string(
            self.option_feed,
            "alpaca.feed" if legacy_feed_provided else "alpaca.option_feed",
        )

    @property
    def feed(self) -> str:
        """Backward-compatible alias for the option snapshot feed."""

        return self.option_feed


@dataclass(frozen=True, slots=True)
class FredConfig:
    api_key_env: str = "FRED_API_KEY"
    base_url: str = "https://api.stlouisfed.org/fred"

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.api_key_env, "fred.api_key_env")
        _validate_non_empty_string(self.base_url, "fred.base_url")


@dataclass(frozen=True, slots=True)
class StorageConfig:
    root: Path
    compression: str = "zstd"

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise TypeError("storage.root must be a pathlib.Path")


@dataclass(frozen=True, slots=True)
class ProviderRetryConfig:
    retry_enabled: bool = True
    max_attempts: int = 3
    wait_initial_seconds: float = 0.05
    wait_max_seconds: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.retry_enabled, bool):
            raise TypeError("retry.retry_enabled must be a boolean")
        _validate_positive_int(self.max_attempts, "retry.max_attempts")
        _validate_nonnegative_number(
            self.wait_initial_seconds,
            "retry.wait_initial_seconds",
        )
        _validate_nonnegative_number(
            self.wait_max_seconds,
            "retry.wait_max_seconds",
        )


@dataclass(frozen=True, slots=True)
class StaticDividendYield:
    dividend_yield: float
    source: str = DIVIDEND_POLICY_MANUAL_STATIC
    note: str | None = None

    def __post_init__(self) -> None:
        _validate_nonnegative_finite_number(
            self.dividend_yield,
            "dividends.static_yields.dividend_yield",
        )
        _validate_non_empty_string(self.source, "dividends.static_yields.source")
        _validate_dividend_policy_name(self.source, "dividends.static_yields.source")
        if self.note is not None:
            _validate_non_empty_string(self.note, "dividends.static_yields.note")


@dataclass(frozen=True, slots=True)
class DividendPolicyConfig:
    default_policy: str = DIVIDEND_POLICY_ZERO_ASSUMPTION
    static_yields: Mapping[str, StaticDividendYield] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_dividend_policy_name(
            self.default_policy,
            "dividends.default_policy",
        )
        if not isinstance(self.static_yields, Mapping):
            raise TypeError("dividends.static_yields must be a mapping")

        normalized: dict[str, StaticDividendYield] = {}
        for symbol, value in self.static_yields.items():
            cleaned_symbol = _validate_symbol_key(symbol)
            normalized[cleaned_symbol] = _coerce_static_dividend_yield(
                value,
                f"dividends.static_yields.{cleaned_symbol}",
            )
        object.__setattr__(self, "static_yields", normalized)


@dataclass(frozen=True, slots=True)
class MarketDataPolicyConfig:
    dividends: DividendPolicyConfig = field(default_factory=DividendPolicyConfig)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> MarketDataPolicyConfig:
        if not isinstance(payload, Mapping):
            raise TypeError("policy config payload must be a mapping")
        dividends_payload = payload.get("dividends", {})
        if not isinstance(dividends_payload, Mapping):
            raise TypeError("policy config 'dividends' must be a mapping")
        static_payload = dividends_payload.get("static_yields", {})
        if not isinstance(static_payload, Mapping):
            raise TypeError("policy config dividends.static_yields must be a mapping")
        return cls(
            dividends=DividendPolicyConfig(
                default_policy=str(
                    dividends_payload.get(
                        "default_policy",
                        DIVIDEND_POLICY_ZERO_ASSUMPTION,
                    )
                ),
                static_yields={
                    str(symbol): _coerce_static_dividend_yield(
                        value,
                        f"dividends.static_yields.{symbol}",
                    )
                    for symbol, value in static_payload.items()
                },
            )
        )

    @classmethod
    def from_file(cls, path: str | Path) -> MarketDataPolicyConfig:
        config_path = Path(path)
        suffix = config_path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        elif suffix == ".toml":
            payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
        elif suffix in {".yaml", ".yml"}:
            try:
                yaml = cast(Any, import_module("yaml"))
            except ImportError as exc:  # pragma: no cover - optional dependency guard
                raise ImportError(
                    "Reading YAML marketdata policy configs requires PyYAML"
                ) from exc
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        else:
            raise ValueError(
                "policy config file must end with .json, .toml, .yaml, or .yml"
            )
        if payload is None:
            payload = {}
        return cls.from_mapping(cast(Mapping[str, object], payload))


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    alpaca: AlpacaConfig
    fred: FredConfig
    storage: StorageConfig
    retry: ProviderRetryConfig = field(default_factory=ProviderRetryConfig)
    policy: MarketDataPolicyConfig = field(default_factory=MarketDataPolicyConfig)


def _coerce_static_dividend_yield(
    value: object,
    field_name: str,
) -> StaticDividendYield:
    if isinstance(value, StaticDividendYield):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping or StaticDividendYield")
    dividend_yield = value.get("dividend_yield")
    if dividend_yield is None:
        raise ValueError(f"{field_name}.dividend_yield is required")
    return StaticDividendYield(
        dividend_yield=float(cast(Any, dividend_yield)),
        source=str(value.get("source", DIVIDEND_POLICY_MANUAL_STATIC)),
        note=(None if value.get("note") is None else str(cast(Any, value.get("note")))),
    )


def _validate_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1")


def _validate_nonnegative_number(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be numeric")
    if float(value) < 0.0:
        raise ValueError(f"{field_name} must be >= 0")


def _validate_nonnegative_finite_number(value: float, field_name: str) -> None:
    _validate_nonnegative_number(value, field_name)
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{field_name} must be finite")


def _validate_symbol_key(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("dividends.static_yields keys must be strings")
    cleaned = value.strip().upper()
    if not cleaned:
        raise ValueError("dividends.static_yields keys must be non-empty")
    return cleaned


def _validate_dividend_policy_name(value: str, field_name: str) -> None:
    _validate_non_empty_string(value, field_name)
    if value not in SUPPORTED_DIVIDEND_POLICIES:
        expected = ", ".join(sorted(SUPPORTED_DIVIDEND_POLICIES))
        raise ValueError(f"{field_name} must be one of: {expected}")


__all__ = [
    "AlpacaConfig",
    "DIVIDEND_POLICY_IMPLIED_CARRY",
    "DIVIDEND_POLICY_MANUAL_STATIC",
    "DIVIDEND_POLICY_PROVIDER_TRAILING_YIELD",
    "DIVIDEND_POLICY_ZERO_ASSUMPTION",
    "DividendPolicyConfig",
    "FredConfig",
    "MarketDataPolicyConfig",
    "PipelineConfig",
    "ProviderRetryConfig",
    "StorageConfig",
    "StaticDividendYield",
    "SUPPORTED_DIVIDEND_POLICIES",
]
