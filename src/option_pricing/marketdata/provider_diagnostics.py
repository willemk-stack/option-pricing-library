from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from importlib import import_module
from typing import Any, cast

from option_pricing.marketdata.config import ProviderRetryConfig
from option_pricing.marketdata.errors import (
    MissingProviderCredentialError,
    ProviderDataUnavailableError,
    ProviderRequestError,
)
from option_pricing.marketdata.provider_results import ProviderCallDiagnostic
from option_pricing.marketdata.provider_serialization import (
    _sanitized_request_metadata,
)

_MAX_DIAGNOSTIC_MESSAGE_CHARS = 240


class ProviderCallFailedError(RuntimeError):
    """Internal wrapper carrying sanitized provider-call diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic: ProviderCallDiagnostic,
        original: BaseException,
        failure_kind: str,
    ) -> None:
        self.diagnostic = diagnostic
        self.original = original
        self.failure_kind = failure_kind
        super().__init__(message)


def _call_provider_with_diagnostic[T](
    *,
    provider: str,
    operation: str,
    request_metadata: Mapping[str, Any],
    retry_config: ProviderRetryConfig,
    call: Callable[[], T],
    count_items: Callable[[T], int | None] | None = None,
) -> tuple[T, ProviderCallDiagnostic]:
    started_at = datetime.now(UTC)
    retry_count = 0
    sanitized_request = _sanitized_request_metadata(request_metadata)
    try:
        result, retry_count = _execute_with_optional_tenacity_retry(
            call,
            retry_config=retry_config,
        )
    except _RetryWrappedError as wrapped:
        retry_count = wrapped.retry_count
        diagnostic = _provider_call_diagnostic(
            provider=provider,
            operation=operation,
            status="failed",
            request_metadata=sanitized_request,
            started_at=started_at,
            retry_count=retry_count,
            exception=wrapped.original,
        )
        raise ProviderCallFailedError(
            "provider call failed",
            diagnostic=diagnostic,
            original=wrapped.original,
            failure_kind=_provider_failure_kind(wrapped.original),
        ) from None
    except Exception as exc:
        diagnostic = _provider_call_diagnostic(
            provider=provider,
            operation=operation,
            status="failed",
            request_metadata=sanitized_request,
            started_at=started_at,
            retry_count=retry_count,
            exception=exc,
        )
        raise ProviderCallFailedError(
            "provider call failed",
            diagnostic=diagnostic,
            original=exc,
            failure_kind=_provider_failure_kind(exc),
        ) from None

    item_count = None if count_items is None else count_items(result)
    diagnostic = _provider_call_diagnostic(
        provider=provider,
        operation=operation,
        status=("empty" if item_count == 0 else "ok"),
        request_metadata=sanitized_request,
        started_at=started_at,
        retry_count=retry_count,
        rows_or_contracts_in=item_count,
    )
    return result, diagnostic


def _diagnostics_payload(
    diagnostics: Sequence[ProviderCallDiagnostic],
) -> list[dict[str, object]]:
    return [diagnostic.as_dict() for diagnostic in diagnostics]


def _normalization_failure_diagnostic(
    *,
    provider: str,
    operation: str,
    request_metadata: Mapping[str, Any],
    exception: BaseException,
) -> ProviderCallDiagnostic:
    started_at = datetime.now(UTC)
    return _provider_call_diagnostic(
        provider=provider,
        operation=operation,
        status="failed",
        request_metadata=_sanitized_request_metadata(request_metadata),
        started_at=started_at,
        retry_count=0,
        exception=exception,
        message=_sanitized_exception_message(exception),
    )


class _RetryWrappedError(RuntimeError):
    def __init__(self, original: BaseException, retry_count: int) -> None:
        self.original = original
        self.retry_count = retry_count
        super().__init__(str(type(original).__name__))


def _execute_with_optional_tenacity_retry[T](
    call: Callable[[], T],
    *,
    retry_config: ProviderRetryConfig,
) -> tuple[T, int]:
    if not retry_config.retry_enabled or retry_config.max_attempts <= 1:
        return call(), 0

    try:
        tenacity = import_module("tenacity")
    except ModuleNotFoundError:
        return call(), 0

    attempts = 0
    retrying = tenacity.Retrying(
        retry=tenacity.retry_if_exception(_is_transient_provider_exception),
        stop=tenacity.stop_after_attempt(retry_config.max_attempts),
        wait=tenacity.wait_exponential(
            multiplier=retry_config.wait_initial_seconds,
            max=retry_config.wait_max_seconds,
        ),
        reraise=True,
    )
    try:
        for attempt in retrying:
            with attempt:
                attempts += 1
                return call(), max(0, attempts - 1)
    except Exception as exc:
        raise _RetryWrappedError(exc, max(0, attempts - 1)) from None

    return call(), max(0, attempts - 1)


def _provider_call_diagnostic(
    *,
    provider: str,
    operation: str,
    status: str,
    request_metadata: Mapping[str, object],
    started_at: datetime,
    retry_count: int,
    exception: BaseException | None = None,
    message: str | None = None,
    rows_or_contracts_in: int | None = None,
    rows_or_contracts_out: int | None = None,
) -> ProviderCallDiagnostic:
    ended_at = datetime.now(UTC)
    elapsed_ms = (ended_at - started_at).total_seconds() * 1000.0
    diagnostic_message = message
    if diagnostic_message is None and exception is not None:
        diagnostic_message = _sanitized_exception_message(exception)
    return ProviderCallDiagnostic(
        provider=provider,
        operation=operation,
        status=cast(Any, status),
        request_metadata=_sanitized_request_metadata(request_metadata),
        started_at=_datetime_isoformat(started_at),
        ended_at=_datetime_isoformat(ended_at),
        elapsed_ms=round(elapsed_ms, 3),
        exception_type=None if exception is None else type(exception).__name__,
        message=diagnostic_message,
        retry_count=retry_count,
        rows_or_contracts_in=rows_or_contracts_in,
        rows_or_contracts_out=rows_or_contracts_out,
    )


def _is_transient_provider_exception(exc: BaseException) -> bool:
    if isinstance(exc, MissingProviderCredentialError):
        return False
    if isinstance(exc, ValueError | TypeError):
        return False
    if isinstance(exc, ProviderRequestError):
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            return True
        try:
            code = int(status_code)
        except (TypeError, ValueError):
            return True
        return code == 408 or code == 429 or code >= 500
    if isinstance(exc, TimeoutError | ConnectionError):
        return True
    return False


def _datetime_isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _provider_failure_kind(exc: BaseException) -> str:
    if isinstance(exc, MissingProviderCredentialError):
        return "missing_credentials"
    if isinstance(exc, ProviderRequestError):
        return "provider_request_failed"
    if isinstance(exc, ProviderDataUnavailableError):
        return "provider_returned_empty"
    if isinstance(exc, ValueError | TypeError):
        return "validation_failed"
    return "provider_request_failed"


def _sanitized_exception_message(exc: BaseException) -> str:
    if isinstance(
        exc,
        MissingProviderCredentialError
        | ProviderRequestError
        | ProviderDataUnavailableError,
    ):
        message = str(exc)
        cause_message = _sanitized_cause_message(exc)
        if cause_message is not None:
            message = f"{message}; cause={cause_message}"
    elif isinstance(exc, ValueError | TypeError):
        message = str(exc)
    else:
        message = f"{type(exc).__name__} raised by provider call"
    return _truncate_diagnostic_message(_redact_message_fragments(message))


def _sanitized_cause_message(exc: BaseException) -> str | None:
    cause = exc.__cause__
    if cause is None:
        return None
    cause_message = str(cause).strip()
    if not cause_message:
        return None
    return _redact_message_fragments(cause_message)


def _redact_message_fragments(message: str) -> str:
    lowered = message.lower()
    secret_markers = (
        "api_key",
        "apikey",
        "secret",
        "token",
        "authorization",
        "password",
    )
    if any(marker in lowered for marker in secret_markers):
        return "<redacted>"
    return message


def _truncate_diagnostic_message(message: str) -> str:
    compact = " ".join(str(message).split())
    if len(compact) <= _MAX_DIAGNOSTIC_MESSAGE_CHARS:
        return compact
    return compact[: _MAX_DIAGNOSTIC_MESSAGE_CHARS - 3] + "..."
