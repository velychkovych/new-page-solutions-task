"""The only retry layer for tool calls (CONTRACTS.md §2.5).

`call_with_retry` depends only on what it is given: the request callable, a clock, a
sleep function and a uniform RNG. Tests drive it with fakes; nothing here sleeps for
real or reads the wall clock on its own.
"""

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Self

import httpx
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field

from helpdesk.shared import trace_attributes as attr
from helpdesk.shared.settings import Settings

logger = logging.getLogger(__name__)

# Receives the timeout for this attempt (seconds) and performs one HTTP request.
type SendAttempt = Callable[[float], Awaitable[httpx.Response]]
# Monotonic seconds.
type Clock = Callable[[], float]
type Sleep = Callable[[float], Awaitable[None]]
# Same contract as random.uniform(a, b).
type Uniform = Callable[[float, float], float]
# Called before each retry with (next attempt number, max attempts).
type OnRetry = Callable[[int, int], Awaitable[None]]

RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})
# ConnectError covers refused/unresolvable hosts; TimeoutException covers httpx's
# connect/read/write/pool timeouts; TimeoutError is our own per-attempt timeout.
_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    httpx.ConnectError,
    httpx.TimeoutException,
    TimeoutError,
)
RETRY_AFTER_HEADER = "Retry-After"


class RetryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_attempts: int = Field(ge=1)
    backoff_base_s: float = Field(gt=0)
    retry_after_cap_s: float = Field(gt=0)
    attempt_timeout_s: float = Field(gt=0)
    deadline_s: float = Field(gt=0)

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            max_attempts=settings.retry_max_attempts,
            backoff_base_s=settings.retry_backoff_base_s,
            retry_after_cap_s=settings.retry_after_cap_s,
            attempt_timeout_s=settings.tool_attempt_timeout_s,
            deadline_s=settings.tool_deadline_s,
        )


@dataclass(frozen=True, slots=True)
class RetryResult:
    """A final (non-retryable) response; its status may still be an error such as 404."""

    response: httpx.Response
    attempts: int


class UpstreamCallError(Exception):
    """`last_status` is the last attempt's HTTP status; `last_failure` says how it failed
    when there was no response. Both are safe to show: our wording, never upstream bytes."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        last_status: int | None,
        last_failure: str | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_status = last_status
        self.last_failure = last_failure


class DeadlineExceededError(UpstreamCallError):
    """The total deadline was reached (maps to TIMEOUT)."""


class UpstreamUnavailableError(UpstreamCallError):
    """Attempts ran out before the deadline, or a non-retryable httpx error (maps to
    UPSTREAM_UNAVAILABLE)."""


def retry_after_s(response: httpx.Response) -> float | None:
    """Delta-seconds form only. The mock sends integers; an HTTP-date would need the wall
    clock, which this module deliberately does not read, so it falls back to jitter."""
    raw = response.headers.get(RETRY_AFTER_HEADER)
    if raw is None:
        return None
    try:
        value = float(raw.strip())
    except ValueError:
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def describe_failure(exc: Exception) -> str:
    """How an attempt without a response failed. Only the exception type is used: its
    message can carry upstream bytes."""
    match exc:
        case httpx.ConnectError():
            return "could not connect"
        case httpx.TimeoutException() | TimeoutError():
            return "timed out"
        case _:
            return f"failed with {type(exc).__name__}"


def wait_before_attempt(
    attempt: int, response: httpx.Response | None, policy: RetryPolicy, uniform: Uniform
) -> float:
    """Seconds to wait before `attempt` (n >= 2)."""
    if response is not None and (server_wait := retry_after_s(response)) is not None:
        return min(server_wait, policy.retry_after_cap_s)
    return uniform(0.0, policy.backoff_base_s * 2 ** (attempt - 2))


def _record_attempt(attempt: int, status: int | None, wait_s: float) -> None:
    attributes: dict[str, int] = {
        attr.RETRY_ATTEMPT: attempt,
        attr.RETRY_WAIT_MS: round(wait_s * 1000),
    }
    if status is not None:
        attributes[attr.RETRY_HTTP_STATUS] = status
    trace.get_current_span().add_event(attr.EVENT_RETRY, attributes=attributes)


async def call_with_retry(
    send: SendAttempt,
    *,
    policy: RetryPolicy,
    clock: Clock,
    sleep: Sleep,
    uniform: Uniform,
    on_retry: OnRetry | None = None,
) -> RetryResult:
    """Run `send` until it returns a non-retryable response, or give up.

    Raises DeadlineExceededError when the deadline passed during an attempt or the next
    wait would end after it, and UpstreamUnavailableError when the attempts run out first.
    Each attempt adds a `retry.attempt` event to the current span.
    """
    deadline = clock() + policy.deadline_s
    attempt = 0
    while True:
        attempt += 1
        remaining = deadline - clock()
        if remaining <= 0:
            # Only reachable when a sleep overshot; the wait check below normally stops first.
            raise DeadlineExceededError(
                "deadline reached before the next attempt",
                attempts=attempt - 1,
                last_status=None,
            )
        timeout_s = min(policy.attempt_timeout_s, remaining)

        response: httpx.Response | None = None
        failure: str | None = None
        try:
            async with asyncio.timeout(timeout_s):
                response = await send(timeout_s)
        except _RETRYABLE_ERRORS as exc:
            failure = describe_failure(exc)
            logger.debug("attempt %d failed: %s", attempt, type(exc).__name__)
        except httpx.HTTPError as exc:
            # Any other httpx failure (protocol error, bad Content-Encoding, redirect loop)
            # will not get better on retry.
            _record_attempt(attempt, None, 0.0)
            raise UpstreamUnavailableError(
                f"HTTP error: {type(exc).__name__}",
                attempts=attempt,
                last_status=None,
                last_failure=describe_failure(exc),
            ) from exc

        status = None if response is None else response.status_code
        if response is not None and response.status_code not in RETRYABLE_STATUS:
            _record_attempt(attempt, status, 0.0)
            return RetryResult(response=response, attempts=attempt)

        if clock() >= deadline:
            _record_attempt(attempt, status, 0.0)
            raise DeadlineExceededError(
                "deadline reached during an attempt",
                attempts=attempt,
                last_status=status,
                last_failure=failure,
            )
        if attempt >= policy.max_attempts:
            _record_attempt(attempt, status, 0.0)
            raise UpstreamUnavailableError(
                "retry attempts exhausted",
                attempts=attempt,
                last_status=status,
                last_failure=failure,
            )
        wait_s = wait_before_attempt(attempt + 1, response, policy, uniform)
        if clock() + wait_s >= deadline:
            _record_attempt(attempt, status, 0.0)
            raise DeadlineExceededError(
                "the next wait would end after the deadline",
                attempts=attempt,
                last_status=status,
                last_failure=failure,
            )

        _record_attempt(attempt, status, wait_s)
        logger.debug(
            "retrying in %.3f s (attempt %d of %d)", wait_s, attempt + 1, policy.max_attempts
        )
        if on_retry is not None:
            await on_retry(attempt + 1, policy.max_attempts)
        await sleep(wait_s)
