"""The retry module as a pure function of (request, clock, sleep, rng). No real sleeps."""

import asyncio
from collections.abc import Sequence

import httpx
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer
from tests.unit.mcp_server.conftest import FakeClock, RecordingUniform, make_settings

from helpdesk.mcp_server.retry import (
    RETRYABLE_STATUS,
    DeadlineExceededError,
    RetryPolicy,
    RetryResult,
    UpstreamUnavailableError,
    call_with_retry,
    retry_after_s,
)
from helpdesk.shared import trace_attributes as attr

POLICY = RetryPolicy(
    max_attempts=3,
    backoff_base_s=0.5,
    retry_after_cap_s=5.0,
    attempt_timeout_s=5.0,
    deadline_s=15.0,
)

type Outcome = httpx.Response | Exception


class ScriptedSend:
    """Returns (or raises) the scripted outcomes in order, advancing the fake clock by
    `elapsed_s` per attempt, or by the whole attempt timeout when it raises a timeout."""

    def __init__(
        self, clock: FakeClock, outcomes: Sequence[Outcome], *, elapsed_s: float = 0.0
    ) -> None:
        self._clock = clock
        self._outcomes = list(outcomes)
        self._elapsed_s = elapsed_s
        self.timeouts: list[float] = []

    async def __call__(self, timeout_s: float) -> httpx.Response:
        self.timeouts.append(timeout_s)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, httpx.TimeoutException):
            self._clock.now += timeout_s
            raise outcome
        self._clock.now += self._elapsed_s
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def response(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, headers=headers, json={"error": "x"})


async def run(
    send: ScriptedSend,
    clock: FakeClock,
    uniform: RecordingUniform,
    policy: RetryPolicy = POLICY,
    on_retry: list[tuple[int, int]] | None = None,
) -> RetryResult:
    async def record(attempt: int, max_attempts: int) -> None:
        if on_retry is not None:
            on_retry.append((attempt, max_attempts))

    return await call_with_retry(
        send, policy=policy, clock=clock, sleep=clock.sleep, uniform=uniform, on_retry=record
    )


def retry_events(exporter: InMemorySpanExporter) -> list[dict[str, object]]:
    (span,) = exporter.get_finished_spans()
    return [dict(e.attributes or {}) for e in span.events if e.name == attr.EVENT_RETRY]


async def test_success_on_first_attempt(clock: FakeClock, uniform: RecordingUniform) -> None:
    send = ScriptedSend(clock, [response(200)])
    result = await run(send, clock, uniform)
    assert result.attempts == 1
    assert result.response.status_code == 200
    assert clock.sleeps == []


@pytest.mark.parametrize("status", sorted(RETRYABLE_STATUS))
async def test_retryable_status_is_retried(
    status: int, clock: FakeClock, uniform: RecordingUniform
) -> None:
    send = ScriptedSend(clock, [response(status), response(200)])
    result = await run(send, clock, uniform)
    assert result.attempts == 2
    assert result.response.status_code == 200


def test_retry_set_is_exactly_the_contract() -> None:
    assert frozenset({429, 500, 502, 503, 504}) == RETRYABLE_STATUS


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422, 501])
async def test_other_statuses_are_not_retried(
    status: int, clock: FakeClock, uniform: RecordingUniform
) -> None:
    send = ScriptedSend(clock, [response(status)])
    result = await run(send, clock, uniform)
    assert result.attempts == 1
    assert result.response.status_code == status
    assert clock.sleeps == []


async def test_404_is_returned_without_retry(clock: FakeClock, uniform: RecordingUniform) -> None:
    send = ScriptedSend(clock, [response(404), response(200)])
    result = await run(send, clock, uniform)
    assert result.response.status_code == 404
    assert len(send.timeouts) == 1


async def test_429_honours_retry_after(clock: FakeClock, uniform: RecordingUniform) -> None:
    send = ScriptedSend(clock, [response(429, {"Retry-After": "2"}), response(200)])
    await run(send, clock, uniform)
    assert clock.sleeps == [2.0]
    assert uniform.calls == []  # Retry-After replaces the jitter


async def test_retry_after_is_capped_at_5_s(clock: FakeClock, uniform: RecordingUniform) -> None:
    send = ScriptedSend(clock, [response(429, {"Retry-After": "30"}), response(200)])
    await run(send, clock, uniform)
    assert clock.sleeps == [5.0]


async def test_unparseable_retry_after_falls_back_to_jitter(
    clock: FakeClock, uniform: RecordingUniform
) -> None:
    date = "Wed, 21 Oct 2026 07:28:00 GMT"
    send = ScriptedSend(clock, [response(429, {"Retry-After": date}), response(200)])
    await run(send, clock, uniform)
    assert uniform.calls == [(0.0, 0.5)]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1", 1.0), (" 2.5 ", 2.5), ("0", 0.0), ("-1", None), ("nan", None), ("soon", None)],
)
def test_retry_after_parsing(raw: str, expected: float | None) -> None:
    assert retry_after_s(httpx.Response(429, headers={"Retry-After": raw})) == expected
    assert retry_after_s(httpx.Response(429)) is None


async def test_retries_stop_at_3_attempts(clock: FakeClock, uniform: RecordingUniform) -> None:
    send = ScriptedSend(clock, [response(503)] * 5)
    with pytest.raises(UpstreamUnavailableError) as caught:
        await run(send, clock, uniform)
    assert caught.value.attempts == 3
    assert caught.value.last_status == 503
    assert len(send.timeouts) == 3
    assert len(clock.sleeps) == 2


@pytest.mark.parametrize(
    ("outcomes", "last_status", "last_failure"),
    [
        ([response(503)] * 3, 503, None),
        (
            [response(503), response(503), httpx.ConnectError("RAW refused")],
            None,
            "could not connect",
        ),
        (
            [httpx.ConnectError("x"), httpx.ConnectError("x"), httpx.ReadTimeout("RAW")],
            None,
            "timed out",
        ),
        ([httpx.ConnectError("x"), httpx.ConnectError("x"), response(429)], 429, None),
        ([httpx.RemoteProtocolError("RAW bad framing")], None, "failed with RemoteProtocolError"),
    ],
    ids=["503", "connect-error", "read-timeout", "failure-then-429", "protocol-error"],
)
async def test_unavailable_says_how_the_last_attempt_failed(
    outcomes: list[Outcome],
    last_status: int | None,
    last_failure: str | None,
    clock: FakeClock,
    uniform: RecordingUniform,
) -> None:
    """Only our wording and the exception type, never its message (which is upstream's)."""
    with pytest.raises(UpstreamUnavailableError) as caught:
        await run(ScriptedSend(clock, outcomes), clock, uniform)
    assert (caught.value.last_status, caught.value.last_failure) == (last_status, last_failure)


async def test_deadline_says_how_the_last_attempt_failed(
    clock: FakeClock, uniform: RecordingUniform
) -> None:
    policy = POLICY.model_copy(update={"deadline_s": 4.0, "attempt_timeout_s": 5.0})
    send = ScriptedSend(clock, [httpx.ReadTimeout("slow")])
    with pytest.raises(DeadlineExceededError) as caught:
        await run(send, clock, uniform, policy=policy)
    assert (caught.value.attempts, caught.value.last_failure) == (1, "timed out")


@pytest.mark.parametrize("fraction", [0.0, 0.37, 1.0])
async def test_full_jitter_wait_bounds(fraction: float, clock: FakeClock) -> None:
    uniform = RecordingUniform(fraction)
    send = ScriptedSend(clock, [response(500), response(502), response(200)])
    await run(send, clock, uniform)
    # Before attempt n the wait is uniform(0, base * 2^(n-2)): (0, 0.5) then (0, 1.0).
    assert uniform.calls == [(0.0, 0.5), (0.0, 1.0)]
    assert clock.sleeps == pytest.approx([0.5 * fraction, 1.0 * fraction])


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("refused"),
        httpx.ConnectTimeout("connect timeout"),
        httpx.ReadTimeout("read timeout"),
    ],
    ids=["connect-error", "connect-timeout", "read-timeout"],
)
async def test_connect_errors_and_timeouts_are_retried(
    error: Exception, clock: FakeClock, uniform: RecordingUniform
) -> None:
    send = ScriptedSend(clock, [error, response(200)])
    result = await run(send, clock, uniform)
    assert result.attempts == 2


async def test_other_transport_errors_are_not_retried(
    clock: FakeClock, uniform: RecordingUniform
) -> None:
    send = ScriptedSend(clock, [httpx.RemoteProtocolError("bad framing"), response(200)])
    with pytest.raises(UpstreamUnavailableError) as caught:
        await run(send, clock, uniform)
    assert caught.value.attempts == 1
    assert len(send.timeouts) == 1


@pytest.mark.parametrize(
    "error",
    [httpx.DecodingError("bad gzip"), httpx.TooManyRedirects("loop")],
    ids=["decoding-error", "too-many-redirects"],
)
async def test_non_transport_http_errors_are_unavailable_without_retry(
    error: Exception, clock: FakeClock, uniform: RecordingUniform
) -> None:
    send = ScriptedSend(clock, [error, response(200)])
    with pytest.raises(UpstreamUnavailableError) as caught:
        await run(send, clock, uniform)
    assert caught.value.attempts == 1
    assert len(send.timeouts) == 1


async def test_per_attempt_timeout_is_enforced(clock: FakeClock, uniform: RecordingUniform) -> None:
    policy = POLICY.model_copy(update={"attempt_timeout_s": 0.01})
    calls = 0

    async def hang_then_answer(timeout_s: float) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.Event().wait()  # never set: only the attempt timeout ends it
        return response(200)

    result = await call_with_retry(
        hang_then_answer, policy=policy, clock=clock, sleep=clock.sleep, uniform=uniform
    )
    assert result.attempts == 2


async def test_attempt_timeout_is_min_of_5_s_and_time_left(
    clock: FakeClock, uniform: RecordingUniform
) -> None:
    send = ScriptedSend(clock, [httpx.ReadTimeout("t")] * 3)
    with pytest.raises(DeadlineExceededError) as caught:
        await run(send, clock, uniform)
    # t=0 -> 5 s attempt, wait 0.5 -> t=5.5 -> 5 s attempt, wait 1.0 -> t=11.5 -> 3.5 s left.
    assert send.timeouts == pytest.approx([5.0, 5.0, 3.5])
    assert caught.value.attempts == 3
    assert clock.now == pytest.approx(15.0)


async def test_deadline_passing_during_an_attempt_is_timeout(
    clock: FakeClock, uniform: RecordingUniform
) -> None:
    send = ScriptedSend(clock, [response(503), response(200)], elapsed_s=16.0)
    with pytest.raises(DeadlineExceededError) as caught:
        await run(send, clock, uniform)
    assert caught.value.attempts == 1
    assert clock.sleeps == []


async def test_wait_ending_after_deadline_is_timeout(
    clock: FakeClock, uniform: RecordingUniform
) -> None:
    send = ScriptedSend(clock, [response(429, {"Retry-After": "5"}), response(200)], elapsed_s=11.0)
    with pytest.raises(DeadlineExceededError) as caught:
        await run(send, clock, uniform)
    assert caught.value.attempts == 1
    assert clock.sleeps == []  # it gives up instead of sleeping past the deadline


async def test_attempts_exhausted_before_deadline_is_unavailable_not_timeout(
    clock: FakeClock, uniform: RecordingUniform
) -> None:
    send = ScriptedSend(clock, [response(503)] * 3, elapsed_s=1.0)
    with pytest.raises(UpstreamUnavailableError):
        await run(send, clock, uniform)
    assert clock.now < POLICY.deadline_s


async def test_on_retry_is_called_before_each_retry(
    clock: FakeClock, uniform: RecordingUniform
) -> None:
    calls: list[tuple[int, int]] = []
    send = ScriptedSend(clock, [response(503), response(503), response(200)])
    await run(send, clock, uniform, on_retry=calls)
    assert calls == [(2, 3), (3, 3)]


async def test_every_attempt_is_a_span_event(
    clock: FakeClock, uniform: RecordingUniform, tracer: Tracer, span_exporter: InMemorySpanExporter
) -> None:
    send = ScriptedSend(
        clock,
        [response(429, {"Retry-After": "1"}), httpx.ConnectError("refused"), response(201)],
    )
    with tracer.start_as_current_span("tool"):
        await run(send, clock, uniform)
    assert retry_events(span_exporter) == [
        {attr.RETRY_ATTEMPT: 1, attr.RETRY_HTTP_STATUS: 429, attr.RETRY_WAIT_MS: 1000},
        # No status code on a connect error; wait is the jitter for attempt 3 (max 1.0 s).
        {attr.RETRY_ATTEMPT: 2, attr.RETRY_WAIT_MS: 1000},
        {attr.RETRY_ATTEMPT: 3, attr.RETRY_HTTP_STATUS: 201, attr.RETRY_WAIT_MS: 0},
    ]


async def test_last_attempt_event_has_zero_wait_when_giving_up(
    clock: FakeClock, uniform: RecordingUniform, tracer: Tracer, span_exporter: InMemorySpanExporter
) -> None:
    send = ScriptedSend(clock, [response(503)] * 3)
    with tracer.start_as_current_span("tool"), pytest.raises(UpstreamUnavailableError):
        await run(send, clock, uniform)
    events = retry_events(span_exporter)
    assert [e[attr.RETRY_ATTEMPT] for e in events] == [1, 2, 3]
    assert [e[attr.RETRY_WAIT_MS] for e in events] == [500, 1000, 0]


def test_policy_comes_from_settings() -> None:
    settings = make_settings(
        retry_max_attempts=4,
        retry_backoff_base_s=0.25,
        retry_after_cap_s=2.0,
        tool_attempt_timeout_s=3.0,
        tool_deadline_s=9.0,
    )
    assert RetryPolicy.from_settings(settings) == RetryPolicy(
        max_attempts=4,
        backoff_base_s=0.25,
        retry_after_cap_s=2.0,
        attempt_timeout_s=3.0,
        deadline_s=9.0,
    )
