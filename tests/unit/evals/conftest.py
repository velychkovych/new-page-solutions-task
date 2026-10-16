"""Builders for the evals unit tests: spans, responses, cases and evidence, all in memory.

Nothing here starts a process or touches the network.
"""

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import JsonValue

from evals.cases import Case, MessageTurn
from evals.evidence import CaseEvidence, Facts, TurnRun
from evals.traces import TurnTrace
from helpdesk.shared import trace_attributes as ta
from helpdesk.shared.api import (
    ChatRequest,
    ChatResponse,
    ChatStatus,
    PendingActionView,
    ToolCallRecord,
)
from helpdesk.shared.tickets import WritesReport
from helpdesk.telemetry.openinference_keys import (
    LLM_MODEL_NAME,
    LLM_TOKENS_CACHE_READ,
    LLM_TOKENS_CACHE_WRITE,
    LLM_TOKENS_COMPLETION,
    LLM_TOKENS_PROMPT,
    SPAN_KIND,
    SPAN_KIND_LLM,
)
from helpdesk.telemetry.records import SpanEventRecord, SpanRecord

T0 = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
TRACE_ID = "0123456789abcdef0123456789abcdef"
_span_counter = iter(range(1, 1_000_000))


def span(
    name: str,
    *,
    service: str = "agent",
    attributes: dict[str, JsonValue] | None = None,
    events: Sequence[SpanEventRecord] = (),
    offset_ms: int = 0,
    trace_id: str = TRACE_ID,
) -> SpanRecord:
    start = T0 + timedelta(milliseconds=offset_ms)
    return SpanRecord(
        trace_id=trace_id,
        span_id=f"{next(_span_counter):016x}",
        parent_span_id=None,
        name=name,
        service=service,
        start_time=start,
        end_time=start + timedelta(milliseconds=5),
        duration_ms=5.0,
        status="ok",
        attributes=attributes or {},
        events=list(events),
    )


def request_span(trace_id: str = TRACE_ID) -> SpanRecord:
    return span(ta.SPAN_REQUEST, trace_id=trace_id)


def retry_event(attempt: int, status: int | None) -> SpanEventRecord:
    attributes: dict[str, JsonValue] = {ta.RETRY_ATTEMPT: attempt, ta.RETRY_WAIT_MS: 0}
    if status is not None:
        attributes[ta.RETRY_HTTP_STATUS] = status
    return SpanEventRecord(name=ta.EVENT_RETRY, timestamp=T0, attributes=attributes)


def tool_span(
    tool: str,
    *,
    args: str = "{}",
    result: str = "",
    error_code: str | None = None,
    statuses: Sequence[int | None] = (200,),
    offset_ms: int = 0,
    trace_id: str = TRACE_ID,
) -> SpanRecord:
    attributes: dict[str, JsonValue] = {
        ta.TOOL_NAME: tool,
        ta.TOOL_ARGS: args,
        ta.TOOL_RESULT: result,
        ta.TOOL_ATTEMPTS: len(statuses),
    }
    if error_code is not None:
        attributes[ta.TOOL_ERROR_CODE] = error_code
    events = [retry_event(i, status) for i, status in enumerate(statuses, start=1)]
    return span(
        ta.SPAN_TOOL,
        service="mcp",
        attributes=attributes,
        events=events,
        offset_ms=offset_ms,
        trace_id=trace_id,
    )


def mock_span(name: str = "GET /tickets", *, trace_id: str = TRACE_ID) -> SpanRecord:
    return span(name, service="mock", trace_id=trace_id)


def llm_span(
    messages: Sequence[tuple[str, str]] = (("system", "You are the IT helpdesk assistant."),),
    *,
    model: str | None = "claude-test",
    tokens: tuple[int, int, int, int] = (0, 0, 0, 0),
    trace_id: str = TRACE_ID,
) -> SpanRecord:
    attributes: dict[str, JsonValue] = {SPAN_KIND: SPAN_KIND_LLM, "llm.provider": "anthropic"}
    if model is not None:
        attributes[LLM_MODEL_NAME] = model
    for i, (role, content) in enumerate(messages):
        attributes[f"llm.input_messages.{i}.message.role"] = role
        attributes[f"llm.input_messages.{i}.message.content"] = content
    keys = (LLM_TOKENS_PROMPT, LLM_TOKENS_COMPLETION, LLM_TOKENS_CACHE_READ, LLM_TOKENS_CACHE_WRITE)
    attributes.update(dict(zip(keys, tokens, strict=True)))
    return span("ChatAnthropic", attributes=attributes, trace_id=trace_id)


def tool_call(
    name: str, *, latency_ms: int = 25, error: str | None = None, result: str = "ok"
) -> ToolCallRecord:
    return ToolCallRecord(
        name=name, args={}, result_summary=result, error=error, latency_ms=latency_ms
    )


def pending(
    action_id: str = "action-1",
    tool: str = "create_ticket",
    args: dict[str, JsonValue] | None = None,
) -> PendingActionView:
    return PendingActionView(
        action_id=action_id,
        tool=tool,
        args=args if args is not None else {"title": "Printer jam", "description": "Jams"},
        summary=f"{tool} {args}",
    )


def response(
    *,
    status: ChatStatus = ChatStatus.OK,
    reply: str = "Done.",
    tool_calls: Iterable[ToolCallRecord] = (),
    pending_action: PendingActionView | None = None,
    trace_id: str = TRACE_ID,
) -> ChatResponse:
    if pending_action is not None:
        status = ChatStatus.AWAITING_APPROVAL
    return ChatResponse(
        reply=reply,
        status=status,
        tool_calls=list(tool_calls),
        pending_action=pending_action,
        trace_id=trace_id,
    )


BASE_CASE: dict[str, Any] = {
    "id": "unit_case",
    "title": "A case for unit tests",
    "plan_case": 1,
    "user_id": "alice",
    "turns": [{"message": "My printer jams."}],
    "fake": [[{"content": "ok"}]],
}


def make_case(**overrides: Any) -> Case:
    return Case.model_validate({**BASE_CASE, **overrides})


def turn_run(
    resp: ChatResponse,
    spans: Iterable[SpanRecord] = (),
    *,
    index: int = 1,
    case: Case | None = None,
    message: str | None = "My printer jams.",
    writes_before: int = 0,
    writes_after: int = 0,
) -> TurnRun:
    turn = (case or make_case()).turns[index - 1] if case is not None else None
    if turn is None:
        turn = MessageTurn(message=message or "x")
    request = (
        ChatRequest(user_id="alice", session_id="s1", message=message)
        if message is not None
        else ChatRequest.model_validate(
            {
                "user_id": "alice",
                "session_id": "s1",
                "decision": {"action_id": "a", "approve": True},
            }
        )
    )
    return TurnRun(
        index=index,
        turn=turn,
        user_id="alice",
        session_id="s1",
        request=request,
        response=resp,
        latency_ms=10.0,
        writes_before=writes_before,
        writes_after=writes_after,
        trace=TurnTrace(trace_id=resp.trace_id, spans=tuple(spans)),
    )


def evidence(
    runs: Sequence[TurnRun],
    *,
    case: Case | None = None,
    writes: WritesReport | None = None,
    facts: Facts | None = None,
) -> CaseEvidence:
    return CaseEvidence(
        case=case or make_case(),
        turns=list(runs),
        writes=writes or WritesReport(total=0, by_key={}),
        facts=facts or {},
    )
