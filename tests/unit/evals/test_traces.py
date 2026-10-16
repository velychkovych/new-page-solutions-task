"""Reading a turn's spans, and the vacuity guard: no check runs on an incomplete trace."""

from pathlib import Path

import pytest
from tests.unit.evals.conftest import (
    TRACE_ID,
    llm_span,
    mock_span,
    request_span,
    response,
    tool_call,
    tool_span,
)

from evals.traces import (
    TokenUsage,
    TraceIncompleteError,
    TurnTrace,
    calls_reaching_mcp,
    llm_input_text,
    llm_tool_results,
    missing_parts,
    read_trace,
    wait_for_trace,
)
from helpdesk.telemetry.records import SpanRecord

OTHER_TRACE = "f" * 32


def _trace(*spans: SpanRecord) -> TurnTrace:
    return TurnTrace(trace_id=TRACE_ID, spans=spans)


def _write(path: Path, *spans: SpanRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for span in spans:
            file.write(span.model_dump_json() + "\n")


# --- calls_reaching_mcp ----------------------------------------------------------------------


def test_calls_that_never_ran_or_run_in_the_agent_need_no_tool_span() -> None:
    resp = response(
        tool_calls=[
            tool_call("search_tickets", latency_ms=30),
            tool_call(
                "create_ticket", latency_ms=0, error="ERROR INVALID_ARGUMENT: x. Next step: y."
            ),
            tool_call("remember_fact", latency_ms=3),
            tool_call("get_ticket", latency_ms=12, error="ERROR NOT_FOUND: x. Next step: y."),
        ]
    )
    assert calls_reaching_mcp(resp) == 2


# --- missing_parts (the guard) ---------------------------------------------------------------


def test_a_complete_trace_has_no_missing_parts() -> None:
    resp = response(tool_calls=[tool_call("search_tickets")])
    trace = _trace(
        request_span(), tool_span("search_tickets", statuses=(503, 200)), mock_span(), mock_span()
    )
    assert missing_parts(trace, resp) == []


def test_the_request_span_is_required() -> None:
    assert missing_parts(_trace(), response()) == ["no agent.request span"]


def test_one_tool_span_per_call_that_reached_mcp_is_required() -> None:
    resp = response(tool_calls=[tool_call("search_tickets"), tool_call("get_ticket")])
    trace = _trace(request_span(), tool_span("search_tickets"), mock_span())
    assert missing_parts(trace, resp) == ["1 of 2 mcp.tool spans"]


def test_one_mock_span_per_attempt_with_a_status_is_required() -> None:
    resp = response(tool_calls=[tool_call("search_tickets")])
    # Three attempts got a status, so the mock served three requests in this trace.
    trace = _trace(
        request_span(), tool_span("search_tickets", statuses=(503, 429, 200)), mock_span()
    )
    assert missing_parts(trace, resp) == ["1 of 3 mock spans"]


def test_attempts_without_a_status_may_not_have_reached_the_mock() -> None:
    resp = response(tool_calls=[tool_call("search_tickets")])
    trace = _trace(request_span(), tool_span("search_tickets", statuses=(None, 200)), mock_span())
    assert missing_parts(trace, resp) == []


# --- reading files ---------------------------------------------------------------------------


def test_read_trace_merges_directories_and_keeps_only_the_trace(tmp_path: Path) -> None:
    request, tool = request_span(), tool_span("search_tickets")
    _write(tmp_path / "eval" / "agent.jsonl", request, request_span(trace_id=OTHER_TRACE))
    _write(tmp_path / "mcp.jsonl", tool, tool)  # a duplicate line counts once
    trace = read_trace([tmp_path, tmp_path / "eval"], TRACE_ID)
    assert {span.span_id for span in trace.spans} == {request.span_id, tool.span_id}
    assert len(trace.spans) == 2


async def test_wait_for_trace_returns_once_the_trace_is_complete(tmp_path: Path) -> None:
    resp = response(tool_calls=[tool_call("get_ticket")])
    _write(tmp_path / "agent.jsonl", request_span())
    _write(tmp_path / "mcp.jsonl", tool_span("get_ticket"))
    _write(tmp_path / "mock.jsonl", mock_span("GET /tickets/{ticket_id}"))
    trace = await wait_for_trace([tmp_path], resp, timeout_s=1, poll_s=0.01)
    assert len(trace.tool_spans) == 1


async def test_wait_for_trace_fails_on_an_incomplete_trace(tmp_path: Path) -> None:
    resp = response(tool_calls=[tool_call("create_ticket")])
    # The request span is there, but the write's mcp.tool span never arrives: a check like
    # "no write span" must not get to run on this.
    _write(tmp_path / "agent.jsonl", request_span())
    with pytest.raises(TraceIncompleteError, match=r"trace incomplete: .*0 of 1 mcp.tool spans"):
        await wait_for_trace([tmp_path], resp, timeout_s=0.05, poll_s=0.01)


async def test_wait_for_trace_fails_when_no_span_was_written(tmp_path: Path) -> None:
    with pytest.raises(TraceIncompleteError, match=r"no agent\.request span"):
        await wait_for_trace([tmp_path / "missing"], response(), timeout_s=0.05, poll_s=0.01)


# --- span contents ---------------------------------------------------------------------------


def test_llm_input_text_and_tool_results() -> None:
    span = llm_span(
        [("system", "prompt - team: finance"), ("user", "hi"), ("tool", '{"id":"INC-0003"}')]
    )
    assert "team: finance" in llm_input_text(span)
    assert "hi" in llm_input_text(span)
    assert llm_tool_results(span) == ['{"id":"INC-0003"}']


def test_tokens_models_and_providers_come_from_llm_spans_only() -> None:
    trace = _trace(
        request_span(),
        llm_span(tokens=(100, 20, 80, 10), model="claude-a"),
        llm_span(tokens=(50, 5, 0, 0), model=None),
        tool_span("search_tickets"),
    )
    assert trace.tokens == TokenUsage(input=150, output=25, cache_read=80, cache_write=10)
    assert trace.tokens.total == 175
    assert trace.models == {"claude-a"}
    assert trace.providers == {"anthropic"}


def test_tool_spans_are_sorted_by_start_time() -> None:
    late, early = tool_span("get_ticket", offset_ms=50), tool_span("search_tickets", offset_ms=1)
    assert [s.span_id for s in _trace(late, early).tool_spans] == [early.span_id, late.span_id]
