"""One fake request across agent and MCP server, through the real exporters and show_trace."""

import io
import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, format_trace_id
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from helpdesk.shared import trace_attributes as ta
from helpdesk.shared.settings import Settings
from helpdesk.telemetry import setup_telemetry
from helpdesk.telemetry.setup import build_telemetry
from helpdesk.telemetry.show_trace import ExitCode, main

CONTRACT_FIELDS = {
    "trace_id",
    "span_id",
    "parent_span_id",
    "name",
    "service",
    "start_time",
    "end_time",
    "duration_ms",
    "status",
    "attributes",
    "events",
}
REQUEST_KEYS = {
    ta.USER_ID,
    ta.SESSION_ID,
    ta.THREAD_ID,
    ta.REQUEST_STATUS,
    ta.PENDING_ACTION_ID,
}
TOOL_KEYS = {ta.TOOL_NAME, ta.TOOL_ARGS, ta.TOOL_RESULT, ta.TOOL_ERROR_CODE, ta.TOOL_ATTEMPTS}
RETRY_KEYS = {ta.RETRY_ATTEMPT, ta.RETRY_HTTP_STATUS, ta.RETRY_WAIT_MS}
APPROVAL_KEYS = {ta.APPROVAL_ACTION_ID, ta.APPROVAL_DECISION, ta.APPROVAL_IDEMPOTENCY_KEY}


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _fake_model() -> GenericFakeChatModel:
    # usage_metadata in the shape langchain-anthropic produces without TTL breakdown.
    reply = AIMessage(
        content="I found INC-0003.",
        usage_metadata={
            "input_tokens": 1200,
            "output_tokens": 80,
            "total_tokens": 1280,
            "input_token_details": {"cache_read": 1000, "cache_creation": 150},
        },
        response_metadata={"finish_reason": "end_turn"},
    )
    return GenericFakeChatModel(messages=iter([reply]))


async def _run_fake_request(settings: Settings) -> str:
    """Emit one request's spans for agent and MCP server, flush them, return the trace id."""
    agent = setup_telemetry("agent", settings)
    # A second provider stands in for the MCP server process.
    mcp = build_telemetry("mcp", settings)
    agent_tracer = trace.get_tracer("test.agent")
    mcp_tracer = mcp.provider.get_tracer("test.mcp")
    propagator = TraceContextTextMapPropagator()

    with agent_tracer.start_as_current_span(ta.SPAN_REQUEST) as request:
        request.set_attributes(
            {
                ta.USER_ID: "alice",
                ta.SESSION_ID: "s1",
                ta.THREAD_ID: "alice:s1",
                ta.REQUEST_STATUS: "ok",
                ta.PENDING_ACTION_ID: "act-1",
            }
        )
        trace_id = format_trace_id(request.get_span_context().trace_id)

        await _fake_model().ainvoke("VPN keeps dropping")

        # Across processes the context travels as a traceparent header.
        carrier: dict[str, str] = {}
        propagator.inject(carrier)
        with mcp_tracer.start_as_current_span(
            ta.SPAN_TOOL, context=propagator.extract(carrier)
        ) as tool:
            tool.set_attributes(
                {
                    ta.TOOL_NAME: "search_tickets",
                    ta.TOOL_ARGS: json.dumps({"query": "vpn"}),
                    ta.TOOL_RESULT: "ERROR UPSTREAM_UNAVAILABLE: ...",
                    ta.TOOL_ERROR_CODE: "UPSTREAM_UNAVAILABLE",
                    ta.TOOL_ATTEMPTS: 3,
                }
            )
            tool.add_event(
                ta.EVENT_RETRY,
                {ta.RETRY_ATTEMPT: 1, ta.RETRY_HTTP_STATUS: 503, ta.RETRY_WAIT_MS: 250},
            )
            tool.add_event(
                ta.EVENT_RETRY,
                {ta.RETRY_ATTEMPT: 2, ta.RETRY_HTTP_STATUS: 429, ta.RETRY_WAIT_MS: 1000},
            )
            tool.set_status(Status(StatusCode.ERROR, "retries exhausted"))

        with agent_tracer.start_as_current_span(ta.SPAN_APPROVAL) as approval:
            approval.set_attributes(
                {
                    ta.APPROVAL_ACTION_ID: "act-1",
                    ta.APPROVAL_DECISION: "approve",
                    ta.APPROVAL_IDEMPOTENCY_KEY: "idem-1",
                }
            )
    assert agent.flush()
    mcp.provider.shutdown()  # flushes the MCP stand-in's batch
    return trace_id


async def test_fake_run_span_tree_is_complete(make_settings: Any) -> None:
    settings: Settings = make_settings()
    trace_id = await _run_fake_request(settings)

    agent_records = _records(settings.traces_dir / "agent.jsonl")
    mcp_records = _records(settings.traces_dir / "mcp.jsonl")
    records = agent_records + mcp_records
    for record in records:
        assert set(record) == CONTRACT_FIELDS
        assert record["trace_id"] == trace_id
    assert {r["service"] for r in agent_records} == {"agent"}
    assert {r["service"] for r in mcp_records} == {"mcp"}

    (root,) = [r for r in records if r["name"] == ta.SPAN_REQUEST]
    (llm,) = [r for r in records if r["attributes"].get("openinference.span.kind") == "LLM"]
    (tool_rec,) = [r for r in records if r["name"] == ta.SPAN_TOOL]
    (approval_rec,) = [r for r in records if r["name"] == ta.SPAN_APPROVAL]
    assert len(records) == 4

    assert root["parent_span_id"] is None
    for child in (llm, tool_rec, approval_rec):
        assert child["parent_span_id"] == root["span_id"]

    assert set(root["attributes"]) >= REQUEST_KEYS
    assert set(tool_rec["attributes"]) >= TOOL_KEYS
    assert set(approval_rec["attributes"]) >= APPROVAL_KEYS
    assert tool_rec["status"] == "error"
    retries = [e for e in tool_rec["events"] if e["name"] == ta.EVENT_RETRY]
    assert len(retries) == 2
    assert all(set(e["attributes"]) == RETRY_KEYS for e in retries)

    # Keys the installed OpenInference LangChain instrumentation emits.
    assert llm["attributes"]["llm.token_count.prompt"] == 1200
    assert llm["attributes"]["llm.token_count.completion"] == 80
    assert llm["attributes"]["llm.token_count.prompt_details.cache_read"] == 1000
    assert llm["attributes"]["llm.token_count.prompt_details.cache_write"] == 150
    assert llm["attributes"]["llm.finish_reason"] == "end_turn"
    assert "llm.provider" in llm["attributes"]


async def test_show_trace_renders_the_fake_run(make_settings: Any) -> None:
    settings: Settings = make_settings()
    trace_id = await _run_fake_request(settings)

    out, err = io.StringIO(), io.StringIO()
    assert main([trace_id], settings=settings, out=out, err=err) == ExitCode.OK
    lines = out.getvalue().splitlines()
    assert lines[0] == f"trace {trace_id}: 4 spans, services: agent, mcp"
    assert lines[1].startswith(f"{ta.SPAN_REQUEST} [agent] ")
    assert "request_status=ok" in lines[1]
    assert "pending_action=act-1" in lines[1]
    body = "\n".join(lines[2:])
    assert "  GenericFakeChatModel [agent] " in body
    assert "tokens in=1200 out=80 cache_read=1000 cache_write=150" in body
    assert "  mcp.tool [mcp] " in body
    assert "tool=search_tickets  attempts=3  error_code=UPSTREAM_UNAVAILABLE" in body
    assert "    · retry.attempt attempt=1 http.status_code=503 wait_ms=250" in body
    assert "    · retry.attempt attempt=2 http.status_code=429 wait_ms=1000" in body
    assert "  agent.approval [agent] " in body
    assert "decision=approve" in body
    assert err.getvalue() == ""


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Open question: langchain-anthropic 1.7.6 moves cache writes to "
        "input_token_details.ephemeral_5m/1h_input_tokens and sets cache_creation=0; "
        "OpenInference 0.1.79 reads only cache_creation from usage_metadata, so on the "
        "streaming path cache_write is not recorded"
    ),
)
async def test_cache_write_recorded_with_ttl_breakdown(make_settings: Any) -> None:
    settings: Settings = make_settings()
    agent = setup_telemetry("agent", settings)
    reply = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": 1200,
            "output_tokens": 80,
            "total_tokens": 1280,
            "input_token_details": {
                "cache_read": 1000,
                "cache_creation": 0,
                "ephemeral_5m_input_tokens": 150,
            },
        },
    )
    with trace.get_tracer("test").start_as_current_span(ta.SPAN_REQUEST):
        await GenericFakeChatModel(messages=iter([reply])).ainvoke("hi")
    assert agent.flush()

    (llm,) = [
        r
        for r in _records(settings.traces_dir / "agent.jsonl")
        if r["attributes"].get("openinference.span.kind") == "LLM"
    ]
    assert llm["attributes"].get("llm.token_count.prompt_details.cache_write") == 150
