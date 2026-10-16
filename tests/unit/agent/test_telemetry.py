"""App-start telemetry (CONTRACTS §6): the real lifespan, no injected tracer."""

import json

import httpx
import httpx2
from langchain_anthropic.chat_models import AnthropicOverloadedError

from helpdesk.agent.api import SERVICE_NAME, create_app
from helpdesk.shared import trace_attributes as attrs
from helpdesk.shared.api import ChatResponse, ChatStatus

ZERO_TRACE_ID = "0" * 32


async def _chat_once(settings, model, provider, body) -> ChatResponse:
    """One request on a started app; the lifespan has exited (and flushed) on return."""
    app = create_app(settings=settings, tool_provider=provider, model=model)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://agent.test") as client:
            response = await client.post("/chat", json=body)
    assert response.status_code == 200, response.text
    return ChatResponse.model_validate(response.json())


def _spans(settings) -> list[dict]:
    path = settings.traces_dir / f"{SERVICE_NAME}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _request_span(settings) -> dict:
    (span,) = [s for s in _spans(settings) if s["name"] == attrs.SPAN_REQUEST]
    return span


def _assert_real_trace_id(trace_id: str) -> None:
    assert len(trace_id) == 32
    assert int(trace_id, 16) >= 0
    assert trace_id == trace_id.lower()
    assert trace_id != ZERO_TRACE_ID


async def test_chat_returns_the_trace_id_of_the_recorded_request_span(
    settings, provider, fake_model, make_ai
):
    model = fake_model(
        responses=[make_ai("", ("search_tickets", {"query": "vpn"}, "r1")), make_ai("Found it.")]
    )
    body = {"user_id": "alice", "session_id": "s1", "message": "Is there a VPN ticket?"}

    reply = await _chat_once(settings, model, provider, body)

    _assert_real_trace_id(reply.trace_id)
    span = _request_span(settings)
    assert span["trace_id"] == reply.trace_id
    assert span["service"] == SERVICE_NAME
    assert span["attributes"][attrs.USER_ID] == "alice"
    assert span["attributes"][attrs.SESSION_ID] == "s1"
    assert span["attributes"][attrs.THREAD_ID] == "alice:s1"
    assert span["attributes"][attrs.REQUEST_STATUS] == "ok"
    # The server span from instrument_app is in the same trace, so a traceparent sent by
    # a caller (and on to the MCP server later) shares this id.
    server = [s for s in _spans(settings) if s["parent_span_id"] is None]
    assert [s["trace_id"] for s in server] == [reply.trace_id]
    assert span["parent_span_id"] == server[0]["span_id"]


async def test_a_proposal_records_the_pending_action_id(settings, provider, fake_model, make_ai):
    create = {
        "title": "Laptop battery drains fast",
        "description": "Battery goes from 100% to 20% in an hour.",
        "category": "hardware",
        "priority": "medium",
    }
    model = fake_model(responses=[make_ai("", ("create_ticket", create, "w1"))])
    body = {"user_id": "alice", "session_id": "s1", "message": "My battery dies"}

    reply = await _chat_once(settings, model, provider, body)

    assert reply.status is ChatStatus.AWAITING_APPROVAL
    assert reply.pending_action is not None
    span = _request_span(settings)
    assert span["trace_id"] == reply.trace_id
    assert span["attributes"][attrs.PENDING_ACTION_ID] == reply.pending_action.action_id
    assert span["attributes"][attrs.REQUEST_STATUS] == "awaiting_approval"


async def test_an_error_turn_also_has_a_real_trace_id(settings, provider, fake_model):
    # The anthropic SDK is built on httpx2, so its errors carry httpx2 responses.
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    overloaded = AnthropicOverloadedError(
        message="Overloaded", response=httpx2.Response(529, request=request), body=None
    )
    model = fake_model(responses=[], raises=overloaded)
    body = {"user_id": "alice", "session_id": "s1", "message": "hello"}

    reply = await _chat_once(settings, model, provider, body)

    assert reply.status is ChatStatus.ERROR
    _assert_real_trace_id(reply.trace_id)
    span = _request_span(settings)
    assert span["trace_id"] == reply.trace_id
    assert span["status"] == "error"
    assert span["attributes"][attrs.REQUEST_STATUS] == "error"
