"""PLAN Phase 2 item 5 (and item 2's trace check), against the compose stack.

search -> read -> propose -> approve -> the ticket exists in the mock, written once.
"""

import pytest
from tests.e2e.conftest import (
    PROPOSED_ARGS,
    Agent,
    Mock,
    assert_exactly_one_write,
    created_ticket,
    named,
    tokens,
    wait_for_spans,
)

from helpdesk.agent.final_reply import EMPTY_AFTER_WRITE_REPLY
from helpdesk.shared import trace_attributes as attrs
from helpdesk.shared.api import SseEvent

pytestmark = pytest.mark.e2e


def test_search_read_propose_approve_creates_the_ticket_once(agent: Agent, mock: Mock) -> None:
    pending = agent.propose()
    # Nothing is written before the user approves.
    assert mock.writes()["total"] == 0
    assert mock.search("printer floor 3") == []

    approved = agent.chat(action_id=pending.action_id)

    assert approved.status.value == "ok", approved
    assert approved.pending_action is None
    (write,) = approved.tool_calls
    assert write.name == "create_ticket"
    assert write.error is None, write.error
    assert write.args == PROPOSED_ARGS
    assert approved.reply == EMPTY_AFTER_WRITE_REPLY
    ticket = created_ticket(mock)
    assert ticket["id"] in write.result_summary
    assert ticket["status"] == "open"
    assert_exactly_one_write(mock)


def test_one_request_is_one_trace_across_agent_mcp_and_mock(agent: Agent, mock: Mock) -> None:
    response = agent.chat("The printer on floor 3 keeps jamming on double-sided jobs.")
    trace_id = response.trace_id
    assert trace_id != "0" * 32

    agent_spans = wait_for_spans("agent", trace_id, attrs.SPAN_REQUEST)
    # Both tool calls' spans, so the snapshot holds every MCP span the mock spans need.
    mcp_spans = wait_for_spans("mcp", trace_id, attrs.SPAN_TOOL, count=2)
    mock_spans = wait_for_spans("mock", trace_id, "GET /tickets/{ticket_id}")

    tools = {s["attributes"][attrs.TOOL_NAME] for s in mcp_spans if s["name"] == attrs.SPAN_TOOL}
    assert tools == {"search_tickets", "get_ticket"}
    assert {"GET /tickets", "GET /tickets/{ticket_id}"} <= {s["name"] for s in mock_spans}
    # Each hop's server span is a child of a span of the calling service.
    agent_ids = {s["span_id"] for s in agent_spans}
    mcp_ids = {s["span_id"] for s in mcp_spans}
    carriers = [s for s in mcp_spans if s["name"] == "POST /mcp"]
    assert carriers
    assert any(s["parent_span_id"] in agent_ids for s in carriers)
    assert all(s["parent_span_id"] in mcp_ids for s in mock_spans if s["parent_span_id"])


def test_stream_happy_path(agent: Agent, mock: Mock) -> None:
    events = agent.stream("The printer on floor 3 keeps jamming on double-sided jobs.")

    assert [s.name for s in named(events, SseEvent.TOOL_START)] == ["search_tickets", "get_ticket"]
    assert all(end.error is None for end in named(events, SseEvent.TOOL_END))
    assert events[-1].data.status.value == "awaiting_approval"
    (proposal,) = named(events, SseEvent.APPROVAL_REQUIRED)
    assert proposal.tool == "create_ticket"
    assert proposal.args == PROPOSED_ARGS
    assert mock.writes()["total"] == 0

    approved = agent.stream(action_id=proposal.action_id)

    (start,) = named(approved, SseEvent.TOOL_START)
    (end,) = named(approved, SseEvent.TOOL_END)
    assert start.name == "create_ticket"
    assert end.call_id == start.call_id
    assert end.error is None
    assert approved[-1].data.status.value == "ok"
    assert not named(approved, SseEvent.ERROR)
    assert tokens(approved) == EMPTY_AFTER_WRITE_REPLY
    created_ticket(mock)
    assert_exactly_one_write(mock)


def test_the_agent_serves_the_web_page(agent: Agent) -> None:
    response = agent.client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<meta name="helpdesk-jaeger-ui-url"' in response.text
