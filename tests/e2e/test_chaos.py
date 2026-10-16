"""PLAN Phase 2 item 7: chaos in the mock, after the proposal and before the approval.

- Every mock request fails: the user gets an honest failure reply, no ticket id is made
  up, and nothing is written.
- The write commits but its response is lost: the MCP server's retry (same idempotency
  key) gets the stored result as a replay, so the ticket is written exactly once.
"""

import re

import pytest
from tests.e2e.conftest import (
    CREATED_QUERY,
    Agent,
    Mock,
    assert_exactly_one_write,
    created_ticket,
    named,
    tokens,
    wait_for_spans,
)

from helpdesk.agent.final_reply import EMPTY_AFTER_FAILED_WRITE_REPLY, EMPTY_AFTER_WRITE_REPLY
from helpdesk.shared import trace_attributes as attrs
from helpdesk.shared.api import SseEvent

pytestmark = pytest.mark.e2e

# Seeded, so the 500/503 choices repeat run to run.
ALL_FAIL = {"enabled": True, "error_rate": 1.0, "seed": 7}
FAIL_AFTER_COMMIT = {"enabled": True, "fail_after_commit_rate": 1.0, "seed": 7}
WRITE_FAILED = re.compile(r"^ERROR (UPSTREAM_UNAVAILABLE|TIMEOUT): ")
TICKET_ID = re.compile(r"INC-\d{4}")


def _tool_span(trace_id: str) -> dict[str, object]:
    spans = wait_for_spans("mcp", trace_id, attrs.SPAN_TOOL)
    (tool,) = [s for s in spans if s["name"] == attrs.SPAN_TOOL]
    attributes: dict[str, object] = tool["attributes"]
    assert attributes[attrs.TOOL_NAME] == "create_ticket"
    return attributes


def _assert_nothing_written(mock: Mock) -> None:
    # Chaos would fail these reads too.
    mock.chaos_off()
    assert mock.writes() == {"total": 0, "by_key": {}}
    assert mock.search(CREATED_QUERY) == []


def test_all_requests_failing_gives_an_honest_failure_and_no_write(
    agent: Agent, mock: Mock
) -> None:
    pending = agent.propose()
    mock.chaos(**ALL_FAIL)

    approved = agent.chat(action_id=pending.action_id)

    (write,) = approved.tool_calls
    assert write.name == "create_ticket"
    assert write.error is not None
    assert WRITE_FAILED.match(write.error), write.error
    # The turn itself ran to its end; the reply is the honest one about the failed write.
    assert approved.status.value == "ok", approved
    assert approved.reply == EMPTY_AFTER_FAILED_WRITE_REPLY
    assert not TICKET_ID.search(approved.reply)
    assert approved.pending_action is None
    assert _tool_span(approved.trace_id)[attrs.TOOL_ATTEMPTS] == 3
    _assert_nothing_written(mock)


def test_stream_shows_the_retries_of_a_failing_write(agent: Agent, mock: Mock) -> None:
    pending = agent.propose()
    mock.chaos(**ALL_FAIL)

    events = agent.stream(action_id=pending.action_id)

    statuses = [s.message for s in named(events, SseEvent.STATUS)]
    assert "create_ticket: Retrying (attempt 2 of 3)" in statuses, statuses
    assert "create_ticket: Retrying (attempt 3 of 3)" in statuses, statuses
    (end,) = named(events, SseEvent.TOOL_END)
    assert end.error is not None
    assert WRITE_FAILED.match(end.error), end.error
    assert events[-1].data.status.value == "ok"
    assert tokens(events) == EMPTY_AFTER_FAILED_WRITE_REPLY
    _assert_nothing_written(mock)


def test_a_lost_response_after_commit_still_writes_exactly_once(agent: Agent, mock: Mock) -> None:
    pending = agent.propose()
    mock.chaos(**FAIL_AFTER_COMMIT)

    approved = agent.chat(action_id=pending.action_id)

    (write,) = approved.tool_calls
    assert write.name == "create_ticket"
    assert write.error is None, write.error
    assert approved.status.value == "ok", approved
    assert approved.reply == EMPTY_AFTER_WRITE_REPLY
    # The first attempt committed and lost its response; the retry was a replay.
    assert _tool_span(approved.trace_id)[attrs.TOOL_ATTEMPTS] == 2
    assert_exactly_one_write(mock)
    mock.chaos_off()
    ticket = created_ticket(mock)
    assert ticket["id"] in write.result_summary
    assert_exactly_one_write(mock)
