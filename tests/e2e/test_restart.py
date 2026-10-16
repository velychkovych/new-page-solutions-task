"""PLAN Phase 2 item 6: a proposal survives an agent restart and is written exactly once.

The PLAN says `docker compose restart agent`. The e2e tests run against `agent-e2e`, the
fake-model twin of `agent` (same image and entry point), so that is the one restarted.
"""

import pytest
from tests.e2e.conftest import (
    USER_ID,
    Agent,
    Mock,
    assert_exactly_one_write,
    created_ticket,
    restart_e2e_agent,
)

from helpdesk.agent.final_reply import EMPTY_AFTER_WRITE_REPLY
from helpdesk.agent.service import STALE_DECISION_REPLY

pytestmark = pytest.mark.e2e


def test_approve_after_restart_writes_exactly_once(agent: Agent, mock: Mock) -> None:
    pending = agent.propose()

    restart_e2e_agent()

    # The pending action came back from the checkpoint, not from process memory.
    view = agent.client.get(f"/sessions/{USER_ID}/{agent.session_id}")
    assert view.status_code == 200
    restored = view.json()["pending_action"]
    assert restored is not None
    assert restored["action_id"] == pending.action_id

    approved = agent.chat(action_id=pending.action_id)

    assert approved.status.value == "ok", approved
    assert [c.name for c in approved.tool_calls] == ["create_ticket"]
    assert approved.tool_calls[0].error is None
    assert approved.reply == EMPTY_AFTER_WRITE_REPLY
    assert_exactly_one_write(mock)
    created_ticket(mock)

    again = agent.chat(action_id=pending.action_id)

    assert again.reply == STALE_DECISION_REPLY
    assert again.status.value == "ok"
    assert again.tool_calls == []
    assert_exactly_one_write(mock)
