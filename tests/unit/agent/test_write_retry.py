"""Retrying a write whose outcome is unknown reuses its key, so it is never applied twice."""

import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from helpdesk.agent.decision import resume_value
from helpdesk.agent.proposal import idempotency_key_for
from helpdesk.agent.run_config import build_run_config
from helpdesk.agent.tool_provider import ToolResult
from helpdesk.shared.approval import WriteResult
from helpdesk.shared.errors import ErrorCode, format_tool_error
from helpdesk.shared.tools import ToolName

CONFIG = build_run_config("alice", "s1", recursion_limit=25)
COMMENT = {"ticket_id": "INC-0003", "body": "Dropped again at 10:00."}
REWORDED = {"ticket_id": "INC-0003", "body": "VPN dropped again at 10:00."}
OK = ToolResult(text='{"ticket_id": "INC-0003", "comment_id": "c1"}', is_error=False)


def _failure(code: ErrorCode) -> ToolResult:
    return ToolResult(
        text=format_tool_error(code, "add_comment failed", "Tell the user"), is_error=True
    )


async def _turn(graph, message: str) -> None:
    """One user message that ends in a proposal, approved in the next request."""
    result = await graph.ainvoke({"messages": [HumanMessage(message)], "user_id": "alice"}, CONFIG)
    pending = result["pending_action"]
    await graph.ainvoke(Command(resume=resume_value(pending.action_id, approve=True)), CONFIG)


async def _two_writes(make_graph, provider, fake_model, make_ai, *, first: ToolResult, retry_args):
    provider.replies["add_comment"] = [first, OK]
    model = fake_model(
        responses=[
            make_ai("", ("add_comment", COMMENT, "w1")),
            make_ai("I could not confirm the comment was saved."),
            make_ai("", ("add_comment", retry_args, "w2")),
            make_ai("Comment added."),
        ]
    )
    graph = await make_graph(model, provider)
    await _turn(graph, "vpn dropped")
    await _turn(graph, "please try again")
    return [w.idempotency_key for w in provider.writes()]


@pytest.mark.parametrize(
    "code", [ErrorCode.UPSTREAM_UNAVAILABLE, ErrorCode.TIMEOUT, ErrorCode.INTERNAL]
)
async def test_retrying_an_unconfirmed_write_with_the_same_args_reuses_its_key(
    make_graph, provider, fake_model, make_ai, code
):
    keys = await _two_writes(
        make_graph, provider, fake_model, make_ai, first=_failure(code), retry_args=COMMENT
    )

    # The mock replays a write it already saved under this key instead of saving it again.
    assert len(keys) == 2
    assert keys[0] == keys[1]


async def test_a_retry_with_other_args_gets_a_new_key(make_graph, provider, fake_model, make_ai):
    # The mock rejects a reused key with a different body (409), so the key cannot carry over.
    failure = _failure(ErrorCode.UPSTREAM_UNAVAILABLE)
    keys = await _two_writes(
        make_graph, provider, fake_model, make_ai, first=failure, retry_args=REWORDED
    )

    assert len(set(keys)) == 2


async def test_a_write_that_definitely_failed_gets_a_new_key_on_retry(
    make_graph, provider, fake_model, make_ai
):
    failure = _failure(ErrorCode.NOT_FOUND)
    keys = await _two_writes(
        make_graph, provider, fake_model, make_ai, first=failure, retry_args=COMMENT
    )

    assert len(set(keys)) == 2


async def test_the_same_change_after_a_saved_write_is_a_new_write(
    make_graph, provider, fake_model, make_ai
):
    keys = await _two_writes(
        make_graph, provider, fake_model, make_ai, first=OK, retry_args=COMMENT
    )

    assert len(set(keys)) == 2


async def test_a_declined_retry_keeps_the_unconfirmed_key_for_the_next_one(
    make_graph, provider, fake_model, make_ai
):
    provider.replies["add_comment"] = [_failure(ErrorCode.TIMEOUT), OK]
    model = fake_model(
        responses=[
            make_ai("", ("add_comment", COMMENT, "w1")),
            make_ai("I could not confirm the comment was saved."),
            make_ai("", ("add_comment", COMMENT, "w2")),
            make_ai("OK, I won't add it."),
            make_ai("", ("add_comment", COMMENT, "w3")),
            make_ai("Comment added."),
        ]
    )
    graph = await make_graph(model, provider)
    await _turn(graph, "vpn dropped")
    declined = await graph.ainvoke(
        {"messages": [HumanMessage("try again")], "user_id": "alice"}, CONFIG
    )
    await graph.ainvoke(
        Command(resume=resume_value(declined["pending_action"].action_id, approve=False)), CONFIG
    )
    await _turn(graph, "actually, do try again")

    keys = [w.idempotency_key for w in provider.writes()]
    assert len(keys) == 2
    assert keys[0] == keys[1]


def test_the_failed_write_records_its_args_for_the_retry_check(provider):
    unconfirmed = WriteResult(
        action_id="a1",
        tool=ToolName.ADD_COMMENT,
        idempotency_key="k1",
        args=COMMENT,
        ok=False,
        error_code=ErrorCode.TIMEOUT,
    )

    assert idempotency_key_for(ToolName.ADD_COMMENT, dict(COMMENT), unconfirmed) == "k1"
    assert idempotency_key_for(ToolName.CREATE_TICKET, dict(COMMENT), unconfirmed) != "k1"
    assert idempotency_key_for(ToolName.ADD_COMMENT, dict(COMMENT), None) != "k1"


def test_a_write_result_checkpointed_without_args_still_loads():
    legacy = {
        "action_id": "a1",
        "tool": "add_comment",
        "idempotency_key": "k1",
        "ok": False,
        "error_code": "TIMEOUT",
    }

    loaded = WriteResult.model_validate(legacy)

    # No recorded args, so nothing can match it: a retry gets a fresh key.
    assert idempotency_key_for(ToolName.ADD_COMMENT, dict(COMMENT), loaded) != "k1"
