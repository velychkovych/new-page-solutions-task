"""Whatever the model or a tool sends back, the turn stays well-formed (CONTRACTS §3.1, §5.3).

Every case is one scripted turn through the API. Each checks:
- the ToolMessage the graph wrote (status and ERROR code);
- the `tool_calls` record in the response;
- that the next model input answers every tool call exactly once, and that
  langchain-anthropic's own request formatting pairs every tool_use with one tool_result;
- a non-empty reply;
- that nothing was written.
"""

import asyncio
import json
from collections import Counter
from typing import Any

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import SecretStr

from helpdesk.agent import final_reply, turn_errors
from helpdesk.agent.history import model_messages
from helpdesk.agent.model_calls import UNNAMED_TOOL
from helpdesk.agent.run_config import build_run_config
from helpdesk.agent.tool_errors import UNCONFIRMED_WRITE_NEXT_STEP, rejected_batch_error
from helpdesk.agent.tool_messages import USER_DECLINED
from helpdesk.agent.tool_provider import ToolResult
from helpdesk.shared.api import ChatResponse, ChatStatus, ToolCallRecord
from helpdesk.shared.errors import ErrorCode, ToolError, parse_tool_error

CONFIG = build_run_config("alice", "s1", recursion_limit=25)
CREATE_ARGS = {
    "title": "Laptop battery drains fast",
    "description": "Battery goes from 100% to 20% in an hour.",
    "category": "hardware",
    "priority": "medium",
}
COMMENT_ARGS = {"ticket_id": "INC-0003", "body": "Dropped again at 10:00."}
TOOL_NAMES = ["search_tickets", "get_ticket", "create_ticket", "add_comment", "remember_fact"]
# Used only to format requests, never to send one.
ANTHROPIC = ChatAnthropic(model="claude-test", api_key=SecretStr("not-a-real-key"))


# --- helpers ---------------------------------------------------------------------------


def _tool_use_ids(message: AIMessage) -> list[str]:
    """Ids of every call the message asks for, read straight off the message."""
    ids = [call["id"] for call in message.tool_calls]
    ids += [call["id"] for call in message.invalid_tool_calls]
    assert all(ids), f"a tool call without an id reached the model input: {message}"
    return [i for i in ids if i]


def assert_every_call_answered_once(messages: list[BaseMessage]) -> None:
    asked = Counter(i for m in messages if isinstance(m, AIMessage) for i in _tool_use_ids(m))
    answered = Counter(m.tool_call_id for m in messages if isinstance(m, ToolMessage))
    assert asked == answered
    assert set(asked.values()) <= {1}


def assert_anthropic_request_is_valid(messages: list[BaseMessage]) -> None:
    """Every tool_use the formatted request sends gets exactly one tool_result next."""
    payload = ANTHROPIC._get_request_payload(messages)
    formatted = payload["messages"]
    uses: Counter[str] = Counter()
    results: Counter[str] = Counter()
    for index, message in enumerate(formatted):
        blocks = message["content"] if isinstance(message["content"], list) else []
        for block in blocks:
            if block["type"] == "text":
                assert block["text"].strip(), "the API rejects whitespace-only text"
        if isinstance(message["content"], str):
            assert message["content"].strip()
        turn_uses = [b["id"] for b in blocks if b["type"] == "tool_use"]
        uses.update(turn_uses)
        results.update(b["tool_use_id"] for b in blocks if b["type"] == "tool_result")
        if turn_uses:
            following = formatted[index + 1]["content"]
            answers = [b["tool_use_id"] for b in following if b["type"] == "tool_result"]
            assert sorted(answers) == sorted(turn_uses)
    assert uses == results
    assert set(uses.values()) <= {1}


def assert_well_formed(messages: list[BaseMessage]) -> None:
    assert_every_call_answered_once(messages)
    assert_anthropic_request_is_valid(messages)


def _tool_messages(messages: list[BaseMessage]) -> list[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage)]


def _error(message: ToolMessage) -> ToolError:
    assert message.status == "error"
    error = parse_tool_error(str(message.content))
    assert error is not None, message.content
    return error


def _rejected_record(record: ToolCallRecord, name: str, code: ErrorCode) -> ToolError:
    """A call that never ran: listed with its error and 0 ms."""
    assert record.name == name
    assert record.latency_ms == 0
    assert record.error is not None
    error = parse_tool_error(record.error)
    assert error is not None
    assert error.code is code
    return error


async def _chat(client, **body) -> ChatResponse:
    response = await client.post("/chat", json={"user_id": "alice", "session_id": "s1", **body})
    assert response.status_code == 200, response.text
    reply = ChatResponse.model_validate(response.json())
    assert reply.reply.strip()
    return reply


async def _turn(make_client, provider, model, text="My VPN keeps dropping", **overrides):
    client = await make_client(model, provider, **overrides)
    return client, await _chat(client, message=text)


def _reply(*blocks: dict[str, Any], **fields: Any) -> AIMessage:
    """An AI message as langchain-anthropic builds it: content blocks plus parsed calls."""
    return AIMessage(content=list(blocks), **fields)


def _invalid(name: str | None, raw_args: str, call_id: str | None) -> dict[str, Any]:
    return {
        "type": "invalid_tool_call",
        "name": name,
        "args": raw_args,
        "id": call_id,
        "error": "not valid JSON",
    }


# --- bad model responses ---------------------------------------------------------------


async def test_unparseable_arguments_get_invalid_argument_and_nothing_runs(
    make_client, provider, fake_model, make_ai
):
    bad = _reply(
        {"type": "text", "text": "Searching."},
        {
            "type": "tool_use",
            "id": "bad1",
            "name": "search_tickets",
            "input": {},
            "partial_json": '{"query": "vp',
        },
        invalid_tool_calls=[_invalid("search_tickets", '{"query": "vp', "bad1")],
    )
    model = fake_model(responses=[bad, make_ai("Sorry, let me try that again later.")])

    _, reply = await _turn(make_client, provider, model)

    (message,) = _tool_messages(model.calls[1])
    assert message.tool_call_id == "bad1"
    error = _error(message)
    assert error.code is ErrorCode.INVALID_ARGUMENT
    assert "search_tickets" in error.what
    assert "JSON" in error.what
    (record,) = reply.tool_calls
    assert _rejected_record(record, "search_tickets", ErrorCode.INVALID_ARGUMENT) == error
    assert record.args == {}
    assert_well_formed(model.calls[1])
    assert reply.reply == "Sorry, let me try that again later."
    assert reply.status is ChatStatus.OK
    assert provider.calls == []


async def test_valid_calls_next_to_an_unparseable_one_run_in_model_order(
    make_client, provider, fake_model, make_ai
):
    mixed = _reply(
        {"type": "tool_use", "id": "r1", "name": "search_tickets", "input": {"query": "vpn"}},
        {"type": "tool_use", "id": "bad", "name": "get_ticket", "input": {}, "partial_json": "{"},
        {"type": "tool_use", "id": "r2", "name": "get_ticket", "input": {"ticket_id": "INC-0003"}},
        tool_calls=[
            {"name": "search_tickets", "args": {"query": "vpn"}, "id": "r1"},
            {"name": "get_ticket", "args": {"ticket_id": "INC-0003"}, "id": "r2"},
        ],
        invalid_tool_calls=[_invalid("get_ticket", "{", "bad")],
    )
    model = fake_model(responses=[mixed, make_ai("INC-0003 covers it.")])

    _, reply = await _turn(make_client, provider, model)

    assert [m.tool_call_id for m in _tool_messages(model.calls[1])] == ["r1", "bad", "r2"]
    assert [r.name for r in reply.tool_calls] == ["search_tickets", "get_ticket", "get_ticket"]
    assert [r.error is None for r in reply.tool_calls] == [True, False, True]
    _rejected_record(reply.tool_calls[1], "get_ticket", ErrorCode.INVALID_ARGUMENT)
    assert [c.name for c in provider.calls] == ["search_tickets", "get_ticket"]
    assert_well_formed(model.calls[1])
    assert provider.writes() == []


async def test_unknown_tool_lists_the_available_tools(make_client, provider, fake_model, make_ai):
    model = fake_model(
        responses=[make_ai("", ("close_all_tickets", {}, "u1")), make_ai("I can't do that.")]
    )

    _, reply = await _turn(make_client, provider, model)

    (message,) = _tool_messages(model.calls[1])
    error = _error(message)
    assert error.code is ErrorCode.INVALID_ARGUMENT
    assert "close_all_tickets" in error.what
    assert all(name in error.next_step for name in TOOL_NAMES)
    (record,) = reply.tool_calls
    _rejected_record(record, "close_all_tickets", ErrorCode.INVALID_ARGUMENT)
    assert_well_formed(model.calls[1])
    assert reply.reply == "I can't do that."
    assert provider.calls == []


@pytest.mark.parametrize(
    ("calls", "combined"),
    [
        pytest.param(
            [("create_ticket", CREATE_ARGS, "w1"), ("add_comment", COMMENT_ARGS, "w2")],
            "2 writes (create_ticket, add_comment)",
            id="two-writes",
        ),
        pytest.param(
            [("search_tickets", {"query": "vpn"}, "r1"), ("create_ticket", CREATE_ARGS, "w1")],
            "the write create_ticket with search_tickets",
            id="write-and-read",
        ),
    ],
)
async def test_a_rejected_batch_is_listed_call_by_call(
    *, make_client, provider, fake_model, make_ai, calls, combined
):
    model = fake_model(responses=[make_ai("", *calls), make_ai("One change at a time, then.")])

    _, reply = await _turn(make_client, provider, model)

    messages = _tool_messages(model.calls[1])
    assert [m.tool_call_id for m in messages] == [c[2] for c in calls]
    for message in messages:
        error = _error(message)
        assert error.code is ErrorCode.INVALID_ARGUMENT
        assert error.what == "propose one change at a time, after your reads"
        assert combined in error.next_step
    assert [r.name for r in reply.tool_calls] == [c[0] for c in calls]
    for record, call in zip(reply.tool_calls, calls, strict=True):
        _rejected_record(record, call[0], ErrorCode.INVALID_ARGUMENT)
        assert record.args == call[1]
    assert_well_formed(model.calls[1])
    assert reply.status is ChatStatus.OK
    assert reply.pending_action is None
    assert provider.calls == []


async def test_a_write_next_to_an_unparseable_call_is_rejected_too(
    make_client, provider, fake_model, make_ai
):
    bad = AIMessage(
        content="",
        tool_calls=[{"name": "create_ticket", "args": CREATE_ARGS, "id": "w1"}],
        invalid_tool_calls=[_invalid("search_tickets", '{"q', "bad")],
    )
    model = fake_model(responses=[bad, make_ai("Let me search first.")])

    _, reply = await _turn(make_client, provider, model)

    write, unparseable = _tool_messages(model.calls[1])
    assert _error(write).what == "propose one change at a time, after your reads"
    assert _error(unparseable).code is ErrorCode.INVALID_ARGUMENT
    assert "JSON" in _error(unparseable).what
    assert [r.name for r in reply.tool_calls] == ["create_ticket", "search_tickets"]
    assert reply.pending_action is None
    assert_well_formed(model.calls[1])
    assert provider.calls == []


@pytest.mark.parametrize(
    ("args", "named"),
    [
        pytest.param(
            {**COMMENT_ARGS, "ticket_id": "INC-7"}, ["`ticket_id`", "INC-[0-9]{4}"], id="pattern"
        ),
        pytest.param({"ticket_id": "INC-0003"}, ["`body` is required"], id="missing"),
        pytest.param(
            {**COMMENT_ARGS, "author_id": "bob"},
            ["`author_id` is not an argument of add_comment", "fills it in"],
            id="injected",
        ),
    ],
)
async def test_invalid_write_args_name_the_argument_and_why(
    *, make_client, provider, fake_model, make_ai, args, named
):
    model = fake_model(
        responses=[make_ai("", ("add_comment", args, "w1")), make_ai("Which ticket did you mean?")]
    )

    _, reply = await _turn(make_client, provider, model)

    (message,) = _tool_messages(model.calls[1])
    error = _error(message)
    assert error.code is ErrorCode.INVALID_ARGUMENT
    for text in named:
        assert text in error.what
    assert "ticket_id, body" in error.next_step
    (record,) = reply.tool_calls
    _rejected_record(record, "add_comment", ErrorCode.INVALID_ARGUMENT)
    assert "author_id" not in record.args
    assert_well_formed(model.calls[1])
    assert reply.status is ChatStatus.OK
    assert reply.pending_action is None
    assert provider.writes() == []


async def test_invalid_enum_value_lists_the_allowed_values(
    make_client, provider, fake_model, make_ai
):
    args = {**CREATE_ARGS, "priority": "urgent"}
    model = fake_model(
        responses=[make_ai("", ("create_ticket", args, "w1")), make_ai("Which priority?")]
    )

    _, reply = await _turn(make_client, provider, model)

    error = _error(_tool_messages(model.calls[1])[0])
    assert "`priority`" in error.what
    for value in ("low", "medium", "high", "critical"):
        assert value in error.what
    assert reply.pending_action is None
    assert_well_formed(model.calls[1])
    assert provider.writes() == []


@pytest.mark.parametrize(
    ("args", "named"),
    [
        pytest.param({"key": "salary", "value": "a lot"}, "`key`", id="bad-key"),
        pytest.param({"key": "team", "value": ""}, "`value`", id="empty-value"),
        pytest.param({"key": "team", "value": "x", "user_id": "bob"}, "`user_id`", id="extra"),
    ],
)
async def test_invalid_remember_fact_names_the_allowed_keys(
    *, make_client, provider, fake_model, make_ai, args, named
):
    model = fake_model(
        responses=[make_ai("", ("remember_fact", args, "m1")), make_ai("I can't save that.")]
    )

    _, reply = await _turn(make_client, provider, model)

    error = _error(_tool_messages(model.calls[1])[0])
    assert error.code is ErrorCode.INVALID_ARGUMENT
    assert named in error.what
    for key in ("team", "office_location", "device", "preferred_contact"):
        assert key in error.next_step
    (record,) = reply.tool_calls
    _rejected_record(record, "remember_fact", ErrorCode.INVALID_ARGUMENT)
    assert_well_formed(model.calls[1])
    assert reply.reply == "I can't save that."
    assert provider.calls == []


async def test_a_call_without_an_id_is_answered_and_never_run(
    make_client, provider, fake_model, make_ai
):
    no_id = AIMessage(
        content="", tool_calls=[{"name": "search_tickets", "args": {"query": "vpn"}, "id": None}]
    )
    model = fake_model(responses=[no_id, make_ai("Let me try again.")])

    _, reply = await _turn(make_client, provider, model)

    (message,) = _tool_messages(model.calls[1])
    assert message.tool_call_id
    error = _error(message)
    assert error.code is ErrorCode.INVALID_ARGUMENT
    assert "without a tool call id" in error.what
    (record,) = reply.tool_calls
    _rejected_record(record, "search_tickets", ErrorCode.INVALID_ARGUMENT)
    assert record.args == {"query": "vpn"}
    assert_well_formed(model.calls[1])
    assert provider.calls == []


async def test_an_invalid_call_without_id_or_name_still_gets_one_answer(
    make_client, provider, fake_model, make_ai
):
    nameless = AIMessage(content="", invalid_tool_calls=[_invalid(None, "{", None)])
    model = fake_model(responses=[nameless, make_ai("Sorry, something went wrong.")])

    _, reply = await _turn(make_client, provider, model)

    (message,) = _tool_messages(model.calls[1])
    assert message.name == UNNAMED_TOOL
    assert _error(message).code is ErrorCode.INVALID_ARGUMENT
    assert [r.name for r in reply.tool_calls] == [UNNAMED_TOOL]
    assert_well_formed(model.calls[1])
    assert provider.calls == []


@pytest.mark.parametrize("content", ["", "   \n", []], ids=["empty", "whitespace", "no-blocks"])
async def test_an_empty_final_answer_gets_the_fallback_reply(
    make_client, provider, fake_model, make_ai, content
):
    model = fake_model(responses=[AIMessage(content=content), make_ai("Hello again.")])

    client, reply = await _turn(make_client, provider, model)

    assert reply.reply == final_reply.EMPTY_REPLY
    assert reply.status is ChatStatus.OK
    assert reply.tool_calls == []
    # The blank answer is left out of the next request, which stays valid.
    later = await _chat(client, message="hello?")
    assert later.reply == "Hello again."
    assert not any(
        isinstance(m, AIMessage) and not str(m.text).strip() and not m.tool_calls
        for m in model.calls[1]
    )
    assert_well_formed(model.calls[1])
    assert provider.calls == []


async def test_an_empty_answer_after_a_saved_write_says_it_was_saved(
    make_client, provider, fake_model, make_ai
):
    model = fake_model(responses=[make_ai("", ("create_ticket", CREATE_ARGS, "w1")), AIMessage("")])
    client, proposal = await _turn(make_client, provider, model)

    done = await _chat(client, message="yes")

    assert done.reply == final_reply.EMPTY_AFTER_WRITE_REPLY
    assert [r.name for r in done.tool_calls] == ["create_ticket"]
    assert len(provider.writes()) == 1
    assert proposal.pending_action is not None


async def test_an_empty_answer_after_a_failed_write_does_not_claim_it_was_saved(
    make_client, provider, fake_model, make_ai
):
    conflict = "ERROR CONFLICT: Key reused with a different body. Next step: Do not retry."
    provider.replies["create_ticket"] = [ToolResult(text=conflict, is_error=True)]
    model = fake_model(responses=[make_ai("", ("create_ticket", CREATE_ARGS, "w1")), AIMessage("")])
    client, _ = await _turn(make_client, provider, model)

    done = await _chat(client, message="yes")

    assert done.reply == final_reply.EMPTY_AFTER_FAILED_WRITE_REPLY
    assert "could not confirm" in done.reply


async def test_a_truncated_text_answer_says_it_was_cut_off(
    make_client, provider, fake_model, make_ai
):
    cut = AIMessage(
        "Here is what to try: first, restart", response_metadata={"stop_reason": "max_tokens"}
    )
    model = fake_model(responses=[cut])

    _, reply = await _turn(make_client, provider, model)

    assert reply.reply.startswith("Here is what to try: first, restart")
    assert reply.reply.endswith(final_reply.TRUNCATED_NOTE)
    assert reply.tool_calls == []
    assert provider.calls == []


async def test_a_truncated_reply_with_a_cut_off_call_runs_none_of_its_calls(
    make_client, provider, fake_model, make_ai
):
    cut = _reply(
        {"type": "tool_use", "id": "r1", "name": "search_tickets", "input": {"query": "vpn"}},
        {
            "type": "tool_use",
            "id": "r2",
            "name": "get_ticket",
            "input": {},
            "partial_json": '{"ticket_id": "INC-00',
        },
        tool_calls=[{"name": "search_tickets", "args": {"query": "vpn"}, "id": "r1"}],
        invalid_tool_calls=[_invalid("get_ticket", '{"ticket_id": "INC-00', "r2")],
        response_metadata={"stop_reason": "max_tokens"},
    )
    model = fake_model(responses=[cut, make_ai("Let me search again.")])

    _, reply = await _turn(make_client, provider, model)

    messages = _tool_messages(model.calls[1])
    assert [m.tool_call_id for m in messages] == ["r1", "r2"]
    for message in messages:
        error = _error(message)
        assert error.code is ErrorCode.INVALID_ARGUMENT
        assert "cut off" in error.what
    assert [r.name for r in reply.tool_calls] == ["search_tickets", "get_ticket"]
    assert all(r.latency_ms == 0 and r.error for r in reply.tool_calls)
    assert_well_formed(model.calls[1])
    assert reply.reply == "Let me search again."
    assert provider.calls == []


@pytest.mark.parametrize(
    "metadata",
    [{"stop_reason": "max_tokens"}, {"finish_reason": "length"}],
    ids=["anthropic", "openai"],
)
async def test_a_truncated_write_never_becomes_a_pending_action(
    make_client, provider, fake_model, make_ai, metadata
):
    cut = AIMessage(
        content="",
        tool_calls=[{"name": "create_ticket", "args": CREATE_ARGS, "id": "w1"}],
        response_metadata=metadata,
    )
    model = fake_model(responses=[cut, make_ai("I'll propose it again.")])

    _, reply = await _turn(make_client, provider, model)

    (message,) = _tool_messages(model.calls[1])
    assert "cut off" in _error(message).what
    assert reply.status is ChatStatus.OK
    assert reply.pending_action is None
    (record,) = reply.tool_calls
    _rejected_record(record, "create_ticket", ErrorCode.INVALID_ARGUMENT)
    assert_well_formed(model.calls[1])
    assert provider.writes() == []


# --- bad tool responses ----------------------------------------------------------------


async def test_an_mcp_tool_error_is_kept_exactly(make_client, provider, fake_model, make_ai):
    not_found = "ERROR NOT_FOUND: Ticket INC-0099 does not exist. Next step: Check the id."
    provider.replies["get_ticket"] = [ToolResult(text=not_found, is_error=True)]
    model = fake_model(
        responses=[make_ai("", ("get_ticket", {"ticket_id": "INC-0099"}, "r1")), make_ai("No.")]
    )

    _, reply = await _turn(make_client, provider, model)

    (message,) = _tool_messages(model.calls[1])
    assert message.content == not_found
    assert _error(message).code is ErrorCode.NOT_FOUND
    (record,) = reply.tool_calls
    assert record.error == not_found
    assert record.latency_ms >= 0
    assert_well_formed(model.calls[1])
    assert len(provider.calls) == 1


async def test_a_transport_failure_is_upstream_unavailable_once(
    make_client, provider, fake_model, make_ai, transport_error
):
    provider.replies["search_tickets"] = [transport_error("connection refused")]
    model = fake_model(
        responses=[make_ai("", ("search_tickets", {"query": "vpn"}, "r1")), make_ai("It's down.")]
    )

    _, reply = await _turn(make_client, provider, model)

    error = _error(_tool_messages(model.calls[1])[0])
    assert error.code is ErrorCode.UPSTREAM_UNAVAILABLE
    assert "search_tickets" in error.what
    assert "connection refused" not in error.what
    assert parse_tool_error(reply.tool_calls[0].error or "") == error
    assert_well_formed(model.calls[1])
    assert len(provider.calls) == 1


@pytest.mark.parametrize(
    ("result", "cause"),
    [
        pytest.param(ToolResult(text="", is_error=False), "empty result", id="empty"),
        pytest.param(ToolResult(text="  ", is_error=True), "empty result", id="empty-error"),
        pytest.param(
            ToolResult(text="Traceback (most recent call last): token=abc", is_error=True),
            "not in the ERROR format",
            id="error-not-in-contract-format",
        ),
        pytest.param({"text": "not a ToolResult"}, "no tool result", id="not-a-tool-result"),
    ],
)
async def test_a_result_that_breaks_the_contract_is_internal(
    *, make_client, provider, fake_model, make_ai, result, cause
):
    async def reply_with(_args):
        return result

    provider.replies["search_tickets"] = [reply_with]
    model = fake_model(
        responses=[make_ai("", ("search_tickets", {"query": "vpn"}, "r1")), make_ai("It failed.")]
    )

    _, reply = await _turn(make_client, provider, model)

    (message,) = _tool_messages(model.calls[1])
    error = _error(message)
    assert error.code is ErrorCode.INTERNAL
    assert "search_tickets" in error.what
    assert cause in error.what
    # The original is not passed on: it could hold a raw body or a stack trace.
    assert "Traceback" not in message.content
    assert "token" not in message.content
    assert reply.tool_calls[0].error == message.content
    assert_well_formed(model.calls[1])
    assert reply.reply == "It failed."
    assert len(provider.calls) == 1


async def test_an_unexpected_exception_from_a_read_is_internal(
    make_client, provider, fake_model, make_ai
):
    provider.replies["get_ticket"] = [KeyError("secret-token-in-a-message")]
    model = fake_model(
        responses=[make_ai("", ("get_ticket", {"ticket_id": "INC-0003"}, "r1")), make_ai("Hmm.")]
    )

    _, reply = await _turn(make_client, provider, model)

    (message,) = _tool_messages(model.calls[1])
    error = _error(message)
    assert error.code is ErrorCode.INTERNAL
    assert "get_ticket" in error.what
    assert "KeyError" in error.what
    assert "secret-token" not in message.content
    assert reply.status is ChatStatus.OK
    assert_well_formed(model.calls[1])
    assert len(provider.calls) == 1


async def test_a_failing_fact_store_is_internal(
    make_client, provider, fake_model, make_ai, monkeypatch
):
    from langgraph.store.sqlite.aio import AsyncSqliteStore  # noqa: PLC0415

    async def broken_put(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(AsyncSqliteStore, "aput", broken_put)
    model = fake_model(
        responses=[
            make_ai("", ("remember_fact", {"key": "team", "value": "Finance"}, "m1")),
            make_ai("I could not save that."),
        ]
    )

    _, reply = await _turn(make_client, provider, model)

    error = _error(_tool_messages(model.calls[1])[0])
    assert error.code is ErrorCode.INTERNAL
    assert "remember_fact" in error.what
    assert reply.tool_calls[0].error is not None
    assert_well_formed(model.calls[1])


async def _approve_with(make_client, provider, fake_model, make_ai, *after):
    model = fake_model(responses=[make_ai("", ("create_ticket", CREATE_ARGS, "w1")), *after])
    client, proposal = await _turn(make_client, provider, model)
    assert proposal.pending_action is not None
    return client, model, await _chat(client, message="yes")


@pytest.mark.parametrize(
    ("reply", "cause"),
    [
        pytest.param(RuntimeError("boom"), "raised RuntimeError", id="exception"),
        pytest.param(
            ToolResult(text="Created it!", is_error=False), "not a JSON object", id="not-json"
        ),
        pytest.param(ToolResult(text="", is_error=False), "empty result", id="empty"),
    ],
)
async def test_a_write_failure_says_the_change_could_not_be_confirmed(
    *, make_client, provider, fake_model, make_ai, reply, cause
):
    provider.replies["create_ticket"] = [reply]
    client, model, done = await _approve_with(
        make_client, provider, fake_model, make_ai, make_ai("I could not confirm it.")
    )

    (message,) = [m for m in _tool_messages(model.calls[-1]) if m.tool_call_id == "w1"]
    error = _error(message)
    assert error.code is ErrorCode.INTERNAL
    assert cause in error.what
    assert "could not be confirmed" in error.what
    assert error.next_step == UNCONFIRMED_WRITE_NEXT_STEP.rstrip(".")
    (record,) = done.tool_calls
    assert record.name == "create_ticket"
    assert record.error == message.content
    assert done.status is ChatStatus.OK
    assert done.reply == "I could not confirm it."
    assert_well_formed(model.calls[-1])
    # One attempt, no retry, and the action is not left pending.
    assert len(provider.writes()) == 1
    state = (await client.get("/sessions/alice/s1")).json()
    assert state["pending_action"] is None


async def test_a_write_cancelled_from_inside_reruns_with_the_same_key(
    make_client, provider, fake_model, make_ai
):
    async def cancelled(_args):
        # A CancelledError the node raises itself (not the turn deadline): LangGraph
        # turns it into NodeCancelledError. The write must not be recorded as done.
        raise asyncio.CancelledError

    provider.replies["create_ticket"] = [cancelled]
    client, model, stopped = await _approve_with(
        make_client, provider, fake_model, make_ai, make_ai("Created INC-0016."), make_ai("Hi.")
    )

    assert stopped.status is ChatStatus.ERROR
    assert stopped.reply == turn_errors.INTERRUPTED_WRITE_UNFINISHED_REPLY
    assert stopped.pending_action is None

    later = await _chat(client, message="hello?")
    keys = [w.idempotency_key for w in provider.writes()]
    assert len(keys) == 2
    assert len(set(keys)) == 1
    assert later.reply == "Hi."
    assert [r.name for r in later.tool_calls] == ["create_ticket"]
    assert_well_formed(model.calls[-1])


async def test_a_decline_is_an_error_result_and_is_not_listed(
    make_client, provider, fake_model, make_ai
):
    model = fake_model(
        responses=[make_ai("", ("create_ticket", CREATE_ARGS, "w1")), make_ai("OK, I won't.")]
    )
    client, _ = await _turn(make_client, provider, model)

    declined = await _chat(client, message="no")

    (message,) = [m for m in _tool_messages(model.calls[-1]) if m.tool_call_id == "w1"]
    assert message.status == "error"
    assert message.content == USER_DECLINED
    assert declined.tool_calls == []
    assert_well_formed(model.calls[-1])
    assert provider.writes() == []


async def test_a_read_the_provider_did_not_list_is_unknown(
    make_client, provider, fake_model, make_ai
):
    class NoGetTicket(type(provider)):
        async def list_tools(self):
            return [d for d in await super().list_tools() if d.name != "get_ticket"]

    limited = NoGetTicket()
    model = fake_model(
        responses=[make_ai("", ("get_ticket", {"ticket_id": "INC-0003"}, "r1")), make_ai("Hm.")]
    )

    _, reply = await _turn(make_client, limited, model)

    error = _error(_tool_messages(model.calls[1])[0])
    assert error.code is ErrorCode.INVALID_ARGUMENT
    assert "get_ticket" not in error.next_step
    assert "search_tickets" in error.next_step
    _rejected_record(reply.tool_calls[0], "get_ticket", ErrorCode.INVALID_ARGUMENT)
    assert_well_formed(model.calls[1])
    assert limited.calls == []


# --- the whole history ------------------------------------------------------------------


async def test_the_final_history_formats_as_a_valid_claude_request(
    *, make_client, make_graph, provider, fake_model, make_ai, transport_error
):
    """Every kind of bad reply in one session; the stored history still converts cleanly."""
    provider.replies["search_tickets"] = [transport_error("down")]
    provider.replies["get_ticket"] = [ToolResult(text="", is_error=False)]
    turns = [
        AIMessage(content="", invalid_tool_calls=[_invalid("search_tickets", "{", "bad1")]),
        make_ai("", ("close_all_tickets", {}, "u1")),
        AIMessage(
            content="",
            tool_calls=[{"name": "get_ticket", "args": {"ticket_id": "INC-1"}, "id": None}],
        ),
        make_ai(
            "", ("search_tickets", {"query": "vpn"}, "r1"), ("create_ticket", CREATE_ARGS, "w1")
        ),
        make_ai("", ("search_tickets", {"query": "vpn"}, "r2")),
        make_ai("", ("get_ticket", {"ticket_id": "INC-0003"}, "r3")),
        AIMessage("   "),
        make_ai("", ("add_comment", COMMENT_ARGS, "w2")),
        make_ai("I won't add it."),
        AIMessage("cut", response_metadata={"stop_reason": "max_tokens"}),
    ]
    model = fake_model(responses=turns)
    client = await make_client(model, provider)
    await _chat(client, message="first")
    await _chat(client, message="second")
    await _chat(client, message="no")
    await _chat(client, message="third")

    graph = await make_graph(fake_model(responses=[]), provider)
    stored = (await graph.aget_state(CONFIG)).values["messages"]
    history = model_messages(stored, token_budget=100_000)
    assert_well_formed(history)
    for sent in model.calls:
        assert_well_formed(sent)
    payload = ANTHROPIC._get_request_payload(history)
    assert json.dumps(payload["messages"])
    assert provider.writes() == []
    assert rejected_batch_error(["search_tickets", "create_ticket"]) in [
        m.content for m in _tool_messages(stored)
    ]
