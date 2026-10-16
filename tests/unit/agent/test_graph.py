"""Graph topology, router and node rules (CONTRACTS §5.1, §5.3)."""

from typing import Any

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from pydantic import Field, SecretStr

from helpdesk.agent.decision import resume_value
from helpdesk.agent.graph import InjectedArgumentVisibleError
from helpdesk.agent.prompt import SYSTEM_PROMPT_ID
from helpdesk.agent.prompt_caching import CACHE_CONTROL
from helpdesk.agent.run_config import build_run_config
from helpdesk.agent.tool_errors import rejected_batch_error, transport_failure
from helpdesk.agent.tool_messages import USER_DECLINED
from helpdesk.agent.tool_provider import ToolDefinition, ToolResult
from helpdesk.shared.approval import PendingAction, WriteResult
from helpdesk.shared.errors import ErrorCode, parse_tool_error
from helpdesk.shared.tools import ToolName

CONFIG = build_run_config("alice", "s1", recursion_limit=25)
CREATE_ARGS = {
    "title": "VPN drops every hour",
    "description": "The VPN disconnects about once an hour since this morning.",
    "category": "network",
    "priority": "high",
}


def _start(text: str = "My VPN keeps dropping") -> dict[str, object]:
    return {"messages": [HumanMessage(text)], "user_id": "alice"}


def _tool_messages(state: dict[str, object]) -> list[ToolMessage]:
    messages = state["messages"]
    assert isinstance(messages, list)
    return [m for m in messages if isinstance(m, ToolMessage)]


async def _propose_create(make_graph, provider, fake_model, make_ai, *, after: list | None = None):
    model = fake_model(
        responses=[
            make_ai("", ("search_tickets", {"query": "vpn"}, "c1")),
            make_ai("I'll open a ticket.", ("create_ticket", CREATE_ARGS, "w1")),
            *(after or [make_ai("Done.")]),
        ]
    )
    graph = await make_graph(model, provider)
    result = await graph.ainvoke(_start(), CONFIG)
    return graph, model, result


async def test_read_tools_route_to_tools_and_back(make_graph, provider, fake_model, make_ai):
    model = fake_model(
        responses=[
            make_ai(
                "",
                ("search_tickets", {"query": "vpn"}, "c1"),
                ("get_ticket", {"ticket_id": "INC-0003"}, "c2"),
            ),
            make_ai("INC-0003 already covers this."),
        ]
    )
    graph = await make_graph(model, provider)
    result = await graph.ainvoke(_start(), CONFIG)

    assert [c.name for c in provider.calls] == ["search_tickets", "get_ticket"]
    assert [m.tool_call_id for m in _tool_messages(result)] == ["c1", "c2"]
    assert len(model.calls) == 2
    snapshot = await graph.aget_state(CONFIG)
    assert snapshot.next == ()
    assert snapshot.values.get("pending_action") is None


async def test_a_write_routes_to_propose_then_interrupts(make_graph, provider, fake_model, make_ai):
    graph, _, result = await _propose_create(make_graph, provider, fake_model, make_ai)

    assert provider.writes() == []
    pending = result["pending_action"]
    assert isinstance(pending, PendingAction)
    assert pending.tool is ToolName.CREATE_TICKET
    assert pending.args == CREATE_ARGS
    assert pending.tool_call_id == "w1"
    assert pending.summary == 'Create a high-priority network ticket: "VPN drops every hour"'
    interrupts = result["__interrupt__"]
    assert len(interrupts) == 1
    assert interrupts[0].value == pending
    snapshot = await graph.aget_state(CONFIG)
    assert snapshot.next == ("approval",)


async def test_decline_writes_nothing(make_graph, provider, fake_model, make_ai):
    graph, model, result = await _propose_create(
        make_graph, provider, fake_model, make_ai, after=[make_ai("OK, I won't.")]
    )
    pending = result["pending_action"]

    result = await graph.ainvoke(
        Command(resume=resume_value(pending.action_id, approve=False)), CONFIG
    )

    assert provider.writes() == []
    assert result["pending_action"] is None
    decline = _tool_messages(result)[-1]
    assert decline.tool_call_id == "w1"
    assert decline.content == USER_DECLINED
    # The write did not run, so its result is an error (CONTRACTS §5.3).
    assert decline.status == "error"
    assert len(model.calls) == 3


async def test_approve_executes_exactly_one_write_with_the_proposed_key(
    make_graph, provider, fake_model, make_ai
):
    graph, _, result = await _propose_create(make_graph, provider, fake_model, make_ai)
    pending = result["pending_action"]

    result = await graph.ainvoke(
        Command(resume=resume_value(pending.action_id, approve=True)), CONFIG
    )

    writes = provider.writes()
    assert len(writes) == 1
    assert writes[0].idempotency_key == pending.idempotency_key
    # The owner comes from the run config, never from the model's arguments.
    assert writes[0].args == {
        **CREATE_ARGS,
        "reporter_id": "alice",
        "idempotency_key": pending.idempotency_key,
    }
    assert result["pending_action"] is None
    last_write = result["last_write"]
    assert isinstance(last_write, WriteResult)
    assert last_write.ok
    assert last_write.idempotency_key == pending.idempotency_key
    assert _tool_messages(result)[-1].tool_call_id == "w1"


async def test_add_comment_injects_author_id(make_graph, provider, fake_model, make_ai):
    model = fake_model(
        responses=[
            make_ai("", ("add_comment", {"ticket_id": "INC-0003", "body": "Again today."}, "w1")),
            make_ai("Comment added."),
        ]
    )
    graph = await make_graph(model, provider)
    result = await graph.ainvoke(_start(), CONFIG)
    pending = result["pending_action"]
    await graph.ainvoke(Command(resume=resume_value(pending.action_id, approve=True)), CONFIG)

    (write,) = provider.writes()
    assert write.args["author_id"] == "alice"
    assert write.args["idempotency_key"] == pending.idempotency_key


async def test_resume_with_a_mismatched_action_id_never_writes(
    make_graph, provider, fake_model, make_ai
):
    graph, _, _ = await _propose_create(make_graph, provider, fake_model, make_ai)

    await graph.ainvoke(Command(resume=resume_value("other-action", approve=True)), CONFIG)

    assert provider.writes() == []


async def test_model_cannot_supply_injected_args(make_graph, provider, fake_model, make_ai):
    forged = {**CREATE_ARGS, "reporter_id": "bob", "idempotency_key": "replayed-key"}
    model = fake_model(responses=[make_ai("", ("create_ticket", forged, "w1")), make_ai("Sorry.")])
    graph = await make_graph(model, provider)
    result = await graph.ainvoke(_start(), CONFIG)

    assert result.get("pending_action") is None
    error = parse_tool_error(str(_tool_messages(result)[-1].content))
    assert error is not None
    assert error.code is ErrorCode.INVALID_ARGUMENT
    assert provider.writes() == []


@pytest.mark.parametrize(
    "calls",
    [
        pytest.param(
            [
                ("create_ticket", CREATE_ARGS, "w1"),
                ("add_comment", {"ticket_id": "INC-0003", "body": "x"}, "w2"),
            ],
            id="two-writes",
        ),
        pytest.param(
            [("search_tickets", {"query": "vpn"}, "r1"), ("create_ticket", CREATE_ARGS, "w1")],
            id="write-mixed-with-read",
        ),
        pytest.param(
            [
                ("create_ticket", CREATE_ARGS, "w1"),
                ("remember_fact", {"key": "team", "value": "x"}, "m1"),
            ],
            id="write-mixed-with-remember-fact",
        ),
    ],
)
async def test_parallel_or_mixed_writes_are_rejected_call_by_call(
    make_graph, provider, fake_model, make_ai, calls
):
    model = fake_model(responses=[make_ai("", *calls), make_ai("Let me do one at a time.")])
    graph = await make_graph(model, provider)
    result = await graph.ainvoke(_start(), CONFIG)

    tool_messages = _tool_messages(result)
    assert [m.tool_call_id for m in tool_messages] == [call[2] for call in calls]
    expected = rejected_batch_error([call[0] for call in calls])
    assert all(m.content == expected for m in tool_messages)
    assert all(m.status == "error" for m in tool_messages)
    assert expected.startswith(
        "ERROR INVALID_ARGUMENT: propose one change at a time, after your reads. Next step: "
    )
    # Specific (CONTRACTS §2.4): it says what was combined.
    assert all(call[0] in expected for call in calls)
    assert parse_tool_error(expected) is not None
    assert provider.calls == []
    assert result.get("pending_action") is None
    assert len(model.calls) == 2


async def test_unknown_tool_gets_invalid_argument(make_graph, provider, fake_model, make_ai):
    model = fake_model(
        responses=[
            make_ai(
                "",
                ("close_all_tickets", {}, "u1"),
                ("search_tickets", {"query": "vpn"}, "r1"),
            ),
            make_ai("Done."),
        ]
    )
    graph = await make_graph(model, provider)
    result = await graph.ainvoke(_start(), CONFIG)

    unknown, read = _tool_messages(result)
    error = parse_tool_error(str(unknown.content))
    assert error is not None
    assert error.code is ErrorCode.INVALID_ARGUMENT
    assert unknown.status == "error"
    assert read.status == "success"
    assert [c.name for c in provider.calls] == ["search_tickets"]


async def test_invalid_write_args_create_no_pending_action(
    make_graph, provider, fake_model, make_ai
):
    model = fake_model(
        responses=[
            make_ai("", ("add_comment", {"ticket_id": "INC-7", "body": "x"}, "w1")),
            make_ai("Which ticket?"),
        ]
    )
    graph = await make_graph(model, provider)
    result = await graph.ainvoke(_start(), CONFIG)

    assert result.get("pending_action") is None
    (message,) = _tool_messages(result)
    error = parse_tool_error(str(message.content))
    assert error is not None
    assert error.code is ErrorCode.INVALID_ARGUMENT
    assert "ticket_id" in error.what


async def test_mcp_tool_error_reaches_the_model_as_an_error_tool_message(
    make_graph, provider, fake_model, make_ai
):
    not_found = "ERROR NOT_FOUND: Ticket INC-0099 does not exist. Next step: Check the id."
    provider.replies["get_ticket"] = [ToolResult(text=not_found, is_error=True)]
    model = fake_model(
        responses=[make_ai("", ("get_ticket", {"ticket_id": "INC-0099"}, "r1")), make_ai("?")]
    )
    graph = await make_graph(model, provider)
    result = await graph.ainvoke(_start(), CONFIG)

    (message,) = _tool_messages(result)
    assert message.content == not_found
    assert message.status == "error"


async def test_read_transport_failure_is_upstream_unavailable_without_retry(
    make_graph, provider, fake_model, make_ai, transport_error
):
    provider.replies["search_tickets"] = [transport_error("connection refused")]
    model = fake_model(
        responses=[make_ai("", ("search_tickets", {"query": "vpn"}, "r1")), make_ai("Down.")]
    )
    graph = await make_graph(model, provider)
    result = await graph.ainvoke(_start(), CONFIG)

    (message,) = _tool_messages(result)
    assert message.content == transport_failure("search_tickets", write=False)
    assert message.status == "error"
    assert parse_tool_error(message.content).code is ErrorCode.UPSTREAM_UNAVAILABLE
    assert "search_tickets" in message.content
    assert len(provider.calls) == 1


async def test_write_transport_failure_is_upstream_unavailable_without_retry(
    make_graph, provider, fake_model, make_ai, transport_error
):
    provider.replies["create_ticket"] = [transport_error("session lost")]
    graph, _, result = await _propose_create(
        make_graph, provider, fake_model, make_ai, after=[make_ai("The system is down.")]
    )
    pending = result["pending_action"]

    result = await graph.ainvoke(
        Command(resume=resume_value(pending.action_id, approve=True)), CONFIG
    )

    assert len(provider.writes()) == 1
    last_write = result["last_write"]
    assert not last_write.ok
    assert last_write.error_code is ErrorCode.UPSTREAM_UNAVAILABLE
    text = _tool_messages(result)[-1].content
    assert text == transport_failure("create_ticket", write=True)
    assert "could not be confirmed" in text
    assert "do not make up a ticket id" in text
    assert result["pending_action"] is None


async def test_system_prompt_is_first_and_replaced_not_duplicated(
    make_graph, provider, fake_model, make_ai
):
    model = fake_model(responses=[make_ai("Hi."), make_ai("Hi again.")])
    graph = await make_graph(model, provider)
    await graph.ainvoke(_start("hello"), CONFIG)
    result = await graph.ainvoke(_start("hello again"), CONFIG)

    systems = [m for m in result["messages"] if isinstance(m, SystemMessage)]
    assert [m.id for m in systems] == [SYSTEM_PROMPT_ID]
    for call in model.calls:
        assert isinstance(call[0], SystemMessage)
        assert sum(isinstance(m, SystemMessage) for m in call) == 1


class RecordingAnthropic(ChatAnthropic):
    """ChatAnthropic that records what the agent node sends instead of calling the API."""

    calls: list[tuple[list[BaseMessage], dict[str, Any]]] = Field(default_factory=list)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append((list(messages), kwargs))
        return ChatResult(generations=[ChatGeneration(message=AIMessage("Hi."))])


async def test_agent_node_marks_the_system_prompt_and_tools_cacheable(make_graph, provider):
    model = RecordingAnthropic(model="claude-test", api_key=SecretStr("not-a-real-key"))
    graph = await make_graph(model, provider)
    await graph.ainvoke(_start("hello"), CONFIG)

    [(messages, kwargs)] = model.calls
    system = messages[0]
    assert isinstance(system, SystemMessage)
    assert isinstance(system.content, list)
    assert system.content[-1]["cache_control"] == CACHE_CONTROL
    assert kwargs["cache_control"] == CACHE_CONTROL
    assert kwargs["tools"][-1]["cache_control"] == CACHE_CONTROL
    # The checkpoint keeps the plain prompt; the cache marker only shapes the request.
    state = (await graph.aget_state(CONFIG)).values
    assert isinstance(state["messages"][0].content, str)


@pytest.mark.parametrize("case", [(8_000, True), (60, False)], ids=["default", "small"])
async def test_agent_node_trims_history_with_the_configured_budget(
    make_graph, provider, fake_model, make_ai, case
):
    budget, sees_first_turn = case
    first = "My laptop will not connect to the office printer. " * 10
    model = fake_model(responses=[make_ai("One."), make_ai("Two."), make_ai("Three.")])
    graph = await make_graph(model, provider, history_token_budget=budget)
    for text in (first, "second question", "third question"):
        await graph.ainvoke(_start(text), CONFIG)

    seen = [m.content for m in model.calls[-1] if isinstance(m, HumanMessage)]
    assert (first in seen) is sees_first_turn
    assert seen[-2:] == ["second question", "third question"]
    assert isinstance(model.calls[-1][0], SystemMessage)
    # Trimming shapes the model input only: the checkpoint keeps every message.
    state = (await graph.aget_state(CONFIG)).values
    assert sum(isinstance(m, HumanMessage) for m in state["messages"]) == 3


async def test_model_sees_no_injected_args(make_graph, provider, fake_model, make_ai):
    model = fake_model(responses=[make_ai("Hi.")])
    await make_graph(model, provider)

    names = {tool["function"]["name"] for tool in model.bound_tools}
    assert names == {
        "search_tickets",
        "get_ticket",
        "create_ticket",
        "add_comment",
        "remember_fact",
    }
    for tool in model.bound_tools:
        properties = set(tool["function"]["parameters"]["properties"])
        assert properties.isdisjoint({"reporter_id", "author_id", "idempotency_key", "user_id"})


async def test_tool_provider_exposing_injected_args_fails_fast(make_graph, fake_model, provider):
    class LeakyProvider(type(provider)):
        async def list_tools(self):
            schema = {"type": "object", "properties": {"title": {}, "reporter_id": {}}}
            return [ToolDefinition(name="create_ticket", description="x", input_schema=schema)]

    with pytest.raises(InjectedArgumentVisibleError, match="reporter_id"):
        await make_graph(fake_model(responses=[]), LeakyProvider())


async def test_recursion_limit_is_enforced(make_graph, provider, fake_model, make_ai):
    model = fake_model(
        responses=[make_ai("", ("search_tickets", {"query": "vpn"}, "loop"))], repeat_last=True
    )
    graph = await make_graph(model, provider)

    with pytest.raises(GraphRecursionError):
        await graph.ainvoke(_start(), build_run_config("alice", "s1", recursion_limit=6))
