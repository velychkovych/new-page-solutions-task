"""POST /chat/stream (CONTRACTS §4): event order, parity with /chat, and the background run.

Every stream is parsed strictly: each frame must be `event: <name>\\ndata: <compact JSON>\\n\\n`,
its data must validate against the shared.api model for that event, and the whole stream
must follow the §4 ordering rules (`assert_contract_order`).
"""

import asyncio
import json
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import reduce
from typing import Any

import httpx
import httpx2
import pytest
from langchain_anthropic.chat_models import AnthropicOverloadedError
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.messages.utils import message_chunk_to_message
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from opentelemetry.trace import format_trace_id
from pydantic import BaseModel

from helpdesk.agent import turn_errors
from helpdesk.agent.api import create_app
from helpdesk.agent.service import STALE_DECISION_REPLY, reask_reply
from helpdesk.agent.stream import REPLY_SEPARATOR, TurnEvents
from helpdesk.agent.tool_activity import ToolStarted, progress_message, slow_tool_message
from helpdesk.agent.tool_provider import ToolResult
from helpdesk.shared import trace_attributes as attrs
from helpdesk.shared.api import (
    ChatResponse,
    ChatStatus,
    DoneData,
    ErrorData,
    PendingActionView,
    SseEvent,
    StatusData,
    TokenData,
    ToolEndData,
    ToolStartData,
)
from helpdesk.shared.approval import PendingAction

CREATE_ARGS = {
    "title": "Laptop battery drains fast",
    "description": "Battery goes from 100% to 20% in an hour.",
    "category": "hardware",
    "priority": "medium",
}
CONFLICT = "ERROR CONFLICT: Key reused with a different body. Next step: Do not retry."
UNAVAILABLE = (
    "ERROR UPSTREAM_UNAVAILABLE: search_tickets got 503 on all 3 attempts. "
    "Next step: Tell the user the ticketing system is unavailable."
)
_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
OVERLOADED = AnthropicOverloadedError(
    message="Overloaded", response=httpx2.Response(529, request=_REQUEST), body=None
)

DATA_MODELS: dict[SseEvent, type[BaseModel]] = {
    SseEvent.TOKEN: TokenData,
    SseEvent.TOOL_START: ToolStartData,
    SseEvent.TOOL_END: ToolEndData,
    SseEvent.STATUS: StatusData,
    SseEvent.APPROVAL_REQUIRED: PendingActionView,
    SseEvent.ERROR: ErrorData,
    SseEvent.DONE: DoneData,
}
FRAME = re.compile(r"event: (?P<name>[a-z_]+)\ndata: (?P<data>[^\n]*)\n\n")
COMMENT = re.compile(r"(?::[^\n]*\n)+\n")


class CrashInsideWriteError(BaseException):
    """Stands in for the process dying mid-call (see test_api.py)."""


@dataclass(frozen=True)
class Event:
    name: SseEvent
    data: Any


# --- parsing and the §4 rules ----------------------------------------------------------


def parse_sse(body: str) -> list[Event]:
    events: list[Event] = []
    rest = body
    while rest:
        if comment := COMMENT.match(rest):
            rest = rest[comment.end() :]
            continue
        frame = FRAME.match(rest)
        assert frame is not None, f"not an SSE frame: {rest[:120]!r}"
        name = SseEvent(frame["name"])
        data = DATA_MODELS[name].model_validate_json(frame["data"])
        # Compact JSON, exactly as the shared model writes it.
        assert frame["data"] == data.model_dump_json()
        events.append(Event(name, data))
        rest = rest[frame.end() :]
    return events


def assert_contract_order(events: list[Event]) -> None:
    names = [e.name for e in events]
    assert names.count(SseEvent.DONE) == 1, names
    assert names[-1] is SseEvent.DONE, names
    done: DoneData = events[-1].data
    assert names.count(SseEvent.ERROR) <= 1, names
    if SseEvent.ERROR in names:
        assert done.status is ChatStatus.ERROR
    assert names.count(SseEvent.APPROVAL_REQUIRED) <= 1, names
    if done.status is ChatStatus.AWAITING_APPROVAL:
        assert names[-2] is SseEvent.APPROVAL_REQUIRED, names
    else:
        assert SseEvent.APPROVAL_REQUIRED not in names, names
    open_calls: set[str] = set()
    for event in events:
        if event.name is SseEvent.TOOL_START:
            assert event.data.call_id not in open_calls, "a second tool_start for an open call"
            open_calls.add(event.data.call_id)
        elif event.name is SseEvent.TOOL_END:
            assert event.data.call_id in open_calls, "tool_end without its tool_start"
            open_calls.remove(event.data.call_id)


def tokens(events: list[Event]) -> str:
    return "".join(e.data.text for e in events if e.name is SseEvent.TOKEN)


def named(events: list[Event], name: SseEvent) -> list[Any]:
    return [e.data for e in events if e.name is name]


def tool_records(events: list[Event]) -> list[tuple[str, dict[str, Any], str, str | None]]:
    """(name, args, result_summary, error) per tool_end, as /chat's tool_calls lists them."""
    starts: dict[str, ToolStartData] = {}
    records = []
    for event in events:
        if event.name is SseEvent.TOOL_START:
            starts[event.data.call_id] = event.data
        elif event.name is SseEvent.TOOL_END:
            start = starts[event.data.call_id]
            records.append((start.name, start.args, event.data.result_summary, event.data.error))
    return records


def chat_records(response: ChatResponse) -> list[tuple[str, dict[str, Any], str, str | None]]:
    return [(r.name, r.args, r.result_summary, r.error) for r in response.tool_calls]


BODY = {"user_id": "alice", "session_id": "s1"}


async def _stream_response(client: httpx.AsyncClient, **body: Any) -> httpx.Response:
    response = await client.post("/chat/stream", json={**BODY, **body})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return response


async def _stream(client: httpx.AsyncClient, **body: Any) -> list[Event]:
    events = parse_sse((await _stream_response(client, **body)).text)
    assert_contract_order(events)
    return events


async def _chat(client: httpx.AsyncClient, **body: Any) -> ChatResponse:
    response = await client.post("/chat", json={**BODY, **body})
    assert response.status_code == 200, response.text
    return ChatResponse.model_validate(response.json())


def _proposal_script(make_ai: Callable[..., AIMessage], *after: AIMessage) -> list[AIMessage]:
    return [
        make_ai("", ("search_tickets", {"query": "battery"}, "r1")),
        make_ai("No duplicate found.", ("create_ticket", CREATE_ARGS, "w1")),
        *after,
    ]


PROPOSAL_MESSAGE = "My laptop battery dies within an hour"


# --- parity with /chat -----------------------------------------------------------------


@dataclass
class Scenario:
    script: Callable[[Callable[..., AIMessage]], list[AIMessage]]
    final: dict[str, Any]
    setup: list[dict[str, Any]] = field(default_factory=list)
    model: dict[str, Any] = field(default_factory=dict)
    replies: Callable[[], dict[str, list[Any]]] = dict
    overrides: dict[str, Any] = field(default_factory=dict)


SCENARIOS = {
    "read": Scenario(
        lambda ai: [ai("", ("search_tickets", {"query": "vpn"}, "r1")), ai("INC-0003 is open.")],
        {"message": "Is there a VPN ticket?"},
    ),
    "proposal": Scenario(_proposal_script, {"message": PROPOSAL_MESSAGE}),
    "approve": Scenario(
        lambda ai: _proposal_script(ai, ai("Created INC-0016.")),
        {"message": "yes"},
        setup=[{"message": PROPOSAL_MESSAGE}],
    ),
    "decline": Scenario(
        lambda ai: _proposal_script(ai, ai("OK, nothing was created.")),
        {"message": "no"},
        setup=[{"message": PROPOSAL_MESSAGE}],
    ),
    "reask": Scenario(
        _proposal_script,
        {"message": "what will this do?"},
        setup=[{"message": PROPOSAL_MESSAGE}],
    ),
    "stale-decision": Scenario(
        _proposal_script,
        {"decision": {"action_id": "not-the-pending-one", "approve": True}},
        setup=[{"message": PROPOSAL_MESSAGE}],
    ),
    "decision-without-pending": Scenario(
        lambda ai: [], {"decision": {"action_id": "a1", "approve": True}}
    ),
    "read-tool-error": Scenario(
        lambda ai: [
            ai("", ("search_tickets", {"query": "vpn"}, "r1")),
            ai("The ticketing system is down."),
        ],
        {"message": "Is there a VPN ticket?"},
        replies=lambda: {"search_tickets": [ToolResult(text=UNAVAILABLE, is_error=True)]},
    ),
    "write-error": Scenario(
        lambda ai: _proposal_script(ai, ai("It was not created.")),
        {"message": "yes"},
        setup=[{"message": PROPOSAL_MESSAGE}],
        replies=lambda: {"create_ticket": [ToolResult(text=CONFLICT, is_error=True)]},
    ),
    "rejected-batch": Scenario(
        lambda ai: [
            ai(
                "",
                ("search_tickets", {"query": "vpn"}, "r1"),
                ("create_ticket", CREATE_ARGS, "w1"),
            ),
            ai("I will search first."),
        ],
        {"message": "Create a VPN ticket"},
    ),
    "reused-id": Scenario(
        # Each result belongs to the call that was current when it ran, not to the last
        # call with that id.
        lambda ai: [
            ai("", ("search_tickets", {"query": "a"}, "r1")),
            ai("", ("search_tickets", {"query": "b"}, "r1")),
            ai("Done."),
        ],
        {"message": "Is there a VPN ticket?"},
    ),
    "blank-answer": Scenario(lambda ai: [ai("   ")], {"message": "hello"}),
    "model-error": Scenario(lambda ai: [], {"message": "hello"}, model={"raises": OVERLOADED}),
    "deadline": Scenario(
        lambda ai: [ai("too late")],
        {"message": "hello"},
        model={"delay_s": 2.0},
        overrides={"turn_deadline_s": 0.1},
    ),
    "recursion-limit": Scenario(
        lambda ai: [ai("", ("search_tickets", {"query": "vpn"}, "loop"))],
        {"message": "loop forever"},
        model={"repeat_last": True},
        overrides={"recursion_limit": 5},
    ),
    "cut-off-proposal": Scenario(
        lambda ai: _proposal_script(ai, ai("Created.")),
        {"message": "yes"},
        setup=[{"message": PROPOSAL_MESSAGE}],
        overrides={"recursion_limit": 5},
    ),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
async def test_stream_matches_chat_for_the_same_scripted_turn(
    *, name, make_client, provider, fake_model, make_ai, tmp_path
):
    scenario = SCENARIOS[name]

    async def final_turn(mode: str):
        tool_provider = type(provider)(replies=scenario.replies())
        model = fake_model(responses=scenario.script(make_ai), **scenario.model)
        client = await make_client(
            model,
            tool_provider,
            checkpoint_db_path=tmp_path / f"{mode}-checkpoints.sqlite",
            store_db_path=tmp_path / f"{mode}-store.sqlite",
            **scenario.overrides,
        )
        for body in scenario.setup:
            await _chat(client, **body)
        if mode == "chat":
            return await _chat(client, **scenario.final), tool_provider
        return await _stream(client, **scenario.final), tool_provider

    chat, chat_provider = await final_turn("chat")
    events, stream_provider = await final_turn("stream")

    done: DoneData = events[-1].data
    assert done.status is chat.status
    assert tool_records(events) == chat_records(chat)
    if chat.status is ChatStatus.ERROR:
        assert [e.message for e in named(events, SseEvent.ERROR)] == [chat.reply]
    else:
        assert named(events, SseEvent.ERROR) == []
        # None of these scripts has text in an intermediate reply.
        assert tokens(events) == chat.reply
    approvals = named(events, SseEvent.APPROVAL_REQUIRED)
    if chat.pending_action is None:
        assert approvals == []
    else:
        # Action ids are fresh uuids per run; everything else must agree.
        (approval,) = approvals
        ignore = {"action_id"}
        assert approval.model_dump(exclude=ignore) == chat.pending_action.model_dump(exclude=ignore)
    assert len(stream_provider.writes()) == len(chat_provider.writes())


async def test_chat_lists_each_call_with_its_own_args_when_an_id_is_reused(
    make_client, provider, fake_model, make_ai
):
    model = fake_model(responses=SCENARIOS["reused-id"].script(make_ai))
    client = await make_client(model, provider)
    chat = await _chat(client, message="Is there a VPN ticket?")
    assert [(r.name, r.args) for r in chat.tool_calls] == [
        ("search_tickets", {"query": "a"}),
        ("search_tickets", {"query": "b"}),
    ]


# --- single turns ----------------------------------------------------------------------


async def test_read_turn_streams_tools_then_text_then_done_with_the_request_trace(
    make_client, provider, fake_model, make_ai, spans
):
    provider.replies["search_tickets"] = [ToolResult(text="r" * 500, is_error=False)]
    model = fake_model(
        responses=[
            make_ai("", ("search_tickets", {"query": "vpn"}, "r1")),
            make_ai("INC-0003 is open."),
        ]
    )
    client = await make_client(model, provider)

    events = await _stream(client, message="Is there a VPN ticket?")

    assert [e.name for e in events] == [
        SseEvent.TOOL_START,
        SseEvent.TOOL_END,
        SseEvent.TOKEN,
        SseEvent.DONE,
    ]
    start, end, token, done = (e.data for e in events)
    assert start == ToolStartData(call_id="r1", name="search_tickets", args={"query": "vpn"})
    assert end.call_id == "r1"
    assert end.error is None
    assert len(end.result_summary) == 200
    assert token.text == "INC-0003 is open."
    assert done.status is ChatStatus.OK
    (request_span,) = [s for s in spans[1].get_finished_spans() if s.name == attrs.SPAN_REQUEST]
    assert done.trace_id == format_trace_id(request_span.context.trace_id)
    assert request_span.attributes[attrs.REQUEST_STATUS] == "ok"


async def test_reask_turn_is_exact_on_the_wire(make_client, provider, fake_model, make_ai):
    client = await make_client(fake_model(responses=_proposal_script(make_ai)), provider)
    proposal = await _chat(client, message=PROPOSAL_MESSAGE)
    pending = proposal.pending_action

    response = await _stream_response(client, message="maybe")

    trace_id = parse_sse(response.text)[-1].data.trace_id
    pending_json = json.dumps(pending.model_dump(mode="json"), separators=(",", ":"))
    text = json.dumps(
        reask_reply(PendingAction.model_construct(summary=pending.summary)),
        separators=(",", ":"),
        ensure_ascii=False,
    )
    assert response.text == (
        f'event: token\ndata: {{"text":{text}}}\n\n'
        f"event: approval_required\ndata: {pending_json}\n\n"
        f'event: done\ndata: {{"trace_id":"{trace_id}","status":"awaiting_approval"}}\n\n'
    )
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"


async def test_proposal_turn_asks_after_the_text_and_never_starts_the_write(
    make_client, provider, fake_model, make_ai
):
    client = await make_client(fake_model(responses=_proposal_script(make_ai)), provider)

    events = await _stream(client, message=PROPOSAL_MESSAGE)

    assert [d.call_id for d in named(events, SseEvent.TOOL_START)] == ["r1"]
    (approval,) = named(events, SseEvent.APPROVAL_REQUIRED)
    assert approval.tool == "create_ticket"
    assert approval.args == CREATE_ARGS
    text = tokens(events)
    assert text.startswith("No duplicate found." + REPLY_SEPARATOR)
    assert approval.summary in text
    assert "idempotency_key" not in json.dumps([e.data.model_dump() for e in events])
    assert provider.writes() == []


async def test_approve_turn_streams_the_write_under_the_proposed_call_id(
    make_client, provider, fake_model, make_ai
):
    model = fake_model(responses=_proposal_script(make_ai, make_ai("Created INC-0016.")))
    client = await make_client(model, provider)
    proposal = await _chat(client, message=PROPOSAL_MESSAGE)

    events = await _stream(
        client, decision={"action_id": proposal.pending_action.action_id, "approve": True}
    )

    assert [e.name for e in events] == [
        SseEvent.TOOL_START,
        SseEvent.TOOL_END,
        SseEvent.TOKEN,
        SseEvent.DONE,
    ]
    start = events[0].data
    assert start == ToolStartData(call_id="w1", name="create_ticket", args=CREATE_ARGS)
    assert tokens(events) == "Created INC-0016."
    (write,) = provider.writes()
    assert write.args["reporter_id"] == "alice"


async def test_decline_turn_lists_no_tool(make_client, provider, fake_model, make_ai):
    model = fake_model(responses=_proposal_script(make_ai, make_ai("OK, nothing was created.")))
    client = await make_client(model, provider)
    await _chat(client, message=PROPOSAL_MESSAGE)

    events = await _stream(client, message="no")

    assert [e.name for e in events] == [SseEvent.TOKEN, SseEvent.DONE]
    assert tokens(events) == "OK, nothing was created."
    assert provider.writes() == []


async def test_stale_decision_rebuilds_the_buttons_from_the_stream(
    make_client, provider, fake_model, make_ai
):
    client = await make_client(fake_model(responses=_proposal_script(make_ai)), provider)
    proposal = await _chat(client, message=PROPOSAL_MESSAGE)

    events = await _stream(client, decision={"action_id": "stale", "approve": True})

    assert [e.name for e in events] == [
        SseEvent.TOKEN,
        SseEvent.APPROVAL_REQUIRED,
        SseEvent.DONE,
    ]
    assert tokens(events) == STALE_DECISION_REPLY
    assert events[1].data == proposal.pending_action


async def test_model_error_ends_with_one_error_then_done(make_client, provider, fake_model):
    client = await make_client(fake_model(responses=[], raises=OVERLOADED), provider)

    events = await _stream(client, message="hello")

    assert [e.name for e in events] == [SseEvent.ERROR, SseEvent.DONE]
    assert events[0].data.message == turn_errors.OVERLOADED_REPLY
    assert events[1].data.status is ChatStatus.ERROR


async def _hang(_args):
    await asyncio.sleep(30)
    raise AssertionError("the turn deadline should have cancelled this call")


async def test_deadline_mid_tool_leaves_the_call_started_and_ends_with_error(
    make_client, provider, fake_model, make_ai
):
    provider.replies["search_tickets"] = [_hang]
    model = fake_model(responses=_proposal_script(make_ai))
    client = await make_client(model, provider, turn_deadline_s=0.3)

    events = await _stream(client, message=PROPOSAL_MESSAGE)

    # The cut-off call never answered, so /chat does not list it; the stream showed it start.
    assert [e.name for e in events] == [SseEvent.TOOL_START, SseEvent.ERROR, SseEvent.DONE]
    assert events[1].data.message == turn_errors.DEADLINE_REPLY


async def test_unexpected_failure_still_ends_the_stream(make_client, provider, fake_model):
    client = await make_client(fake_model(responses=[], raises=RuntimeError("boom")), provider)

    response = await _stream_response(client, message="hello")

    events = parse_sse(response.text)
    assert_contract_order(events)
    assert [e.name for e in events] == [SseEvent.ERROR, SseEvent.DONE]
    assert events[0].data.message == turn_errors.UNEXPECTED_ERROR_REPLY
    assert "boom" not in response.text


async def test_invalid_request_is_a_plain_422(make_client, provider, fake_model):
    model = fake_model(responses=[])
    client = await make_client(model, provider)

    response = await client.post("/chat/stream", json={"user_id": "al:ice", "session_id": "s1"})

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert "detail" in response.json()
    assert model.calls == []


# --- status events ---------------------------------------------------------------------


async def test_retry_progress_becomes_status_between_start_and_end(
    make_client, provider, fake_model, make_ai
):
    provider.progress = {"search_tickets": ["Retrying (attempt 2 of 3)", ""]}
    model = fake_model(
        responses=[make_ai("", ("search_tickets", {"query": "vpn"}, "r1")), make_ai("Done.")]
    )
    client = await make_client(model, provider)

    events = await _stream(client, message="Is there a VPN ticket?")

    assert [e.name for e in events] == [
        SseEvent.TOOL_START,
        SseEvent.STATUS,
        SseEvent.STATUS,
        SseEvent.TOOL_END,
        SseEvent.TOKEN,
        SseEvent.DONE,
    ]
    # The fake passes "" on as it is; McpToolProvider drops empty messages before this.
    assert [d.message for d in named(events, SseEvent.STATUS)] == [
        progress_message("search_tickets", "Retrying (attempt 2 of 3)"),
        progress_message("search_tickets", ""),
    ]


async def test_retry_progress_on_the_write_reaches_the_stream(
    make_client, provider, fake_model, make_ai
):
    provider.progress = {"create_ticket": ["Retrying (attempt 2 of 3)"]}
    model = fake_model(responses=_proposal_script(make_ai, make_ai("Created.")))
    client = await make_client(model, provider)
    await _chat(client, message=PROPOSAL_MESSAGE)

    events = await _stream(client, message="yes")

    assert [e.name for e in events[:3]] == [
        SseEvent.TOOL_START,
        SseEvent.STATUS,
        SseEvent.TOOL_END,
    ]
    assert events[1].data.message == progress_message("create_ticket", "Retrying (attempt 2 of 3)")


async def test_a_slow_tool_gets_exactly_one_notice(make_client, provider, fake_model, make_ai):
    provider.delay_s = 0.4
    model = fake_model(
        responses=[make_ai("", ("search_tickets", {"query": "vpn"}, "r1")), make_ai("Done.")]
    )
    client = await make_client(model, provider, slow_tool_notice_s=0.05)

    events = await _stream(client, message="Is there a VPN ticket?")

    assert [e.name for e in events] == [
        SseEvent.TOOL_START,
        SseEvent.STATUS,
        SseEvent.TOOL_END,
        SseEvent.TOKEN,
        SseEvent.DONE,
    ]
    assert events[1].data.message == slow_tool_message("search_tickets")


async def test_a_fast_tool_gets_no_notice(make_client, provider, fake_model, make_ai):
    model = fake_model(
        responses=[make_ai("", ("search_tickets", {"query": "vpn"}, "r1")), make_ai("Done.")]
    )
    client = await make_client(model, provider, slow_tool_notice_s=5.0)

    events = await _stream(client, message="Is there a VPN ticket?")

    assert named(events, SseEvent.STATUS) == []


async def test_chat_ignores_progress_and_slow_notices(make_client, provider, fake_model, make_ai):
    provider.progress = {"search_tickets": ["Retrying (attempt 2 of 3)"]}
    provider.delay_s = 0.2
    model = fake_model(
        responses=[make_ai("", ("search_tickets", {"query": "vpn"}, "r1")), make_ai("Done.")]
    )
    client = await make_client(model, provider, slow_tool_notice_s=0.05)

    result = await _chat(client, message="Is there a VPN ticket?")

    assert result.status is ChatStatus.OK
    assert result.reply == "Done."
    assert [r.name for r in result.tool_calls] == ["search_tickets"]


# --- model text ------------------------------------------------------------------------


class AnthropicShapedModel(BaseChatModel):
    """Streams replies as langchain-anthropic 1.7.6 builds its chunks (content blocks,
    tool_use, then input_json_delta fragments); `ainvoke` without a stream handler gets
    the merged reply."""

    replies: list[list[AIMessageChunk]]
    i: int = 0

    @property
    def _llm_type(self) -> str:
        return "anthropic-shaped-test"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _next(self) -> list[AIMessageChunk]:
        chunks = self.replies[self.i]
        self.i += 1
        return [chunk.model_copy(deep=True) for chunk in chunks]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        merged = reduce(operator.add, self._next())
        return ChatResult(generations=[ChatGeneration(message=message_chunk_to_message(merged))])

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        for chunk in self._next():
            yield ChatGenerationChunk(message=chunk)


def _text_block(text: str, index: int = 0) -> AIMessageChunk:
    return AIMessageChunk(content=[{"type": "text", "text": text, "index": index}])


def _anthropic_replies() -> list[list[AIMessageChunk]]:
    search_with_preamble = [
        AIMessageChunk(content=[], id="msg_01"),
        _text_block("Let me check "),
        _text_block("existing tickets.  "),
        AIMessageChunk(
            content=[
                {"type": "tool_use", "id": "toolu_1", "name": "search_tickets", "input": {}}
                | {"index": 1}
            ],
            tool_call_chunks=[
                tool_call_chunk(index=1, id="toolu_1", name="search_tickets", args="")
            ],
        ),
        AIMessageChunk(
            content=[{"type": "input_json_delta", "partial_json": '{"query": ', "index": 1}],
            tool_call_chunks=[tool_call_chunk(index=1, id=None, name=None, args='{"query": ')],
        ),
        AIMessageChunk(
            content=[{"type": "input_json_delta", "partial_json": '"vpn"}', "index": 1}],
            tool_call_chunks=[tool_call_chunk(index=1, id=None, name=None, args='"vpn"}')],
        ),
        AIMessageChunk(content=[], response_metadata={"stop_reason": "tool_use"}),
    ]
    answer = [
        AIMessageChunk(content=[], id="msg_02"),
        _text_block("\n INC-0003 "),
        _text_block(" "),
        _text_block("is open. \n"),
        AIMessageChunk(content=[], response_metadata={"stop_reason": "end_turn"}),
    ]
    return [search_with_preamble, answer]


async def test_streamed_text_skips_tool_json_and_keeps_chats_reply_last(
    make_client, provider, tmp_path
):
    stream_client = await make_client(AnthropicShapedModel(replies=_anthropic_replies()), provider)
    chat_client = await make_client(
        AnthropicShapedModel(replies=_anthropic_replies()),
        type(provider)(),
        checkpoint_db_path=tmp_path / "chat-checkpoints.sqlite",
        store_db_path=tmp_path / "chat-store.sqlite",
    )

    events = await _stream(stream_client, message="Is there a VPN ticket?")
    chat = await _chat(chat_client, message="Is there a VPN ticket?")

    assert chat.reply == "INC-0003  is open."  # /chat trims only the ends
    assert [d.text for d in named(events, SseEvent.TOKEN)] == [
        # Trailing whitespace waits for more text, so each reply ends trimmed.
        "Let me check",
        " existing tickets.",
        REPLY_SEPARATOR + "INC-0003",
        "  is open.",
    ]
    # Intermediate text first, then exactly /chat's reply.
    assert tokens(events) == "Let me check existing tickets." + REPLY_SEPARATOR + chat.reply
    assert "query" not in tokens(events)
    (start,) = named(events, SseEvent.TOOL_START)
    assert start == ToolStartData(call_id="toolu_1", name="search_tickets", args={"query": "vpn"})
    assert tool_records(events) == chat_records(chat)


async def test_service_text_after_streamed_text_is_separated(
    make_client, provider, fake_model, make_ai
):
    # An intermediate reply with text, then a blank answer: the fallback follows it.
    model = fake_model(
        responses=[
            make_ai("Let me look.", ("search_tickets", {"query": "vpn"}, "r1")),
            make_ai(""),
        ]
    )
    client = await make_client(model, provider)

    events = await _stream(client, message="Is there a VPN ticket?")

    assert tokens(
        events
    ) == "Let me look." + REPLY_SEPARATOR + "Sorry, I could not come up with " + (
        "a reply to that. Please try again, or rephrase your request."
    )


# --- the background run ----------------------------------------------------------------


async def test_unfinished_write_is_rerun_and_streamed_on_the_next_request(
    make_client, provider, fake_model, make_ai
):
    provider.replies["create_ticket"] = [CrashInsideWriteError("killed mid-write")]
    model = fake_model(responses=_proposal_script(make_ai, make_ai("Created."), make_ai("Hello.")))
    client = await make_client(model, provider)
    await _chat(client, message=PROPOSAL_MESSAGE)

    crashed = await _stream(client, message="yes")
    later = await _stream(client, message="hello?")

    # The crash is not an Exception, so nothing answered the call: the stream still ends.
    assert [e.name for e in crashed] == [SseEvent.TOOL_START, SseEvent.ERROR, SseEvent.DONE]
    assert crashed[1].data.message == turn_errors.UNEXPECTED_ERROR_REPLY
    assert [d.call_id for d in named(later, SseEvent.TOOL_END)] == ["w1"]
    assert tokens(later) == "Created." + REPLY_SEPARATOR + "Hello."
    keys = {w.idempotency_key for w in provider.writes()}
    assert len(provider.writes()) == 2
    assert len(keys) == 1


async def test_concurrent_stream_and_chat_on_one_thread_write_once(
    make_client, provider, fake_model, make_ai
):
    model = fake_model(responses=_proposal_script(make_ai, make_ai("Created.")))
    client = await make_client(model, provider)
    proposal = await _chat(client, message=PROPOSAL_MESSAGE)
    provider.delay_s = 0.2
    decision = {"action_id": proposal.pending_action.action_id, "approve": True}

    streamed, chatted = await asyncio.gather(
        _stream(client, decision=decision), _chat(client, decision=decision)
    )

    assert len(provider.writes()) == 1
    replies = sorted([tokens(streamed), chatted.reply])
    assert replies == sorted(["Created.", STALE_DECISION_REPLY])


async def test_a_failing_event_mapper_stops_the_write_before_the_next_turn(
    make_client, provider, fake_model, make_ai, monkeypatch
):
    """A bug in event mapping ends the stream; the graph's tasks must be stopped (and
    their cleanup awaited) before the thread lock goes to the next turn."""
    active = 0
    peak = 0

    async def slow_write(_args):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.3)
        except asyncio.CancelledError:
            await asyncio.sleep(0.2)  # a call that takes a while to unwind
            raise
        finally:
            active -= 1
        return ToolResult(text=json.dumps({"id": "INC-0016"}), is_error=False)

    provider.replies["create_ticket"] = [slow_write, slow_write]
    model = fake_model(responses=_proposal_script(make_ai, make_ai("Created.")), repeat_last=True)
    client = await make_client(model, provider)
    await _chat(client, message=PROPOSAL_MESSAGE)

    original = TurnEvents._on_signal
    failed = False

    def broken_on_signal(self, signal):
        nonlocal failed
        if isinstance(signal, ToolStarted) and signal.call_id == "w1" and not failed:
            failed = True
            raise RuntimeError("mapping bug")
        original(self, signal)

    monkeypatch.setattr(TurnEvents, "_on_signal", broken_on_signal)

    async def next_turn() -> ChatResponse:
        await asyncio.sleep(0.05)  # queued behind the stream on the thread lock
        return await _chat(client, message="hello")

    streamed, chatted = await asyncio.gather(_stream_response(client, message="yes"), next_turn())

    events = parse_sse(streamed.text)
    assert [e.name for e in events][-2:] == [SseEvent.ERROR, SseEvent.DONE]
    assert failed
    assert chatted.status is ChatStatus.OK
    assert peak == 1
    assert len({w.idempotency_key for w in provider.writes()}) == 1


def _asgi_scope(path: str) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"host", b"agent.test"), (b"content-type", b"application/json")],
        "server": ("agent.test", 80),
        "client": ("127.0.0.1", 50000),
        "root_path": "",
    }


async def _stream_until_first_event_then_disconnect(app: Any, body: dict[str, Any]) -> bytes:
    """Drive the ASGI app as a server would, with a client that leaves after one event."""
    payload = json.dumps(body).encode()
    first_event = asyncio.Event()
    received: list[bytes] = []
    request_sent = False

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await first_event.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            received.append(message["body"])
            first_event.set()

    await asyncio.wait_for(app(_asgi_scope("/chat/stream"), receive, send), timeout=5)
    return b"".join(received)


async def test_disconnect_mid_stream_lets_the_turn_finish_and_checkpoint(
    settings, provider, fake_model, make_ai, spans
):
    provider.delay_s = 0.5  # the search is still running when the client leaves
    model = fake_model(responses=_proposal_script(make_ai, make_ai("Created.")))
    app = create_app(
        settings=settings, tool_provider=provider, model=model, tracer=spans[0].get_tracer("t")
    )
    async with app.router.lifespan_context(app):
        seen = await _stream_until_first_event_then_disconnect(
            app, {**BODY, "message": PROPOSAL_MESSAGE}
        )
        assert seen.startswith(b"event: tool_start\n")
        assert b"event: done" not in seen

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://agent.test") as client:
            # Waits for the thread lock, so it acts on the finished turn's checkpoint.
            approved = await _chat(client, message="yes")

    assert approved.status is ChatStatus.OK
    assert approved.reply == "Created."
    assert len(provider.writes()) == 1
    request_spans = [s for s in spans[1].get_finished_spans() if s.name == attrs.SPAN_REQUEST]
    assert [s.attributes[attrs.REQUEST_STATUS] for s in request_spans] == [
        "awaiting_approval",
        "ok",
    ]


async def test_shutdown_waits_for_a_streamed_turn_whose_client_left(
    settings, provider, fake_model, make_ai, spans
):
    provider.delay_s = 0.3
    model = fake_model(responses=_proposal_script(make_ai))
    app = create_app(
        settings=settings, tool_provider=provider, model=model, tracer=spans[0].get_tracer("t")
    )
    async with app.router.lifespan_context(app):
        await _stream_until_first_event_then_disconnect(app, {**BODY, "message": PROPOSAL_MESSAGE})
    # The lifespan has closed the checkpointer only after the turn saved its proposal.
    request_spans = [s for s in spans[1].get_finished_spans() if s.name == attrs.SPAN_REQUEST]
    assert [s.attributes[attrs.REQUEST_STATUS] for s in request_spans] == ["awaiting_approval"]

    reopened = create_app(
        settings=settings,
        tool_provider=type(provider)(),
        model=fake_model(responses=[make_ai("Created.")]),
        tracer=spans[0].get_tracer("t"),
    )
    async with reopened.router.lifespan_context(reopened):
        transport = httpx.ASGITransport(app=reopened)
        async with httpx.AsyncClient(transport=transport, base_url="http://agent.test") as client:
            view = (await client.get("/sessions/alice/s1")).json()
    assert view["pending_action"]["tool"] == "create_ticket"


async def _start_stream_and_leave(app: Any, body: dict[str, Any]) -> None:
    """A client that starts a stream and disconnects before reading anything."""
    payload = json.dumps(body).encode()
    request_sent = False

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await asyncio.sleep(0.05)
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        return None

    await asyncio.wait_for(app(_asgi_scope("/chat/stream"), receive, send), timeout=5)


async def test_shutdown_does_not_start_queued_streamed_turns(
    settings, provider, fake_model, make_ai, spans
):
    """Shutdown waits for the running turn (bounded by its deadline) but not for the turns
    queued behind it: they end without running, so the drain is not N deadlines long."""
    deadline_s = 0.5
    model = fake_model(responses=[make_ai("late")], delay_s=30.0, repeat_last=True)
    app = create_app(
        settings=settings.model_copy(update={"turn_deadline_s": deadline_s}),
        tool_provider=provider,
        model=model,
        tracer=spans[0].get_tracer("t"),
    )
    async with app.router.lifespan_context(app):
        for i in range(4):
            await _start_stream_and_leave(app, {**BODY, "message": f"message {i}"})
        stopping = asyncio.get_running_loop().time()
    drained_s = asyncio.get_running_loop().time() - stopping

    assert drained_s < 2 * deadline_s
    assert len(model.calls) == 1  # only the turn that held the lock ever reached the model
    request_spans = [s for s in spans[1].get_finished_spans() if s.name == attrs.SPAN_REQUEST]
    assert [s.attributes[attrs.REQUEST_STATUS] for s in request_spans] == ["error"] * 4
