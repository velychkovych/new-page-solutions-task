"""Agent HTTP API (CONTRACTS §3): validation, approvals, decision safety, limits, traces."""

import asyncio
import json

import httpx2
import pytest
from langchain_anthropic.chat_models import (
    AnthropicAuthenticationError,
    AnthropicOverloadedError,
    AnthropicRateLimitError,
    AnthropicTimeoutError,
)
from opentelemetry.trace import StatusCode, format_trace_id

from helpdesk.agent import turn_errors
from helpdesk.agent.service import STALE_DECISION_REPLY
from helpdesk.agent.tool_provider import ToolResult
from helpdesk.shared import trace_attributes as attrs
from helpdesk.shared.api import ChatResponse, ChatStatus, SessionView

CREATE_ARGS = {
    "title": "Laptop battery drains fast",
    "description": "Battery goes from 100% to 20% in an hour.",
    "category": "hardware",
    "priority": "medium",
}
_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


class CrashInsideWriteError(RuntimeError):
    pass


def _proposal_script(fake_model, make_ai, *after):
    return fake_model(
        responses=[
            make_ai("", ("search_tickets", {"query": "battery"}, "r1")),
            make_ai("No duplicate found.", ("create_ticket", CREATE_ARGS, "w1")),
            *after,
        ]
    )


async def _chat(client, **body) -> ChatResponse:
    response = await client.post("/chat", json={"user_id": "alice", "session_id": "s1", **body})
    assert response.status_code == 200, response.text
    return ChatResponse.model_validate(response.json())


async def _propose(make_client, provider, fake_model, make_ai, *after, **overrides):
    model = _proposal_script(fake_model, make_ai, *after)
    client = await make_client(model, provider, **overrides)
    proposal = await _chat(client, message="My laptop battery dies within an hour")
    assert proposal.status is ChatStatus.AWAITING_APPROVAL
    return client, model, proposal


async def test_health(make_client, provider, fake_model):
    client = await make_client(fake_model(responses=[]), provider)
    response = await client.get("/health")
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"user_id": "al:ice", "session_id": "s1", "message": "hi"}, id="colon-user"),
        pytest.param({"user_id": "", "session_id": "s1", "message": "hi"}, id="empty-user"),
        pytest.param({"user_id": "a" * 65, "session_id": "s1", "message": "hi"}, id="long-user"),
        pytest.param({"user_id": "alice", "session_id": "s 1", "message": "hi"}, id="bad-session"),
        pytest.param({"user_id": "alice", "session_id": "s1"}, id="neither"),
        pytest.param(
            {
                "user_id": "alice",
                "session_id": "s1",
                "message": "yes",
                "decision": {"action_id": "a", "approve": True},
            },
            id="both",
        ),
        pytest.param({"user_id": "alice", "session_id": "s1", "message": "   "}, id="blank"),
        pytest.param({"user_id": "alice", "session_id": "s1", "message": "x" * 4001}, id="long"),
    ],
)
async def test_invalid_requests_return_422(make_client, provider, fake_model, body):
    model = fake_model(responses=[])
    client = await make_client(model, provider)
    response = await client.post("/chat", json=body)
    assert response.status_code == 422
    assert model.calls == []


async def test_invalid_ids_in_session_path_return_422(make_client, provider, fake_model):
    client = await make_client(fake_model(responses=[]), provider)
    assert (await client.get("/sessions/al:ice/s1")).status_code == 422
    assert (await client.get("/sessions/alice/s%201")).status_code == 422


async def test_read_turn_reports_tool_calls_and_trace(
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
    reply = await _chat(client, message="Is there a VPN ticket?")

    assert reply.status is ChatStatus.OK
    assert reply.reply == "INC-0003 is open."
    assert reply.pending_action is None
    (record,) = reply.tool_calls
    assert record.name == "search_tickets"
    assert record.args == {"query": "vpn"}
    assert len(record.result_summary) <= 200
    assert record.error is None
    assert record.latency_ms >= 0

    (request_span,) = [s for s in spans[1].get_finished_spans() if s.name == attrs.SPAN_REQUEST]
    assert reply.trace_id == format_trace_id(request_span.context.trace_id)
    assert request_span.attributes[attrs.USER_ID] == "alice"
    assert request_span.attributes[attrs.SESSION_ID] == "s1"
    assert request_span.attributes[attrs.THREAD_ID] == "alice:s1"
    assert request_span.attributes[attrs.REQUEST_STATUS] == "ok"


async def test_proposal_awaits_approval_without_exposing_the_key(
    make_client, provider, fake_model, make_ai, spans
):
    client, _, proposal = await _propose(make_client, provider, fake_model, make_ai)

    pending = proposal.pending_action
    assert pending is not None
    assert pending.tool == "create_ticket"
    assert pending.args == CREATE_ARGS
    assert pending.summary in proposal.reply
    assert provider.writes() == []
    assert [r.name for r in proposal.tool_calls] == ["search_tickets"]
    raw = (await client.get("/sessions/alice/s1")).text + proposal.model_dump_json()
    assert "idempotency_key" not in raw
    request_span = next(s for s in spans[1].get_finished_spans() if s.name == attrs.SPAN_REQUEST)
    assert request_span.attributes[attrs.PENDING_ACTION_ID] == pending.action_id
    assert request_span.attributes[attrs.REQUEST_STATUS] == "awaiting_approval"


@pytest.mark.parametrize("word", ["yes", " Y ", "approve", "OK", "confirm", "approved"])
async def test_text_approval_executes_one_write(
    *, make_client, provider, fake_model, make_ai, word, spans
):
    client, _, proposal = await _propose(
        make_client, provider, fake_model, make_ai, make_ai("Created INC-0016.")
    )

    result = await _chat(client, message=word)

    assert result.status is ChatStatus.OK
    assert result.pending_action is None
    assert result.reply == "Created INC-0016."
    (write,) = provider.writes()
    assert write.args["reporter_id"] == "alice"
    (record,) = result.tool_calls
    assert record.name == "create_ticket"
    assert record.args == CREATE_ARGS
    approval = next(s for s in spans[1].get_finished_spans() if s.name == attrs.SPAN_APPROVAL)
    assert approval.attributes[attrs.APPROVAL_ACTION_ID] == proposal.pending_action.action_id
    assert approval.attributes[attrs.APPROVAL_DECISION] == "approve"
    assert approval.attributes[attrs.APPROVAL_IDEMPOTENCY_KEY] == write.idempotency_key


@pytest.mark.parametrize("word", ["no", "N", "decline", "cancel", "reject", "declined"])
async def test_text_decline_writes_nothing(
    *, make_client, provider, fake_model, make_ai, word, spans
):
    client, _, _ = await _propose(
        make_client, provider, fake_model, make_ai, make_ai("OK, nothing was created.")
    )

    result = await _chat(client, message=word)

    assert result.status is ChatStatus.OK
    assert result.reply == "OK, nothing was created."
    assert provider.writes() == []
    approval = next(s for s in spans[1].get_finished_spans() if s.name == attrs.SPAN_APPROVAL)
    assert approval.attributes[attrs.APPROVAL_DECISION] == "decline"


@pytest.mark.parametrize("text", ["yes please", "yes.", "what will this do?", "maybe"])
async def test_other_text_while_pending_reasks_without_a_model_call(
    make_client, provider, fake_model, make_ai, text
):
    client, model, proposal = await _propose(make_client, provider, fake_model, make_ai)
    model_calls = len(model.calls)

    result = await _chat(client, message=text)

    assert len(model.calls) == model_calls
    assert result.status is ChatStatus.AWAITING_APPROVAL
    assert result.pending_action == proposal.pending_action
    assert proposal.pending_action.summary in result.reply
    assert result.tool_calls == []
    assert provider.writes() == []


async def test_structured_decision_approves(make_client, provider, fake_model, make_ai):
    client, _, proposal = await _propose(
        make_client, provider, fake_model, make_ai, make_ai("Created.")
    )
    decision = {"action_id": proposal.pending_action.action_id, "approve": True}

    result = await _chat(client, decision=decision)

    assert result.status is ChatStatus.OK
    assert len(provider.writes()) == 1


async def test_stale_action_id_never_resumes(make_client, provider, fake_model, make_ai):
    client, model, proposal = await _propose(make_client, provider, fake_model, make_ai)
    model_calls = len(model.calls)

    result = await _chat(client, decision={"action_id": "not-the-pending-one", "approve": True})

    assert result.reply == STALE_DECISION_REPLY
    assert result.status is ChatStatus.AWAITING_APPROVAL
    assert result.pending_action == proposal.pending_action
    assert provider.writes() == []
    assert len(model.calls) == model_calls


async def test_duplicate_decision_writes_once(make_client, provider, fake_model, make_ai):
    client, _, proposal = await _propose(
        make_client, provider, fake_model, make_ai, make_ai("Created.")
    )
    decision = {"action_id": proposal.pending_action.action_id, "approve": True}

    first = await _chat(client, decision=decision)
    second = await _chat(client, decision=decision)

    assert first.status is ChatStatus.OK
    assert second.reply == STALE_DECISION_REPLY
    assert second.status is ChatStatus.OK
    assert second.pending_action is None
    assert len(provider.writes()) == 1


async def test_two_concurrent_approvals_write_once(make_client, provider, fake_model, make_ai):
    client, _, proposal = await _propose(
        make_client, provider, fake_model, make_ai, make_ai("Created.")
    )
    provider.delay_s = 0.2
    decision = {"action_id": proposal.pending_action.action_id, "approve": True}

    results = await asyncio.gather(
        _chat(client, decision=decision), _chat(client, decision=decision)
    )

    assert len(provider.writes()) == 1
    assert sorted(r.reply for r in results) == sorted(["Created.", STALE_DECISION_REPLY])


async def test_decision_words_without_a_pending_action_go_to_the_model(
    make_client, provider, fake_model, make_ai
):
    model = fake_model(responses=[make_ai("Yes to what?")])
    client = await make_client(model, provider)

    result = await _chat(client, message="yes")

    assert result.reply == "Yes to what?"
    assert len(model.calls) == 1


async def test_decision_without_a_pending_action_is_not_found(make_client, provider, fake_model):
    model = fake_model(responses=[])
    client = await make_client(model, provider)

    result = await _chat(client, decision={"action_id": "a1", "approve": True})

    assert result.reply == STALE_DECISION_REPLY
    assert result.status is ChatStatus.OK
    assert model.calls == []


async def test_recursion_limit_gives_a_clean_error(
    make_client, provider, fake_model, make_ai, spans
):
    model = fake_model(
        responses=[make_ai("", ("search_tickets", {"query": "vpn"}, "loop"))], repeat_last=True
    )
    client = await make_client(model, provider, recursion_limit=5)

    result = await _chat(client, message="loop forever")

    assert result.status is ChatStatus.ERROR
    assert result.reply == turn_errors.RECURSION_REPLY
    assert result.pending_action is None
    assert len(result.trace_id) == 32
    request_span = next(s for s in spans[1].get_finished_spans() if s.name == attrs.SPAN_REQUEST)
    assert request_span.status.status_code is StatusCode.ERROR
    assert request_span.attributes[attrs.REQUEST_STATUS] == "error"


async def test_turn_deadline_gives_a_clean_error(make_client, provider, fake_model, make_ai, spans):
    model = fake_model(responses=[make_ai("too late")], delay_s=2.0)
    client = await make_client(model, provider, turn_deadline_s=0.1)

    result = await _chat(client, message="hello")

    assert result.status is ChatStatus.ERROR
    assert result.reply == turn_errors.DEADLINE_REPLY
    request_span = next(s for s in spans[1].get_finished_spans() if s.name == attrs.SPAN_REQUEST)
    assert request_span.status.status_code is StatusCode.ERROR

    # The thread still works afterwards.
    model.delay_s = 0.0
    model.raises = None
    later = await make_client(fake_model(responses=[make_ai("Back.")]), provider)
    assert (await _chat(later, message="again")).reply == "Back."


def _status_response(code: int) -> httpx2.Response:
    return httpx2.Response(code, request=_REQUEST)


@pytest.mark.parametrize(
    ("error", "reply"),
    [
        pytest.param(
            AnthropicOverloadedError(
                message="Overloaded", response=_status_response(529), body=None
            ),
            turn_errors.OVERLOADED_REPLY,
            id="overloaded",
        ),
        pytest.param(
            AnthropicRateLimitError(
                message="rate_limit", response=_status_response(429), body=None
            ),
            turn_errors.RATE_LIMIT_REPLY,
            id="rate-limit",
        ),
        pytest.param(
            AnthropicAuthenticationError(
                message="bad key", response=_status_response(401), body=None
            ),
            turn_errors.AUTH_REPLY,
            id="auth",
        ),
        pytest.param(
            AnthropicTimeoutError(_REQUEST), turn_errors.MODEL_TIMEOUT_REPLY, id="timeout"
        ),
    ],
)
async def test_anthropic_errors_become_clean_error_replies(
    *, make_client, provider, fake_model, error, reply, spans
):
    model = fake_model(responses=[], raises=error)
    client = await make_client(model, provider)

    result = await _chat(client, message="hello")

    assert result.status is ChatStatus.ERROR
    assert result.reply == reply
    assert "Traceback" not in result.reply
    request_span = next(s for s in spans[1].get_finished_spans() if s.name == attrs.SPAN_REQUEST)
    assert request_span.status.status_code is StatusCode.ERROR
    assert any(event.name == "exception" for event in request_span.events)


async def test_session_view_excludes_the_system_prompt(make_client, provider, fake_model, make_ai):
    client, _, proposal = await _propose(make_client, provider, fake_model, make_ai)

    view = SessionView.model_validate((await client.get("/sessions/alice/s1")).json())

    assert [m.role for m in view.messages] == ["user", "tool", "assistant"]
    assert view.messages[0].content == "My laptop battery dies within an hour"
    assert view.messages[1].tool_name == "search_tickets"
    assert all("IT helpdesk assistant" not in m.content for m in view.messages)
    assert view.pending_action == proposal.pending_action


async def test_unknown_session_returns_no_messages(make_client, provider, fake_model):
    client = await make_client(fake_model(responses=[]), provider)
    response = await client.get("/sessions/nobody/new-session")
    assert response.status_code == 200
    assert response.json() == {"messages": [], "pending_action": None}


async def test_write_error_is_reported_in_tool_calls(make_client, provider, fake_model, make_ai):
    conflict = "ERROR CONFLICT: Key reused with a different body. Next step: Do not retry."
    provider.replies["create_ticket"] = [ToolResult(text=conflict, is_error=True)]
    client, _, _ = await _propose(
        make_client, provider, fake_model, make_ai, make_ai("It was not created.")
    )

    result = await _chat(client, message="yes")

    (record,) = result.tool_calls
    assert record.error == conflict
    assert result.status is ChatStatus.OK


async def test_unfinished_write_is_rerun_with_the_same_key_on_the_next_request(
    make_client, provider, fake_model, make_ai
):
    provider.replies["create_ticket"] = [CrashInsideWriteError("killed mid-write")]
    client, _, proposal = await _propose(
        make_client, provider, fake_model, make_ai, make_ai("Created."), make_ai("Hello.")
    )
    decision = {"action_id": proposal.pending_action.action_id, "approve": True}
    with pytest.raises(CrashInsideWriteError):
        await client.post(
            "/chat", json={"user_id": "alice", "session_id": "s1", "decision": decision}
        )

    result = await _chat(client, message="hello?")

    keys = {w.idempotency_key for w in provider.writes()}
    assert len(provider.writes()) == 2
    assert len(keys) == 1
    assert result.reply == "Hello."
    assert [r.name for r in result.tool_calls] == ["create_ticket"]


async def test_pending_action_present_iff_awaiting_approval(
    make_client, provider, fake_model, make_ai
):
    client, _, proposal = await _propose(
        make_client, provider, fake_model, make_ai, make_ai("Created.")
    )
    done = await _chat(client, message="yes")
    for response in (proposal, done):
        assert (response.pending_action is not None) == (
            response.status is ChatStatus.AWAITING_APPROVAL
        )
    assert json.loads(done.model_dump_json())["pending_action"] is None


async def _hang(_args):
    await asyncio.sleep(30)
    raise AssertionError("the turn deadline should have cancelled this call")


async def test_deadline_during_the_write_does_not_claim_nothing_changed(
    make_client, provider, fake_model, make_ai
):
    # The write is in flight when the deadline fires: the mock may already have saved it.
    provider.replies["create_ticket"] = [_hang]
    client, _, _ = await _propose(
        make_client,
        provider,
        fake_model,
        make_ai,
        make_ai("Created."),
        make_ai("Hello."),
        turn_deadline_s=0.5,
    )

    stopped = await _chat(client, message="yes")

    assert stopped.status is ChatStatus.ERROR
    assert stopped.reply == turn_errors.DEADLINE_WRITE_UNFINISHED_REPLY
    assert "Nothing was changed" not in stopped.reply
    assert stopped.pending_action is None

    later = await _chat(client, message="hello?")
    keys = [w.idempotency_key for w in provider.writes()]
    assert len(keys) == 2
    assert len(set(keys)) == 1
    assert later.reply == "Hello."


async def test_deadline_after_the_write_says_the_change_was_made(
    make_client, provider, fake_model, make_ai
):
    client, model, _ = await _propose(
        make_client, provider, fake_model, make_ai, make_ai("Created."), turn_deadline_s=0.5
    )
    model.delay_s = 30.0  # the follow-up model call after the write never returns in time

    stopped = await _chat(client, message="yes")

    assert stopped.status is ChatStatus.ERROR
    assert stopped.reply == turn_errors.DEADLINE_AFTER_WRITE_REPLY
    assert [r.name for r in stopped.tool_calls] == ["create_ticket"]
    assert len(provider.writes()) == 1


async def test_deadline_before_any_write_says_nothing_changed(
    make_client, provider, fake_model, make_ai
):
    provider.replies["search_tickets"] = [_hang]
    client = await make_client(_proposal_script(fake_model, make_ai), provider, turn_deadline_s=0.3)

    stopped = await _chat(client, message="My laptop battery dies within an hour")

    assert stopped.status is ChatStatus.ERROR
    assert stopped.reply == turn_errors.DEADLINE_REPLY
    assert provider.writes() == []


async def test_a_proposal_cut_off_before_the_interrupt_is_shown_before_any_decision(
    make_client, provider, fake_model, make_ai
):
    # load_context, agent, tools, agent, propose: the limit stops the run before approval.
    model = _proposal_script(fake_model, make_ai, make_ai("Created."))
    client = await make_client(model, provider, recursion_limit=5)
    stopped = await _chat(client, message="My laptop battery dies within an hour")
    assert stopped.status is ChatStatus.ERROR
    assert stopped.pending_action is None
    view = SessionView.model_validate((await client.get("/sessions/alice/s1")).json())
    assert view.pending_action is None

    # "yes" cannot approve an action the user was never shown: show it instead.
    shown = await _chat(client, message="yes")
    assert shown.status is ChatStatus.AWAITING_APPROVAL
    assert shown.pending_action is not None
    assert CREATE_ARGS["title"] in shown.reply
    assert provider.writes() == []

    done = await _chat(client, message="yes")
    assert done.status is ChatStatus.OK
    assert len(provider.writes()) == 1


async def test_a_decision_for_a_cut_off_proposal_is_stale(
    make_client, provider, fake_model, make_ai
):
    model = _proposal_script(fake_model, make_ai, make_ai("Created."))
    client = await make_client(model, provider, recursion_limit=5)
    await _chat(client, message="My laptop battery dies within an hour")

    result = await _chat(client, decision={"action_id": "guessed", "approve": True})

    assert result.reply == STALE_DECISION_REPLY
    assert result.status is ChatStatus.AWAITING_APPROVAL
    assert result.pending_action is not None
    assert provider.writes() == []
