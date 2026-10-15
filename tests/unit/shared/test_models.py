import pytest
from pydantic import ValidationError

from helpdesk.shared.api import ChatRequest, Decision, ToolCallRecord
from helpdesk.shared.memory import RememberFactArgs, facts_namespace
from helpdesk.shared.tickets import ChaosConfig, CreateTicketRequest, TicketSummary
from helpdesk.shared.tools import (
    INJECTED_ARGS,
    READ_TOOLS,
    WRITE_TOOLS,
    AddCommentArgs,
    AddCommentInput,
    CreateTicketArgs,
    CreateTicketInput,
    SearchTicketsArgs,
    ToolName,
)


def test_chat_request_accepts_message_or_decision() -> None:
    ChatRequest(user_id="alice", session_id="s1", message="VPN is down")
    ChatRequest(user_id="alice", session_id="s1", decision=Decision(action_id="a1", approve=True))


def test_chat_request_rejects_both_or_neither() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        ChatRequest(user_id="alice", session_id="s1")
    with pytest.raises(ValidationError, match="exactly one"):
        ChatRequest(
            user_id="alice",
            session_id="s1",
            message="yes",
            decision=Decision(action_id="a1", approve=True),
        )


def test_chat_request_rejects_invalid_user_id() -> None:
    with pytest.raises(ValidationError):
        ChatRequest.model_validate({"user_id": "a:b", "session_id": "s1", "message": "hi"})


def test_tool_partition_is_complete() -> None:
    assert set(ToolName) == READ_TOOLS | WRITE_TOOLS
    assert not READ_TOOLS & WRITE_TOOLS
    assert set(INJECTED_ARGS) == WRITE_TOOLS


@pytest.mark.parametrize(
    ("visible", "full", "tool"),
    [
        (CreateTicketArgs, CreateTicketInput, ToolName.CREATE_TICKET),
        (AddCommentArgs, AddCommentInput, ToolName.ADD_COMMENT),
    ],
)
def test_injected_args_are_exactly_the_hidden_fields(
    visible: type[CreateTicketArgs | AddCommentArgs],
    full: type[CreateTicketInput | AddCommentInput],
    tool: ToolName,
) -> None:
    hidden = set(full.model_json_schema()["properties"]) - set(
        visible.model_json_schema()["properties"]
    )
    assert hidden == set(INJECTED_ARGS[tool])


def test_model_visible_write_args_have_no_identity_or_key() -> None:
    for model in (CreateTicketArgs, AddCommentArgs):
        props = set(model.model_json_schema()["properties"])
        assert not props & {"reporter_id", "author_id", "user_id", "idempotency_key"}


def test_search_page_size_bounds() -> None:
    assert SearchTicketsArgs(query="vpn").page_size == 5
    for bad in (0, 21):
        with pytest.raises(ValidationError):
            SearchTicketsArgs(query="vpn", page_size=bad)


def test_create_ticket_request_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        CreateTicketRequest.model_validate(
            {
                "title": "t",
                "description": "d",
                "category": "network",
                "priority": "low",
                "reporter_id": "alice",
                "status": "closed",
            }
        )


def test_remember_fact_has_no_user_id_and_a_fixed_key_set() -> None:
    assert "user_id" not in RememberFactArgs.model_json_schema()["properties"]
    with pytest.raises(ValidationError):
        RememberFactArgs.model_validate({"key": "salary", "value": "x"})


def test_facts_namespace() -> None:
    assert facts_namespace("alice") == ("users", "alice", "facts")
    with pytest.raises(ValueError, match="invalid user_id"):
        facts_namespace("alice:bob")


def test_chaos_config_rejects_rates_that_do_not_fit_one_draw() -> None:
    base = {"enabled": True, "slow_rate": 0, "slow_ms": 0}
    ChaosConfig.model_validate({**base, "rate_limit_rate": 0.5, "error_rate": 0.5})
    with pytest.raises(ValidationError, match="must be <= 1"):
        ChaosConfig.model_validate({**base, "rate_limit_rate": 0.6, "error_rate": 0.5})


def test_result_summary_is_capped() -> None:
    ToolCallRecord(name="get_ticket", args={}, result_summary="x" * 200, latency_ms=1)
    with pytest.raises(ValidationError):
        ToolCallRecord(name="get_ticket", args={}, result_summary="x" * 201, latency_ms=1)


def test_ticket_title_constraints() -> None:
    with pytest.raises(ValidationError):
        TicketSummary.model_validate(
            {
                "id": "INC-0001",
                "title": "   ",
                "status": "open",
                "priority": "low",
                "category": "network",
                "updated_at": "2026-10-15T12:00:00Z",
            }
        )
