"""Pending approvals survive a restart; a crashed write re-runs with the same key."""

import logging

import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from helpdesk.agent.decision import resume_value
from helpdesk.agent.persistence import CHECKPOINT_TYPES, open_checkpointer
from helpdesk.agent.run_config import build_run_config
from helpdesk.shared.approval import PendingAction, WriteResult
from helpdesk.shared.errors import ErrorCode
from helpdesk.shared.tools import ToolName

CONFIG = build_run_config("alice", "s1", recursion_limit=25)
COMMENT = {"ticket_id": "INC-0003", "body": "Dropped again at 10:00."}


class CrashInsideWriteError(RuntimeError):
    pass


def _script(fake_model, make_ai):
    return fake_model(
        responses=[make_ai("", ("add_comment", COMMENT, "w1")), make_ai("Comment added.")]
    )


async def _propose(make_graph, provider, model):
    graph = await make_graph(model, provider)
    result = await graph.ainvoke(
        {"messages": [HumanMessage("vpn again")], "user_id": "alice"}, CONFIG
    )
    return result["pending_action"]


async def test_restart_resumes_on_a_new_graph_and_writes_once(
    make_graph, provider, fake_model, make_ai
):
    pending = await _propose(make_graph, provider, _script(fake_model, make_ai))

    # A new graph, model and connections on the same SQLite file: a service restart.
    restarted = await make_graph(fake_model(responses=[make_ai("Comment added.")]), provider)
    snapshot = await restarted.aget_state(CONFIG)
    assert snapshot.next == ("approval",)
    assert snapshot.values["pending_action"] == pending
    await restarted.ainvoke(Command(resume=resume_value(pending.action_id, approve=True)), CONFIG)

    writes = provider.writes()
    assert len(writes) == 1
    assert writes[0].idempotency_key == pending.idempotency_key
    assert (await restarted.aget_state(CONFIG)).values["pending_action"] is None


async def test_crash_inside_execute_write_reruns_with_the_same_key(
    make_graph, provider, fake_model, make_ai
):
    provider.replies["add_comment"] = [CrashInsideWriteError("process died mid-call")]
    graph_model = fake_model(
        responses=[make_ai("", ("add_comment", COMMENT, "w1")), make_ai("Done.")]
    )
    pending = await _propose(make_graph, provider, graph_model)
    graph = await make_graph(graph_model, provider)

    with pytest.raises(CrashInsideWriteError):
        await graph.ainvoke(Command(resume=resume_value(pending.action_id, approve=True)), CONFIG)
    assert (await graph.aget_state(CONFIG)).next == ("execute_write",)

    restarted = await make_graph(fake_model(responses=[make_ai("Done.")]), provider)
    await restarted.ainvoke(None, CONFIG)

    keys = [w.idempotency_key for w in provider.writes()]
    assert keys == [pending.idempotency_key, pending.idempotency_key]
    assert (await restarted.aget_state(CONFIG)).values["last_write"].ok


async def test_pending_action_and_write_result_round_trip_as_models(
    make_graph, provider, fake_model, make_ai, caplog
):
    pending = await _propose(make_graph, provider, _script(fake_model, make_ai))
    graph = await make_graph(fake_model(responses=[make_ai("Done.")]), provider)
    await graph.ainvoke(Command(resume=resume_value(pending.action_id, approve=True)), CONFIG)

    with caplog.at_level(logging.WARNING):
        restarted = await make_graph(fake_model(responses=[]), provider)
        values = (await restarted.aget_state(CONFIG)).values
    last_write = values["last_write"]
    assert isinstance(last_write, WriteResult)
    assert last_write.tool is ToolName.ADD_COMMENT
    assert last_write.idempotency_key == pending.idempotency_key
    # Nothing was blocked or let through with an "unregistered type" warning.
    assert "deserializ" not in caplog.text.lower()


async def test_checkpoint_serializer_is_strict_for_unlisted_types(settings):
    async with open_checkpointer(settings.checkpoint_db_path) as saver:
        serde = saver.serde
        pending = PendingAction(
            action_id="a",
            idempotency_key="k",
            tool=ToolName.CREATE_TICKET,
            args={"title": "t"},
            summary="s",
            tool_call_id="c",
        )
        failed = WriteResult(
            action_id="a",
            tool=ToolName.CREATE_TICKET,
            idempotency_key="k",
            ok=False,
            error_code=ErrorCode.TIMEOUT,
        )
        for value in (pending, failed):
            assert serde.loads_typed(serde.dumps_typed(value)) == value

        class NotAllowed(PendingAction):
            pass

        smuggled = NotAllowed(**pending.model_dump())
        restored = serde.loads_typed(serde.dumps_typed(smuggled))
        assert not isinstance(restored, NotAllowed)
    assert {t.__name__ for t in CHECKPOINT_TYPES} == {
        "PendingAction",
        "WriteResult",
        "ToolName",
        "ErrorCode",
    }
