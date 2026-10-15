"""One chat turn: decisions, re-asks, graph runs, limits and the request span (CONTRACTS §3)."""

import asyncio
import logging
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, StateSnapshot
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode, Tracer, format_trace_id

from helpdesk.agent.decision import resume_value
from helpdesk.agent.graph import AgentGraph
from helpdesk.agent.locks import ThreadLocks
from helpdesk.agent.routing import Node
from helpdesk.agent.run_config import build_run_config
from helpdesk.agent.transcript import (
    last_ai_text,
    pending_view,
    session_messages,
    tool_call_records,
)
from helpdesk.agent.turn_errors import (
    DEADLINE_AFTER_WRITE_REPLY,
    DEADLINE_REPLY,
    DEADLINE_WRITE_UNFINISHED_REPLY,
    turn_error_reply,
)
from helpdesk.shared import trace_attributes as attrs
from helpdesk.shared.api import ChatRequest, ChatResponse, ChatStatus, SessionView
from helpdesk.shared.approval import PendingAction, WriteResult, classify_decision_text
from helpdesk.shared.settings import Settings

logger = logging.getLogger(__name__)

STALE_DECISION_REPLY = "That action was already handled or was not found."
EMPTY_REPLY = "Done."


def reask_reply(pending: PendingAction) -> str:
    return (
        f"An action is waiting for your decision: {pending.summary}. "
        'Please reply "yes" to approve it or "no" to decline it.'
    )


def cut_off_proposal_reply(pending: PendingAction) -> str:
    return f"My last reply was cut off before I could ask you about this. {reask_reply(pending)}"


def proposal_reply(ai_text: str, pending: PendingAction) -> str:
    ask = f'I\'d like to: {pending.summary}. Reply "yes" to approve or "no" to decline.'
    return f"{ai_text}\n\n{ask}" if ai_text else ask


@dataclass(frozen=True, slots=True)
class TurnOutcome:
    reply: str
    status: ChatStatus
    pending: PendingAction | None = None


def _messages(snapshot: StateSnapshot) -> list[BaseMessage]:
    values = snapshot.values if isinstance(snapshot.values, dict) else {}
    messages = values.get("messages", [])
    return [m for m in messages if isinstance(m, BaseMessage)]


def _values(snapshot: StateSnapshot) -> dict[str, object]:
    return snapshot.values if isinstance(snapshot.values, dict) else {}


def awaiting_approval(snapshot: StateSnapshot) -> PendingAction | None:
    """The pending action, if the graph is stopped at the approval interrupt.

    Without the interrupt the proposal was never returned to the user (the run stopped
    between propose and approval), so it cannot be approved yet.
    """
    pending = _values(snapshot).get("pending_action")
    if (
        isinstance(pending, PendingAction)
        and Node.APPROVAL in snapshot.next
        and snapshot.interrupts
    ):
        return pending
    return None


def proposal_cut_off(snapshot: StateSnapshot) -> bool:
    return Node.APPROVAL in snapshot.next and not snapshot.interrupts


def _last_write_id(snapshot: StateSnapshot) -> str | None:
    last_write = _values(snapshot).get("last_write")
    return last_write.action_id if isinstance(last_write, WriteResult) else None


def deadline_reply(before: StateSnapshot, after: StateSnapshot) -> str:
    """Never tell the user nothing changed when a write may have happened."""
    if Node.EXECUTE_WRITE in after.next:
        return DEADLINE_WRITE_UNFINISHED_REPLY
    if _last_write_id(after) != _last_write_id(before):
        return DEADLINE_AFTER_WRITE_REPLY
    return DEADLINE_REPLY


class ChatService:
    def __init__(
        self, *, graph: AgentGraph, settings: Settings, tracer: Tracer | None = None
    ) -> None:
        self._graph = graph
        self._settings = settings
        self._tracer = tracer if tracer is not None else trace.get_tracer(__name__)
        self._locks = ThreadLocks()

    def _config(self, user_id: str, session_id: str) -> RunnableConfig:
        return build_run_config(user_id, session_id, recursion_limit=self._settings.recursion_limit)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        config = self._config(request.user_id, request.session_id)
        thread = str(config.get("configurable", {})["thread_id"])
        with self._tracer.start_as_current_span(
            attrs.SPAN_REQUEST,
            attributes={
                attrs.USER_ID: request.user_id,
                attrs.SESSION_ID: request.session_id,
                attrs.THREAD_ID: thread,
            },
        ) as span:
            trace_id = format_trace_id(span.get_span_context().trace_id)
            # The lock covers the whole turn, so a second approval waits and then sees
            # the action already handled. The deadline starts once the lock is held.
            async with self._locks.hold(thread):
                start = await self._graph.aget_state(config)
                before = {m.id for m in _messages(start)}
                outcome = await self._limited_turn(request, config, span, start)
                after = _messages(await self._graph.aget_state(config))
            new_ids = {m.id for m in after if m.id is not None and m.id not in before}
            span.set_attribute(attrs.REQUEST_STATUS, outcome.status.value)
            if outcome.pending is not None:
                span.set_attribute(attrs.PENDING_ACTION_ID, outcome.pending.action_id)
            return ChatResponse(
                reply=outcome.reply,
                status=outcome.status,
                tool_calls=tool_call_records(after, new_ids),
                pending_action=pending_view(outcome.pending)
                if outcome.status is ChatStatus.AWAITING_APPROVAL and outcome.pending is not None
                else None,
                trace_id=trace_id,
            )

    async def _limited_turn(
        self, request: ChatRequest, config: RunnableConfig, span: Span, start: StateSnapshot
    ) -> TurnOutcome:
        try:
            async with asyncio.timeout(self._settings.turn_deadline_s):
                return await self._turn(request, config)
        except TimeoutError as error:
            stopped = await self._graph.aget_state(config)
            return self._failed(span, error, deadline_reply(start, stopped))
        except Exception as error:
            known = turn_error_reply(error)
            if known is None:
                raise
            return self._failed(span, error, known)

    @staticmethod
    def _failed(span: Span, error: BaseException, reply: str) -> TurnOutcome:
        logger.warning("turn failed: %s", type(error).__name__)
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR, type(error).__name__))
        return TurnOutcome(reply=reply, status=ChatStatus.ERROR)

    async def _turn(self, request: ChatRequest, config: RunnableConfig) -> TurnOutcome:
        snapshot = await self._graph.aget_state(config)
        if Node.EXECUTE_WRITE in snapshot.next:
            # A write was approved but did not finish (crash, restart, deadline). Re-run
            # it before anything else: same key, so the mock replays instead of writing twice.
            await self._graph.ainvoke(None, config)
            snapshot = await self._graph.aget_state(config)
        if proposal_cut_off(snapshot):
            # The last turn stopped (recursion limit, deadline) after propose but before
            # the interrupt, so the user never saw the proposal. Raise the interrupt now;
            # approval has no side effects, so this writes nothing.
            await self._graph.ainvoke(None, config)
            snapshot = await self._graph.aget_state(config)
            shown = awaiting_approval(snapshot)
            if shown is not None and request.decision is None:
                # Not even "yes" counts here: it cannot approve an action never shown.
                return TurnOutcome(
                    cut_off_proposal_reply(shown), ChatStatus.AWAITING_APPROVAL, shown
                )
        pending = awaiting_approval(snapshot)

        if request.decision is not None:
            if pending is None or request.decision.action_id != pending.action_id:
                return self._stale(pending)
            return await self._resume(pending, approve=request.decision.approve, config=config)

        message = request.message or ""
        if pending is not None:
            decision = classify_decision_text(message)
            if decision is None:
                # Deterministic: no model call and no resume while a write is pending.
                return TurnOutcome(reask_reply(pending), ChatStatus.AWAITING_APPROVAL, pending)
            return await self._resume(pending, approve=decision, config=config)

        user_id = str(config.get("configurable", {})["user_id"])
        await self._graph.ainvoke(
            {"messages": [HumanMessage(content=message)], "user_id": user_id}, config
        )
        return await self._outcome(config)

    @staticmethod
    def _stale(pending: PendingAction | None) -> TurnOutcome:
        status = ChatStatus.AWAITING_APPROVAL if pending is not None else ChatStatus.OK
        return TurnOutcome(STALE_DECISION_REPLY, status, pending)

    async def _resume(
        self, pending: PendingAction, *, approve: bool, config: RunnableConfig
    ) -> TurnOutcome:
        with self._tracer.start_as_current_span(
            attrs.SPAN_APPROVAL,
            attributes={
                attrs.APPROVAL_ACTION_ID: pending.action_id,
                attrs.APPROVAL_DECISION: "approve" if approve else "decline",
                attrs.APPROVAL_IDEMPOTENCY_KEY: pending.idempotency_key,
            },
        ):
            await self._graph.ainvoke(
                Command(resume=resume_value(pending.action_id, approve=approve)), config
            )
        return await self._outcome(config)

    async def _outcome(self, config: RunnableConfig) -> TurnOutcome:
        snapshot = await self._graph.aget_state(config)
        text = last_ai_text(_messages(snapshot))
        pending = awaiting_approval(snapshot)
        if pending is not None:
            return TurnOutcome(proposal_reply(text, pending), ChatStatus.AWAITING_APPROVAL, pending)
        return TurnOutcome(text or EMPTY_REPLY, ChatStatus.OK)

    async def session(self, user_id: str, session_id: str) -> SessionView:
        snapshot = await self._graph.aget_state(self._config(user_id, session_id))
        pending = awaiting_approval(snapshot)
        return SessionView(
            messages=session_messages(_messages(snapshot)),
            pending_action=pending_view(pending) if pending is not None else None,
        )
