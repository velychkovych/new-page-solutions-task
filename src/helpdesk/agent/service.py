"""One chat turn: decisions, re-asks, graph runs, limits and the request span (CONTRACTS §3).

/chat and /chat/stream run the same turn (`ChatService._run`). The only difference is how
the graph is run: /chat with `ainvoke`, the stream with `astream`, whose chunks go to the
turn's `TurnEvents` (CONTRACTS §4). A streamed turn runs in a background task owned by the
service, so a client that disconnects does not stop it.
"""

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from dataclasses import dataclass
from typing import cast

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, StateSnapshot
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode, Tracer, format_trace_id

from helpdesk.agent.decision import resume_value
from helpdesk.agent.final_reply import TurnWrite, final_reply
from helpdesk.agent.graph import AgentGraph
from helpdesk.agent.locks import ThreadLocks
from helpdesk.agent.routing import Node
from helpdesk.agent.run_config import build_run_config
from helpdesk.agent.state import AgentState
from helpdesk.agent.stream import STREAM_MODES, StreamEvent, TurnEvents
from helpdesk.agent.transcript import (
    last_ai_message,
    last_ai_text,
    pending_view,
    proposal_reply,
    session_messages,
    tool_call_records,
)
from helpdesk.agent.turn_errors import (
    DEADLINE_AFTER_WRITE_REPLY,
    DEADLINE_REPLY,
    DEADLINE_WRITE_UNFINISHED_REPLY,
    INTERRUPTED_AFTER_WRITE_REPLY,
    INTERRUPTED_REPLY,
    INTERRUPTED_WRITE_UNFINISHED_REPLY,
    is_interruption,
    turn_error_reply,
)
from helpdesk.shared import trace_attributes as attrs
from helpdesk.shared.api import ChatRequest, ChatResponse, ChatStatus, SessionView
from helpdesk.shared.approval import PendingAction, WriteResult, classify_decision_text
from helpdesk.shared.settings import Settings

logger = logging.getLogger(__name__)

STALE_DECISION_REPLY = "That action was already handled or was not found."


def reask_reply(pending: PendingAction) -> str:
    return (
        f"An action is waiting for your decision: {pending.summary}. "
        'Please reply "yes" to approve it or "no" to decline it.'
    )


def cut_off_proposal_reply(pending: PendingAction) -> str:
    return f"My last reply was cut off before I could ask you about this. {reask_reply(pending)}"


class ShuttingDownError(Exception):
    """The app began shutting down before this turn got its thread lock."""


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


def _last_write(snapshot: StateSnapshot) -> WriteResult | None:
    last_write = _values(snapshot).get("last_write")
    return last_write if isinstance(last_write, WriteResult) else None


def _last_write_id(snapshot: StateSnapshot) -> str | None:
    last_write = _last_write(snapshot)
    return last_write.action_id if last_write is not None else None


def _response(
    outcome: TurnOutcome,
    history: Sequence[BaseMessage],
    after: Sequence[BaseMessage],
    trace_id: str,
) -> ChatResponse:
    before = {m.id for m in history}
    new_ids = {m.id for m in after if m.id is not None and m.id not in before}
    return ChatResponse(
        reply=outcome.reply,
        status=outcome.status,
        tool_calls=tool_call_records(after, new_ids),
        pending_action=pending_view(outcome.pending)
        if outcome.status is ChatStatus.AWAITING_APPROVAL and outcome.pending is not None
        else None,
        trace_id=trace_id,
    )


def turn_write(before: StateSnapshot, after: StateSnapshot) -> TurnWrite:
    last_write = _last_write(after)
    if last_write is None or last_write.action_id == _last_write_id(before):
        return TurnWrite.NONE
    return TurnWrite.SAVED if last_write.ok else TurnWrite.FAILED


def _stopped_reply(
    before: StateSnapshot, after: StateSnapshot, replies: tuple[str, str, str]
) -> str:
    """Never tell the user nothing changed when a write may have happened."""
    nothing_changed, write_unfinished, after_write = replies
    if Node.EXECUTE_WRITE in after.next:
        return write_unfinished
    if _last_write_id(after) != _last_write_id(before):
        return after_write
    return nothing_changed


def deadline_reply(before: StateSnapshot, after: StateSnapshot) -> str:
    return _stopped_reply(
        before,
        after,
        (DEADLINE_REPLY, DEADLINE_WRITE_UNFINISHED_REPLY, DEADLINE_AFTER_WRITE_REPLY),
    )


def interrupted_reply(before: StateSnapshot, after: StateSnapshot) -> str:
    return _stopped_reply(
        before,
        after,
        (INTERRUPTED_REPLY, INTERRUPTED_WRITE_UNFINISHED_REPLY, INTERRUPTED_AFTER_WRITE_REPLY),
    )


def _current_trace_id() -> str:
    return format_trace_id(trace.get_current_span().get_span_context().trace_id)


class ChatService:
    def __init__(
        self, *, graph: AgentGraph, settings: Settings, tracer: Tracer | None = None
    ) -> None:
        self._graph = graph
        self._settings = settings
        self._tracer = tracer if tracer is not None else trace.get_tracer(__name__)
        self._locks = ThreadLocks()
        # Strong references: the event loop keeps only weak ones to tasks.
        self._streamed_turns: set[asyncio.Task[None]] = set()
        # Set once shutdown starts: a turn still waiting for its thread lock then ends
        # without running, so the drain waits for one turn per thread, not for a queue.
        self._closing = False

    def _config(self, user_id: str, session_id: str) -> RunnableConfig:
        return build_run_config(user_id, session_id, recursion_limit=self._settings.recursion_limit)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        return await self._run(request, None)

    def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        """Start the turn in the background and return its events (CONTRACTS §4).

        The caller may stop reading at any time: the turn still runs to its end, saves its
        checkpoint and releases the thread lock.
        """
        events = TurnEvents(_current_trace_id())
        task = asyncio.create_task(self._streamed_turn(request, events))
        self._streamed_turns.add(task)
        task.add_done_callback(self._streamed_turn_ended)
        return aiter(events)

    async def wait_for_streamed_turns(self) -> None:
        """Let running streamed turns finish (app shutdown), so none loses its checkpoint.

        Turns queued behind them end without running. The wait is bounded by one turn
        deadline: a running turn's deadline started before shutdown did, so it has fired
        by then. Whatever is left is cancelled, which is the same as a crash: every node
        that finished is checkpointed, and an unfinished write is re-run with its key.
        """
        self._closing = True
        if not self._streamed_turns:
            return
        _, pending = await asyncio.wait(
            set(self._streamed_turns), timeout=self._settings.turn_deadline_s
        )
        for task in pending:
            task.cancel()
        if pending:
            logger.warning("cancelled %d streamed turns at shutdown", len(pending))
            await asyncio.gather(*pending, return_exceptions=True)

    async def _streamed_turn(self, request: ChatRequest, events: TurnEvents) -> None:
        try:
            await self._run(request, events)
        except Exception as error:
            # A bug: /chat would answer 500. The request span has recorded the exception.
            logger.error("streamed turn failed: %s", type(error).__name__)
        finally:
            events.close()

    def _streamed_turn_ended(self, task: asyncio.Task[None]) -> None:
        self._streamed_turns.discard(task)
        if not task.cancelled() and task.exception() is not None:
            # A BaseException such as SystemExit; reading it here also stops asyncio from
            # reporting it as never retrieved.
            logger.error("streamed turn stopped: %s", type(task.exception()).__name__)

    async def _run(self, request: ChatRequest, events: TurnEvents | None) -> ChatResponse:
        """The one turn behind /chat and /chat/stream; `events` is None for /chat."""
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
            # The lock covers the whole turn, streaming included, so a second approval
            # waits and then sees the action already handled. The deadline starts once
            # the lock is held.
            async with self._locks.hold(thread):
                start = await self._graph.aget_state(config)
                history = _messages(start)
                if events is not None:
                    events.begin(trace_id, history)
                if self._closing:
                    outcome = self._failed(span, ShuttingDownError(), INTERRUPTED_REPLY)
                else:
                    outcome = await self._limited_turn(request, config, span, start, events)
                after = _messages(await self._graph.aget_state(config))
                response = _response(outcome, history, after, trace_id)
                span.set_attribute(attrs.REQUEST_STATUS, outcome.status.value)
                if outcome.pending is not None:
                    span.set_attribute(attrs.PENDING_ACTION_ID, outcome.pending.action_id)
                if events is not None:
                    # Before the lock is released: the next turn on this thread starts
                    # only after this stream's `done` is queued.
                    events.finish(response)
            return response

    async def _run_graph(
        self,
        graph_input: AgentState | Command[str] | None,
        config: RunnableConfig,
        events: TurnEvents | None,
    ) -> None:
        """The only difference between the two endpoints: how the graph is run.

        Both raise the same errors (ainvoke is built on astream), so the turn's limits and
        error replies are shared.
        """
        if events is None:
            await self._graph.ainvoke(graph_input, config)
            return
        # Closed on the way out: if mapping a chunk raises, the graph's node tasks (an
        # approved write among them) are cancelled and awaited here, before the thread
        # lock goes to the next turn. Left open, they would run on beside it.
        # `astream` is an async generator, though it is annotated as an AsyncIterator.
        stream = cast(
            AsyncGenerator[tuple[str, object]],
            self._graph.astream(graph_input, config, stream_mode=STREAM_MODES),
        )
        async with contextlib.aclosing(stream) as chunks:
            async for mode, chunk in chunks:
                events.graph_chunk(mode, chunk)

    async def _limited_turn(
        self,
        request: ChatRequest,
        config: RunnableConfig,
        span: Span,
        start: StateSnapshot,
        events: TurnEvents | None,
    ) -> TurnOutcome:
        try:
            async with asyncio.timeout(self._settings.turn_deadline_s):
                return await self._turn(request, config, start, events)
        except TimeoutError as error:
            stopped = await self._graph.aget_state(config)
            return self._failed(span, error, deadline_reply(start, stopped))
        except Exception as error:
            if is_interruption(error):
                # Like the deadline: the interrupted node did not finish, and a write
                # left unfinished is re-run with the same key on the next request.
                stopped = await self._graph.aget_state(config)
                return self._failed(span, error, interrupted_reply(start, stopped))
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

    async def _turn(
        self,
        request: ChatRequest,
        config: RunnableConfig,
        start: StateSnapshot,
        events: TurnEvents | None,
    ) -> TurnOutcome:
        snapshot = await self._graph.aget_state(config)
        if Node.EXECUTE_WRITE in snapshot.next:
            # A write was approved but did not finish (crash, restart, deadline). Re-run
            # it before anything else: same key, so the mock replays instead of writing twice.
            await self._run_graph(None, config, events)
            snapshot = await self._graph.aget_state(config)
        if proposal_cut_off(snapshot):
            # The last turn stopped (recursion limit, deadline) after propose but before
            # the interrupt, so the user never saw the proposal. Raise the interrupt now;
            # approval has no side effects, so this writes nothing.
            await self._run_graph(None, config, events)
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
            return await self._resume(
                pending,
                approve=request.decision.approve,
                config=config,
                start=start,
                events=events,
            )

        message = request.message or ""
        if pending is not None:
            decision = classify_decision_text(message)
            if decision is None:
                # Deterministic: no model call and no resume while a write is pending.
                return TurnOutcome(reask_reply(pending), ChatStatus.AWAITING_APPROVAL, pending)
            return await self._resume(
                pending, approve=decision, config=config, start=start, events=events
            )

        user_id = str(config.get("configurable", {})["user_id"])
        await self._run_graph(
            {"messages": [HumanMessage(content=message)], "user_id": user_id}, config, events
        )
        return await self._outcome(config, start)

    @staticmethod
    def _stale(pending: PendingAction | None) -> TurnOutcome:
        status = ChatStatus.AWAITING_APPROVAL if pending is not None else ChatStatus.OK
        return TurnOutcome(STALE_DECISION_REPLY, status, pending)

    async def _resume(
        self,
        pending: PendingAction,
        *,
        approve: bool,
        config: RunnableConfig,
        start: StateSnapshot,
        events: TurnEvents | None,
    ) -> TurnOutcome:
        with self._tracer.start_as_current_span(
            attrs.SPAN_APPROVAL,
            attributes={
                attrs.APPROVAL_ACTION_ID: pending.action_id,
                attrs.APPROVAL_DECISION: "approve" if approve else "decline",
                attrs.APPROVAL_IDEMPOTENCY_KEY: pending.idempotency_key,
            },
        ):
            await self._run_graph(
                Command(resume=resume_value(pending.action_id, approve=approve)), config, events
            )
        return await self._outcome(config, start)

    async def _outcome(self, config: RunnableConfig, start: StateSnapshot) -> TurnOutcome:
        snapshot = await self._graph.aget_state(config)
        messages = _messages(snapshot)
        pending = awaiting_approval(snapshot)
        if pending is not None:
            text = last_ai_text(messages)
            return TurnOutcome(
                proposal_reply(text, pending.summary), ChatStatus.AWAITING_APPROVAL, pending
            )
        reply = final_reply(last_ai_message(messages), turn_write(start, snapshot))
        return TurnOutcome(reply, ChatStatus.OK)

    async def session(self, user_id: str, session_id: str) -> SessionView:
        snapshot = await self._graph.aget_state(self._config(user_id, session_id))
        pending = awaiting_approval(snapshot)
        return SessionView(
            messages=session_messages(_messages(snapshot), pending),
            pending_action=pending_view(pending) if pending is not None else None,
        )
