"""Reading the checkpointed conversation back out for API responses."""

from collections.abc import Collection, Iterable, Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import JsonValue

from helpdesk.agent.model_calls import ModelCall, model_calls
from helpdesk.agent.tool_messages import is_declined, latency_of, proposal_of
from helpdesk.shared.api import (
    MessageRole,
    PendingActionView,
    SessionMessage,
    ToolCallRecord,
)
from helpdesk.shared.approval import PendingAction
from helpdesk.shared.tools import INJECTED_ARGS

RESULT_SUMMARY_MAX_CHARS = 200
_INJECTED_NAMES: frozenset[str] = frozenset(
    name for args in INJECTED_ARGS.values() for name in args
)


def summarize_result(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) <= RESULT_SUMMARY_MAX_CHARS:
        return flat
    return flat[: RESULT_SUMMARY_MAX_CHARS - 1] + "…"


def visible_args(args: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Model-visible arguments only; injected ones never leave the agent."""
    return {name: value for name, value in args.items() if name not in _INJECTED_NAMES}


def message_text(message: BaseMessage) -> str:
    return str(message.text)


def call_index(messages: Iterable[BaseMessage]) -> dict[str, ModelCall]:
    """Every tool call the model made in `messages`, by call id (later replies win)."""
    return {
        call.id: call
        for message in messages
        if isinstance(message, AIMessage)
        for call in model_calls(message)
    }


def call_view(
    call_id: str, calls: Mapping[str, ModelCall], fallback_name: str
) -> tuple[str, dict[str, JsonValue]]:
    """The name and model-visible args shown for a call, by /chat and the stream alike."""
    call = calls.get(call_id)
    if call is None:
        return fallback_name, {}
    return call.name, visible_args(call.args)


def tool_call_record(message: ToolMessage, calls: Mapping[str, ModelCall]) -> ToolCallRecord | None:
    """How one answered tool call is reported; None for a declined write (nothing ran)."""
    if is_declined(message):
        return None
    name, args = call_view(message.tool_call_id, calls, message.name or "")
    text = message_text(message)
    return ToolCallRecord(
        name=name,
        args=args,
        result_summary=summarize_result(text),
        error=text if message.status == "error" else None,
        latency_ms=latency_of(message) or 0,
    )


def tool_call_records(
    messages: Sequence[BaseMessage], new_message_ids: Collection[str]
) -> list[ToolCallRecord]:
    """Every tool call answered during this request, in order (CONTRACTS §3.1).

    Rejected and unparseable calls are included with their error and 0 ms, since they
    never ran. A declined write is not: nothing ran, and nothing failed.

    Calls are indexed reply by reply, as the stream does: a result belongs to the call
    that was current when it ran, even if a later reply reuses the call id.
    """
    calls: dict[str, ModelCall] = {}
    records: list[ToolCallRecord] = []
    for message in messages:
        if isinstance(message, AIMessage):
            calls.update((call.id, call) for call in model_calls(message))
            continue
        if not isinstance(message, ToolMessage) or message.id not in new_message_ids:
            continue
        record = tool_call_record(message, calls)
        if record is not None:
            records.append(record)
    return records


def last_ai_message(messages: Sequence[BaseMessage]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def last_ai_text(messages: Sequence[BaseMessage]) -> str:
    message = last_ai_message(messages)
    return message_text(message).strip() if message is not None else ""


def proposal_reply(ai_text: str, summary: str) -> str:
    ask = f'I\'d like to: {summary}. Reply "yes" to approve or "no" to decline.'
    return f"{ai_text}\n\n{ask}" if ai_text else ask


def decision_text(summary: str, *, approve: bool) -> str:
    # The same words the page shows for an Approve or Decline click.
    return f"{'Approve' if approve else 'Decline'}: {summary}"


def _proposed_in(
    messages: Sequence[BaseMessage], index: int, pending: PendingAction | None
) -> str | None:
    """The summary the user was asked about after the AI message at `index`, if any.

    Its tool results follow it directly; the write the user decided on carries the
    summary. A write still waiting for the decision has no result yet.
    """
    message = messages[index]
    if not isinstance(message, AIMessage):
        return None
    call_ids = {call.id for call in model_calls(message)}
    for later in messages[index + 1 :]:
        if not isinstance(later, ToolMessage):
            break
        summary = proposal_of(later)
        if summary is not None and later.tool_call_id in call_ids:
            return summary
    if pending is not None and pending.tool_call_id in call_ids and message is messages[-1]:
        return pending.summary
    return None


def session_messages(
    messages: Sequence[BaseMessage], pending: PendingAction | None = None
) -> list[SessionMessage]:
    """User, assistant and tool messages in order; never the system prompt.

    The approval request and the user's decision are not messages in the checkpoint
    (the API adds the one to the reply and resumes the graph with the other), so they
    are rebuilt here from the proposal the write's result carries, as the user saw them.
    """
    view: list[SessionMessage] = []
    for index, message in enumerate(messages):
        match message:
            case HumanMessage():
                view.append(SessionMessage(role=MessageRole.USER, content=message_text(message)))
            case AIMessage():
                text = message_text(message).strip()
                proposed = _proposed_in(messages, index, pending)
                if proposed is not None:
                    text = proposal_reply(text, proposed)
                if text:
                    view.append(SessionMessage(role=MessageRole.ASSISTANT, content=text))
            case ToolMessage():
                summary = proposal_of(message)
                if summary is not None:
                    approved = not is_declined(message)
                    content = decision_text(summary, approve=approved)
                    view.append(SessionMessage(role=MessageRole.USER, content=content))
                view.append(
                    SessionMessage(
                        role=MessageRole.TOOL,
                        content=message_text(message),
                        tool_name=message.name,
                    )
                )
    return view


def pending_view(pending: PendingAction) -> PendingActionView:
    """What the user sees of a pending write: never the idempotency key."""
    return PendingActionView(
        action_id=pending.action_id,
        tool=pending.tool,
        args=visible_args(pending.args),
        summary=pending.summary,
    )
