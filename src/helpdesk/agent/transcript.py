"""Reading the checkpointed conversation back out for API responses."""

from collections.abc import Collection, Iterable, Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import JsonValue

from helpdesk.agent.model_calls import ModelCall, model_calls
from helpdesk.agent.tool_messages import is_declined, latency_of
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
    """
    calls = call_index(messages)
    records: list[ToolCallRecord] = []
    for message in messages:
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


def session_messages(messages: Sequence[BaseMessage]) -> list[SessionMessage]:
    """User, assistant and tool messages in order; never the system prompt."""
    view: list[SessionMessage] = []
    for message in messages:
        match message:
            case HumanMessage():
                view.append(SessionMessage(role=MessageRole.USER, content=message_text(message)))
            case AIMessage():
                text = message_text(message)
                if text.strip():
                    view.append(SessionMessage(role=MessageRole.ASSISTANT, content=text))
            case ToolMessage():
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
