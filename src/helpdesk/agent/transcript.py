"""Reading the checkpointed conversation back out for API responses."""

from collections.abc import Collection, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolCall, ToolMessage
from pydantic import JsonValue

from helpdesk.agent.tool_messages import latency_of
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


def tool_call_records(
    messages: Sequence[BaseMessage], new_message_ids: Collection[str]
) -> list[ToolCallRecord]:
    """The tool calls that actually ran during this request, in order."""
    calls: dict[str, ToolCall] = {
        call["id"]: call
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["id"] is not None
    }
    records: list[ToolCallRecord] = []
    for message in messages:
        if not isinstance(message, ToolMessage) or message.id not in new_message_ids:
            continue
        latency = latency_of(message)
        if latency is None:
            continue
        call = calls.get(message.tool_call_id)
        text = message_text(message)
        is_error = message.status == "error"
        records.append(
            ToolCallRecord(
                name=call["name"] if call is not None else (message.name or ""),
                args=visible_args(call["args"]) if call is not None else {},
                result_summary=summarize_result(text),
                error=text if is_error else None,
                latency_ms=latency,
            )
        )
    return records


def last_ai_text(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message_text(message).strip()
    return ""


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
