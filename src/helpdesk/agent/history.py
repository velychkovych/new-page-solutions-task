"""What the model sees of the conversation. The checkpoint always keeps everything."""

from collections.abc import Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately

from helpdesk.shared.errors import ErrorCode, format_tool_error

INTERRUPTED_TOOL_CALL = format_tool_error(
    ErrorCode.TIMEOUT,
    "The previous turn ended before this tool call returned a result",
    "Call the tool again only if you still need its result",
)


def split_system(messages: Sequence[BaseMessage]) -> tuple[SystemMessage | None, list[BaseMessage]]:
    system: SystemMessage | None = None
    rest: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            system = message
        else:
            rest.append(message)
    return system, rest


def answer_orphan_tool_calls(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Give every unanswered tool call a result right after its AI message.

    A turn stopped by the deadline or the recursion limit can leave an AI tool call
    without its ToolMessage in the checkpoint, and the Claude API rejects a tool_use
    with no tool_result. The patch only shapes the model input; state is untouched.
    """
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    patched: list[BaseMessage] = []
    for message in messages:
        patched.append(message)
        if isinstance(message, AIMessage):
            patched.extend(
                ToolMessage(
                    content=INTERRUPTED_TOOL_CALL,
                    tool_call_id=call["id"],
                    name=call["name"],
                    status="error",
                )
                for call in message.tool_calls
                if call["id"] is not None and call["id"] not in answered
            )
    return patched


def trim_history(messages: Sequence[BaseMessage], *, token_budget: int) -> list[BaseMessage]:
    """Keep the newest messages within the budget, starting on a user message.

    Keeping a suffix that starts on a HumanMessage never separates an AI tool call
    from its ToolMessages, because those always follow the AI message directly.
    """
    trimmed = trim_messages(
        messages,
        max_tokens=token_budget,
        token_counter=count_tokens_approximately,
        strategy="last",
        start_on="human",
        allow_partial=False,
    )
    if trimmed:
        return trimmed
    # The current turn alone is over budget. Keep it whole: dropping it would leave the
    # model nothing to answer, and cutting inside it could split a tool-call pair.
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return list(messages[index:])
    return list(messages)


def model_messages(messages: Sequence[BaseMessage], *, token_budget: int) -> list[BaseMessage]:
    """System prompt first, then the repaired and trimmed conversation."""
    system, conversation = split_system(messages)
    history = trim_history(answer_orphan_tool_calls(conversation), token_budget=token_budget)
    return [system, *history] if system is not None else history
