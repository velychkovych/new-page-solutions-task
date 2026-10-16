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

from helpdesk.agent.model_calls import model_calls
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
            # Invalid calls too: langchain-anthropic sends them as tool_use blocks.
            patched.extend(
                ToolMessage(
                    content=INTERRUPTED_TOOL_CALL,
                    tool_call_id=call.id,
                    name=call.name,
                    status="error",
                )
                for call in model_calls(message)
                if call.id not in answered
            )
    return patched


def _is_blank_block(block: str | dict[str, object]) -> bool:
    if isinstance(block, str):
        return not block.strip()
    return block.get("type") == "text" and not str(block.get("text", "")).strip()


def _without_blank_text(message: AIMessage) -> AIMessage | None:
    """The reply without blank text; None when nothing at all is left."""
    content = message.content
    if isinstance(content, str):
        kept: str | list[str | dict[str, object]] = content if content.strip() else ""
    else:
        kept = [block for block in content if not _is_blank_block(block)]
    if not kept and not model_calls(message):
        return None
    return message if kept == content else message.model_copy(update={"content": kept})


def drop_blank_replies(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Leave out what the Claude API would reject in an earlier assistant turn.

    langchain-anthropic 1.7.6 sends string content, and string blocks, as they are, and
    the API rejects whitespace-only text. A reply with no text and no calls carries
    nothing, so it is dropped; one with calls keeps them and loses only the blank text.
    """
    cleaned: list[BaseMessage] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            cleaned.append(message)
        elif (kept := _without_blank_text(message)) is not None:
            cleaned.append(kept)
    return cleaned


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
    repaired = answer_orphan_tool_calls(drop_blank_replies(conversation))
    history = trim_history(repaired, token_budget=token_budget)
    return [system, *history] if system is not None else history
