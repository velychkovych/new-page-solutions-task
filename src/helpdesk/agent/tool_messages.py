"""ToolMessages written by the graph, and the latency they carry for the API's tool_calls."""

from langchain_core.messages import ToolMessage

# Present only on messages for calls that actually ran (tools node, execute_write); a
# call that was rejected before it ran is listed in the API's `tool_calls` with 0 ms.
LATENCY_KEY = "helpdesk_latency_ms"
# Marks record_decline's message: the API does not list a declined write (nothing ran).
DECLINED_KEY = "helpdesk_declined"

# CONTRACTS §5.3 fixes this text; it is the one error ToolMessage not in ERROR format.
USER_DECLINED = "User declined. Do not retry this write unless the user asks."


def tool_message(
    *, call_id: str, name: str, text: str, is_error: bool, latency_ms: int | None = None
) -> ToolMessage:
    return ToolMessage(
        content=text,
        tool_call_id=call_id,
        name=name,
        status="error" if is_error else "success",
        response_metadata={} if latency_ms is None else {LATENCY_KEY: latency_ms},
    )


def declined_message(*, call_id: str, name: str) -> ToolMessage:
    # status="error": the write did not run (CONTRACTS §5.3).
    return ToolMessage(
        content=USER_DECLINED,
        tool_call_id=call_id,
        name=name,
        status="error",
        response_metadata={DECLINED_KEY: True},
    )


def latency_of(message: ToolMessage) -> int | None:
    value = message.response_metadata.get(LATENCY_KEY)
    return value if isinstance(value, int) else None


def is_declined(message: ToolMessage) -> bool:
    return message.response_metadata.get(DECLINED_KEY) is True


def elapsed_ms(started: float, finished: float) -> int:
    return max(0, round((finished - started) * 1000))
