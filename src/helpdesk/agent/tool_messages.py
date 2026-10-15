"""ToolMessages written by the graph, and the latency they carry for the API's tool_calls."""

from langchain_core.messages import ToolMessage

from helpdesk.shared.errors import ErrorCode, format_tool_error

# Present only on messages for calls that actually ran (tools node, execute_write);
# the API lists exactly those as `tool_calls`.
LATENCY_KEY = "helpdesk_latency_ms"

TRANSPORT_FAILURE = format_tool_error(
    ErrorCode.UPSTREAM_UNAVAILABLE, "The ticketing tools could not be reached"
)

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


def latency_of(message: ToolMessage) -> int | None:
    value = message.response_metadata.get(LATENCY_KEY)
    return value if isinstance(value, int) else None


def elapsed_ms(started: float, finished: float) -> int:
    return max(0, round((finished - started) * 1000))
