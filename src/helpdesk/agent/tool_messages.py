"""ToolMessages written by the graph, and the latency they carry for the API's tool_calls."""

from langchain_core.messages import ToolMessage

# Present only on messages for calls that actually ran (tools node, execute_write); a
# call that was rejected before it ran is listed in the API's `tool_calls` with 0 ms.
LATENCY_KEY = "helpdesk_latency_ms"
# Marks record_decline's message: the API does not list a declined write (nothing ran).
DECLINED_KEY = "helpdesk_declined"
# The summary the user approved or declined, on the write's own message, so the session
# view can show the proposal and the decision again: neither is a message of its own.
PROPOSAL_KEY = "helpdesk_proposal"

# CONTRACTS §5.3 fixes this text; it is the one error ToolMessage not in ERROR format.
USER_DECLINED = "User declined. Do not retry this write unless the user asks."


def tool_message(
    *,
    call_id: str,
    name: str,
    text: str,
    is_error: bool,
    latency_ms: int | None = None,
    proposal: str | None = None,
) -> ToolMessage:
    metadata: dict[str, object] = {}
    if latency_ms is not None:
        metadata[LATENCY_KEY] = latency_ms
    if proposal is not None:
        metadata[PROPOSAL_KEY] = proposal
    return ToolMessage(
        content=text,
        tool_call_id=call_id,
        name=name,
        status="error" if is_error else "success",
        response_metadata=metadata,
    )


def declined_message(*, call_id: str, name: str, proposal: str) -> ToolMessage:
    # status="error": the write did not run (CONTRACTS §5.3).
    return ToolMessage(
        content=USER_DECLINED,
        tool_call_id=call_id,
        name=name,
        status="error",
        response_metadata={DECLINED_KEY: True, PROPOSAL_KEY: proposal},
    )


def latency_of(message: ToolMessage) -> int | None:
    value = message.response_metadata.get(LATENCY_KEY)
    return value if isinstance(value, int) else None


def is_declined(message: ToolMessage) -> bool:
    return message.response_metadata.get(DECLINED_KEY) is True


def proposal_of(message: ToolMessage) -> str | None:
    """The summary the user decided on; None for a call that was never proposed."""
    value = message.response_metadata.get(PROPOSAL_KEY)
    return value if isinstance(value, str) else None


def elapsed_ms(started: float, finished: float) -> int:
    return max(0, round((finished - started) * 1000))
