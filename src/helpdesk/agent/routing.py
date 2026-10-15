"""Node names and the routing rules of CONTRACTS §5.1 / §5.3."""

from collections.abc import Sequence
from enum import StrEnum

from langchain_core.messages import AIMessage, ToolCall
from langgraph.graph import END

from helpdesk.agent.state import AgentState
from helpdesk.shared.errors import ErrorCode, format_tool_error
from helpdesk.shared.memory import REMEMBER_FACT_TOOL
from helpdesk.shared.tools import READ_TOOLS, WRITE_TOOLS


class Node(StrEnum):
    LOAD_CONTEXT = "load_context"
    AGENT = "agent"
    TOOLS = "tools"
    PROPOSE = "propose"
    APPROVAL = "approval"
    EXECUTE_WRITE = "execute_write"
    RECORD_DECLINE = "record_decline"


READ_TOOL_NAMES: frozenset[str] = frozenset(t.value for t in READ_TOOLS)
WRITE_TOOL_NAMES: frozenset[str] = frozenset(t.value for t in WRITE_TOOLS)
LOCAL_TOOL_NAMES: frozenset[str] = frozenset({REMEMBER_FACT_TOOL})

ONE_CHANGE_AT_A_TIME = format_tool_error(
    ErrorCode.INVALID_ARGUMENT,
    "propose one change at a time, after your reads",
    "Call the read tools first, then propose a single write in a message of its own",
)


def unknown_tool_error(name: str) -> str:
    return format_tool_error(
        ErrorCode.INVALID_ARGUMENT,
        f"There is no tool named {name!r}",
        "Use only the tools you were given",
    )


class ToolCallBatch(StrEnum):
    NONE = "none"
    # Reads, remember_fact and unknown names: the tools node answers each call.
    READS = "reads"
    SINGLE_WRITE = "single_write"
    # More than one write, or a write next to any other call: every call is rejected.
    REJECTED = "rejected"


def classify_tool_calls(calls: Sequence[ToolCall]) -> ToolCallBatch:
    if not calls:
        return ToolCallBatch.NONE
    writes = sum(1 for call in calls if call["name"] in WRITE_TOOL_NAMES)
    if writes == 0:
        return ToolCallBatch.READS
    if writes == 1 and len(calls) == 1:
        return ToolCallBatch.SINGLE_WRITE
    return ToolCallBatch.REJECTED


def last_ai_message(state: AgentState) -> AIMessage | None:
    messages = state.get("messages", [])
    if messages and isinstance(messages[-1], AIMessage):
        return messages[-1]
    return None


def route_after_agent(state: AgentState) -> str:
    message = last_ai_message(state)
    batch = classify_tool_calls(message.tool_calls if message is not None else [])
    match batch:
        case ToolCallBatch.NONE:
            return END
        case ToolCallBatch.SINGLE_WRITE:
            return Node.PROPOSE
        case ToolCallBatch.READS | ToolCallBatch.REJECTED:
            # The tools node answers a rejected batch with one error per call, so the
            # topology stays exactly as in CONTRACTS §5.1 and no call is ever dropped.
            return Node.TOOLS


def route_after_propose(state: AgentState) -> str:
    return Node.APPROVAL if state.get("pending_action") is not None else Node.AGENT
