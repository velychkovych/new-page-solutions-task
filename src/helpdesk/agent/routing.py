"""Node names and the routing rules of CONTRACTS §5.1 / §5.3."""

from collections.abc import Sequence
from enum import StrEnum

from langchain_core.messages import AIMessage
from langgraph.graph import END

from helpdesk.agent.model_calls import ModelCall, is_truncated, model_calls
from helpdesk.agent.state import AgentState
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


class ToolCallBatch(StrEnum):
    NONE = "none"
    # No write among the calls: the tools node answers each call, running the valid ones.
    READS = "reads"
    SINGLE_WRITE = "single_write"
    # More than one write, or a write next to any other call (invalid ones included):
    # every call is rejected.
    REJECTED = "rejected"


def classify_tool_calls(calls: Sequence[ModelCall]) -> ToolCallBatch:
    if not calls:
        return ToolCallBatch.NONE
    writes = sum(1 for call in calls if call.name in WRITE_TOOL_NAMES)
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


def proposable(message: AIMessage) -> bool:
    """Exactly one write, sent whole: the only reply that may become a pending action."""
    calls = model_calls(message)
    return (
        classify_tool_calls(calls) is ToolCallBatch.SINGLE_WRITE
        and calls[0].invalid is None
        and not is_truncated(message)
    )


def route_after_agent(state: AgentState) -> str:
    message = last_ai_message(state)
    if message is None or not model_calls(message):
        return END
    if proposable(message):
        return Node.PROPOSE
    # Everything else goes to the tools node, which answers every call exactly once:
    # reads run; rejected, unparseable, cut-off and unknown calls get an error. So the
    # topology stays exactly as in CONTRACTS §5.1 and no call is ever dropped.
    return Node.TOOLS


def route_after_propose(state: AgentState) -> str:
    return Node.APPROVAL if state.get("pending_action") is not None else Node.AGENT
