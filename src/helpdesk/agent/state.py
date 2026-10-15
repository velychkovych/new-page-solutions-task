"""Graph state (CONTRACTS §5.2).

Every field is typed concretely: LangGraph derives the checkpoint deserialisation
allowlist from these annotations, and an `Any` field would come back as a plain dict.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from helpdesk.shared.approval import PendingAction, WriteResult


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    pending_action: PendingAction | None
    last_write: WriteResult | None
