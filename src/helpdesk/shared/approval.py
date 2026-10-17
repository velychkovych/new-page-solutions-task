"""Pending writes and the deterministic text-decision rule (CONTRACTS.md §3.2, §5)."""

from pydantic import BaseModel, ConfigDict, JsonValue

from helpdesk.shared.errors import ErrorCode
from helpdesk.shared.ids import IdempotencyKey
from helpdesk.shared.tools import ToolName

APPROVE_WORDS: frozenset[str] = frozenset({"approve", "approved", "yes", "y", "ok", "confirm"})
DECLINE_WORDS: frozenset[str] = frozenset({"decline", "declined", "no", "n", "cancel", "reject"})


def classify_decision_text(message: str) -> bool | None:
    """True = approve, False = decline, None = not a decision (the agent re-asks).

    Deliberately no LLM and no fuzzy matching: "yes please" or "yes." is not an approval.
    """
    normalized = message.strip().lower()
    if normalized in APPROVE_WORDS:
        return True
    if normalized in DECLINE_WORDS:
        return False
    return None


class PendingAction(BaseModel):
    """A proposed write waiting for the user's decision. Created only by `propose`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_id: str
    idempotency_key: IdempotencyKey
    tool: ToolName
    # Model-visible arguments only; injected arguments are added by execute_write.
    args: dict[str, JsonValue]
    summary: str
    # The model's tool call id, so the result can be answered with a matching ToolMessage.
    tool_call_id: str


class WriteResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action_id: str
    tool: ToolName
    idempotency_key: IdempotencyKey
    # The model-visible args, so `propose` can tell a retry of this write from a new one.
    # Defaulted: write results checkpointed before this field existed still load.
    args: dict[str, JsonValue] = {}
    ok: bool
    result: JsonValue = None
    error_code: ErrorCode | None = None
    error_message: str | None = None
