"""Agent HTTP API and SSE event models (CONTRACTS.md §3, §4)."""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

from helpdesk.shared.ids import SessionId, UserId
from helpdesk.shared.tools import ToolName

UserMessage = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
]
TraceId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
LatencyMs = Annotated[int, Field(ge=0)]
ResultSummary = Annotated[str, StringConstraints(max_length=200)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Decision(_Model):
    action_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    approve: bool


class ChatRequest(_Model):
    user_id: UserId
    session_id: SessionId
    message: UserMessage | None = None
    decision: Decision | None = None

    @model_validator(mode="after")
    def _exactly_one_input(self) -> Self:
        if (self.message is None) == (self.decision is None):
            raise ValueError("send exactly one of `message` or `decision`")
        return self


class ChatStatus(StrEnum):
    OK = "ok"
    AWAITING_APPROVAL = "awaiting_approval"
    ERROR = "error"


class ToolCallRecord(_Model):
    name: str
    args: dict[str, JsonValue]
    result_summary: ResultSummary
    # The full `ERROR <CODE>: ...` text when the call failed.
    error: str | None = None
    latency_ms: LatencyMs


class PendingActionView(_Model):
    """What the user sees of a pending write. Never includes the idempotency key."""

    action_id: str
    tool: ToolName
    args: dict[str, JsonValue]
    summary: str


class ChatResponse(_Model):
    reply: str
    status: ChatStatus
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    pending_action: PendingActionView | None = None
    trace_id: TraceId


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class SessionMessage(_Model):
    role: MessageRole
    content: str
    tool_name: str | None = None


class SessionView(_Model):
    messages: list[SessionMessage]
    pending_action: PendingActionView | None = None


class HealthResponse(_Model):
    status: Literal["ok"] = "ok"


# --- SSE events (POST /chat/stream) -----------------------------------------


class SseEvent(StrEnum):
    TOKEN = "token"  # noqa: S105 - an SSE event name
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    STATUS = "status"
    APPROVAL_REQUIRED = "approval_required"
    ERROR = "error"
    DONE = "done"


class TokenData(_Model):
    text: str


class ToolStartData(_Model):
    call_id: str
    name: str
    args: dict[str, JsonValue]


class ToolEndData(_Model):
    call_id: str
    result_summary: ResultSummary
    error: str | None = None
    latency_ms: LatencyMs


class StatusData(_Model):
    message: str


ApprovalRequiredData = PendingActionView


class ErrorData(_Model):
    message: str


class DoneData(_Model):
    trace_id: TraceId
    status: ChatStatus
