"""What one case run reports (the rows behind latest.json and latest.md)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from evals.traces import TokenUsage
from helpdesk.shared.api import ChatStatus


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TurnResult(_Model):
    index: int
    user_id: str
    session: str
    kind: Literal["message", "decision"]
    # The message, or "approve"/"decline".
    text: str
    status: ChatStatus
    latency_ms: float
    trace_id: str
    tokens: TokenUsage = TokenUsage()


class CaseRunResult(_Model):
    case_id: str
    plan_case: list[int]
    run: int
    passed: bool
    reasons: list[str]
    turns: list[TurnResult]
    duration_s: float
    tokens: TokenUsage
    models: list[str]
    providers: list[str]
