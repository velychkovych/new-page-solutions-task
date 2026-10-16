"""What one case run reports (the rows behind latest.json and latest.md)."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from evals.traces import TokenUsage
from helpdesk.shared.api import ChatStatus


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Outcome(StrEnum):
    PASSED = "pass"
    FAILED = "fail"
    # The run never reached the situation a check is about (PLAN case 4: no search
    # returned a next page, so there was no cursor to follow). It neither passes nor
    # fails, and the pass rate leaves it out.
    NOT_EXERCISED = "not_exercised"


class MetricKind(StrEnum):
    DETERMINISTIC = "deterministic"
    JUDGE = "judge"


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


class MetricResult(_Model):
    """One DeepEval metric's result for one run (from deepeval's MetricData)."""

    name: str
    kind: MetricKind
    outcome: Outcome
    score: float | None
    threshold: float | None
    reason: str | None
    # The judge model, or "deterministic".
    evaluation_model: str | None
    error: str | None = None
    tokens: TokenUsage = TokenUsage()


class CaseRunResult(_Model):
    case_id: str
    plan_case: list[int]
    run: int
    outcome: Outcome
    # Every failed check's reason (the deterministic checks' own text, then the judges'),
    # then what was not exercised.
    reasons: list[str]
    turns: list[TurnResult]
    duration_s: float
    # The agent's tokens, from the LLM spans of its traces.
    tokens: TokenUsage
    models: list[str]
    providers: list[str]
    metrics: list[MetricResult] = []
    judge_tokens: TokenUsage = TokenUsage()
    # The run ended before the metrics could run (an HTTP error, an incomplete trace): it
    # fails with this reason and no metric result.
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.outcome is Outcome.PASSED
