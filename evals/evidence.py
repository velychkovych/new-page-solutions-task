"""What a case run leaves behind for the checks: each turn's response and trace, the mock's
write counts and the stored facts at the end.

`CaseEvidence` is what the checks read. `EvidenceRecord` is the same evidence as plain JSON:
it rides in the DeepEval test case's metadata, and every metric rebuilds `CaseEvidence`
from it, so a metric sees exactly what the test case carries and nothing else.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict

from evals.cases import Case, DecisionTurn, MessageTurn, Turn
from evals.traces import TurnTrace
from helpdesk.shared.api import ChatRequest, ChatResponse, PendingActionView
from helpdesk.shared.tickets import WritesReport
from helpdesk.telemetry.records import SpanRecord

type Facts = dict[str, dict[str, str]]

EVIDENCE_KEY = "evidence"


@dataclass(frozen=True, slots=True)
class TurnRun:
    index: int  # 1-based, as in failure messages
    turn: MessageTurn | DecisionTurn
    user_id: str
    session_id: str
    request: ChatRequest
    response: ChatResponse
    latency_ms: float
    # /admin/writes total just before and just after the turn.
    writes_before: int
    writes_after: int
    trace: TurnTrace


@dataclass(frozen=True, slots=True)
class CaseEvidence:
    case: Case
    turns: list[TurnRun]
    writes: WritesReport
    facts: Facts

    def proposals(self) -> list[PendingActionView]:
        """Distinct pending actions across the case, in the order they were first shown."""
        seen: dict[str, PendingActionView] = {}
        for run in self.turns:
            pending = run.response.pending_action
            if pending is not None and pending.action_id not in seen:
                seen[pending.action_id] = pending
        return list(seen.values())

    def user_messages(self) -> Iterator[str]:
        for run in self.turns:
            if run.request.message is not None:
                yield run.request.message


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TurnRecord(_Record):
    index: int
    turn: Turn
    user_id: str
    session_id: str
    request: ChatRequest
    response: ChatResponse
    latency_ms: float
    writes_before: int
    writes_after: int
    trace_id: str
    spans: list[SpanRecord]


class EvidenceRecord(_Record):
    case: Case
    turns: list[TurnRecord]
    writes: WritesReport
    facts: Facts

    @classmethod
    def of(cls, evidence: CaseEvidence) -> EvidenceRecord:
        return cls(
            case=evidence.case,
            turns=[
                TurnRecord(
                    index=run.index,
                    turn=run.turn,
                    user_id=run.user_id,
                    session_id=run.session_id,
                    request=run.request,
                    response=run.response,
                    latency_ms=run.latency_ms,
                    writes_before=run.writes_before,
                    writes_after=run.writes_after,
                    trace_id=run.trace.trace_id,
                    spans=list(run.trace.spans),
                )
                for run in evidence.turns
            ],
            writes=evidence.writes,
            facts=evidence.facts,
        )

    def evidence(self) -> CaseEvidence:
        return CaseEvidence(
            case=self.case,
            turns=[
                TurnRun(
                    index=turn.index,
                    turn=turn.turn,
                    user_id=turn.user_id,
                    session_id=turn.session_id,
                    request=turn.request,
                    response=turn.response,
                    latency_ms=turn.latency_ms,
                    writes_before=turn.writes_before,
                    writes_after=turn.writes_after,
                    trace=TurnTrace(trace_id=turn.trace_id, spans=tuple(turn.spans)),
                )
                for turn in self.turns
            ],
            writes=self.writes,
            facts=self.facts,
        )


def evidence_metadata(evidence: CaseEvidence) -> dict[str, Any]:
    """The evidence as JSON-safe metadata for a DeepEval test case."""
    return {EVIDENCE_KEY: EvidenceRecord.of(evidence).model_dump(mode="json")}


def evidence_from_metadata(metadata: dict[str, Any] | None) -> CaseEvidence:
    if metadata is None or EVIDENCE_KEY not in metadata:
        raise ValueError("the test case carries no run evidence in its metadata")
    return EvidenceRecord.model_validate(metadata[EVIDENCE_KEY]).evidence()
