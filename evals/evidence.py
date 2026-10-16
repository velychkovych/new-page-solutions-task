"""What a case run leaves behind for the checks: each turn's response and trace, the mock's
write counts and the stored facts at the end."""

from collections.abc import Iterator
from dataclasses import dataclass

from evals.cases import Case, DecisionTurn, MessageTurn
from evals.traces import TurnTrace
from helpdesk.shared.api import ChatRequest, ChatResponse, PendingActionView
from helpdesk.shared.tickets import WritesReport

type Facts = dict[str, dict[str, str]]


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
