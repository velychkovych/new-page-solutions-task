"""The case files as a DeepEval dataset, and each run as a DeepEval test case.

- A golden (`ConversationalGolden`) per case: scenario, expected outcome, the user turns,
  and the case's setup, checks and judge metrics in `additional_metadata`. The YAML case
  stays the source of truth; the golden is built from it on every run.
- A test case (`ConversationalTestCase`) per run, named `<case_id>#<run>`: the user and
  assistant turns as they happened, every assistant turn with its tool calls (arguments,
  and the result or error the MCP server recorded), and the run's evidence in `metadata`.
"""

from collections import defaultdict, deque
from collections.abc import Sequence

from deepeval.dataset import ConversationalGolden, EvaluationDataset
from deepeval.test_case import ConversationalTestCase, ToolCall
from deepeval.test_case import Turn as ConversationTurn

from evals.cases import Case, DecisionTurn, MessageTurn
from evals.evidence import CaseEvidence, TurnRun, evidence_metadata
from evals.traces import MCP_TOOLS, tool_name, tool_result
from helpdesk.shared.api import PendingActionView
from helpdesk.telemetry.records import SpanRecord

# What a proposal's tool call "returned": nothing ran yet, the user was asked.
PROPOSED_OUTPUT = "not run: proposed to the user, waiting for approval"


def run_name(case_id: str, run: int) -> str:
    return f"{case_id}#{run}"


def user_text(turn: MessageTurn | DecisionTurn) -> str:
    """A turn as the user's side of the conversation; a decision is a button press."""
    if isinstance(turn, MessageTurn):
        return turn.message
    return f"[the user pressed {turn.decision.capitalize()} on the proposed action]"


def golden(case: Case) -> ConversationalGolden:
    return ConversationalGolden(
        name=case.id,
        scenario=case.scenario or case.title,
        expected_outcome=case.expected_outcome or case.title,
        turns=[
            ConversationTurn(role="user", content=user_text(turn), user_id=case.turn_user(turn))
            for turn in case.turns
        ],
        additional_metadata={
            "title": case.title,
            "plan_case": case.plan_case,
            "user_id": case.user_id,
            "setup": case.setup.model_dump(mode="json"),
            "expect": case.expect.model_dump(mode="json", include=set(case.expect.set_keys())),
            "turn_expect": [
                turn.expect.model_dump(mode="json", include=set(turn.expect.set_keys()))
                for turn in case.turns
            ],
            "judge": [spec.model_dump(mode="json") for spec in case.judge],
        },
    )


def build_dataset(cases: Sequence[Case]) -> EvaluationDataset:
    return EvaluationDataset(goldens=[golden(case) for case in cases])


def _span_queues(run: TurnRun) -> dict[str, deque[SpanRecord]]:
    queues: dict[str, deque[SpanRecord]] = defaultdict(deque)
    for span in run.trace.tool_spans:
        queues[tool_name(span)].append(span)
    return queues


def _proposal(pending: PendingActionView) -> ToolCall:
    return ToolCall(name=pending.tool.value, input_parameters=pending.args, output=PROPOSED_OUTPUT)


def tools_called(run: TurnRun) -> list[ToolCall]:
    """The turn's tool calls in order, then its proposal (a write waiting for approval).

    The response keeps only a short result summary, so a call that reached the MCP server
    takes its result from the mcp.tool span (same tool, same order): the judge of PLAN case
    3 has to see the ticket's comments.
    """
    spans = _span_queues(run)
    calls: list[ToolCall] = []
    for call in run.response.tool_calls:
        output = call.error or call.result_summary
        queue = spans[call.name]
        if call.name in MCP_TOOLS and call.latency_ms > 0 and queue:
            output = tool_result(queue.popleft()) or output
        calls.append(ToolCall(name=call.name, input_parameters=call.args, output=output))
    if run.response.pending_action is not None:
        calls.append(_proposal(run.response.pending_action))
    return calls


def conversation(evidence: CaseEvidence) -> list[ConversationTurn]:
    turns: list[ConversationTurn] = []
    for run in evidence.turns:
        turns.append(
            ConversationTurn(role="user", content=user_text(run.turn), user_id=run.user_id)
        )
        turns.append(
            ConversationTurn(
                role="assistant",
                content=run.response.reply,
                tools_called=tools_called(run) or None,
                metadata={
                    "turn": run.index,
                    "status": run.response.status.value,
                    "trace_id": run.response.trace_id,
                },
            )
        )
    return turns


def build_test_case(
    golden: ConversationalGolden, evidence: CaseEvidence, run: int
) -> ConversationalTestCase:
    return ConversationalTestCase(
        name=run_name(evidence.case.id, run),
        scenario=golden.scenario,
        expected_outcome=golden.expected_outcome,
        turns=conversation(evidence),
        metadata=evidence_metadata(evidence),
    )
