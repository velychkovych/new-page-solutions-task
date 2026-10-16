"""The case files as DeepEval goldens, and a run as a DeepEval test case."""

import json

from tests.unit.evals.conftest import (
    deepeval_case,
    evidence,
    make_case,
    pending,
    request_span,
    response,
    tool_call,
    tool_span,
    turn_run,
)

from evals.cases import load_cases
from evals.dataset import PROPOSED_OUTPUT, build_dataset, golden, user_text
from evals.evidence import EVIDENCE_KEY, evidence_from_metadata
from helpdesk.shared.tickets import WritesReport


def test_every_case_is_a_conversational_golden() -> None:
    cases = load_cases()
    dataset = build_dataset(cases)
    assert [g.name for g in dataset.goldens] == [case.id for case in cases]
    status = next(g for g in dataset.goldens if g.name == "status_question")
    assert status.scenario is not None
    assert "INC-0007" in status.scenario
    assert status.expected_outcome
    assert status.turns is not None
    assert [t.content for t in status.turns] == [
        "What's happening with INC-0007? Any news on my laptop battery replacement?"
    ]
    meta = status.additional_metadata
    assert meta is not None
    assert meta["plan_case"] == [3]
    assert meta["expect"]["writes_executed"] == 0
    assert meta["turn_expect"][0]["tool_calls_include"] == ["get_ticket"]
    assert meta["judge"][0]["name"] == "answers_from_the_ticket"


def test_the_golden_falls_back_to_the_title() -> None:
    case = make_case()
    assert golden(case).scenario == case.title
    assert golden(case).expected_outcome == case.title


def test_a_decision_is_shown_as_a_button_press() -> None:
    case = make_case(
        turns=[{"message": "My printer jams."}, {"decision": "decline"}],
        fake=[[{"content": "x"}]],
    )
    assert user_text(case.turns[1]) == "[the user pressed Decline on the proposed action]"
    assert user_text(case.turns[0]) == "My printer jams."


def test_a_run_becomes_a_test_case_with_its_turns_tools_and_evidence() -> None:
    case = make_case(id="unit_case")
    ticket = json.dumps({"id": "INC-0007", "comments": [{"body": "Battery ordered"}]})
    first = turn_run(
        response(
            reply="I'd like to open a ticket.",
            tool_calls=[tool_call("get_ticket", result="short summary")],
            pending_action=pending(args={"title": "Printer jam"}),
        ),
        [request_span(), tool_span("get_ticket", result=ticket)],
        case=case,
    )
    ev = evidence([first], case=case, writes=WritesReport(total=0, by_key={}))
    test_case = deepeval_case(ev, run=2)
    assert test_case.name == "unit_case#2"
    assert test_case.scenario == case.title
    user, assistant = test_case.turns
    assert (user.role, user.content, user.user_id) == ("user", "My printer jams.", "alice")
    assert (assistant.role, assistant.content) == ("assistant", "I'd like to open a ticket.")
    assert assistant.tools_called is not None
    read, proposal = assistant.tools_called
    # The full result from the mcp.tool span, not the response's 200-character summary.
    assert (read.name, read.output) == ("get_ticket", ticket)
    assert (proposal.name, proposal.input_parameters) == ("create_ticket", {"title": "Printer jam"})
    assert proposal.output == PROPOSED_OUTPUT
    assert assistant.metadata == {
        "turn": 1,
        "status": "awaiting_approval",
        "trace_id": first.response.trace_id,
    }


def test_the_evidence_survives_the_test_case_metadata_as_json() -> None:
    case = make_case()
    run = turn_run(response(reply="Done."), [request_span(), tool_span("search_tickets")])
    ev = evidence([run], case=case, facts={"alice": {"team": "finance"}})
    test_case = deepeval_case(ev)
    assert test_case.metadata is not None
    # JSON-safe: what the metrics read is exactly what a JSON round trip gives back.
    metadata = json.loads(json.dumps(test_case.metadata))
    assert set(metadata) == {EVIDENCE_KEY}
    back = evidence_from_metadata(metadata)
    assert back.case == ev.case
    assert back.facts == ev.facts
    assert back.writes == ev.writes
    (turn,) = back.turns
    assert turn.response == run.response
    assert turn.trace.spans == run.trace.spans
    assert turn.turn == run.turn


def test_a_failed_call_shows_its_error_to_the_judge() -> None:
    error = "ERROR NOT_FOUND: no ticket INC-0999. Next step: ask the user."
    run = turn_run(
        response(tool_calls=[tool_call("get_ticket", error=error, latency_ms=0)]),
        [request_span()],
    )
    (assistant,) = deepeval_case(evidence([run])).turns[1:]
    assert assistant.tools_called is not None
    assert assistant.tools_called[0].output == error
