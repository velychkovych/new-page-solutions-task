"""The DeepEval metrics: every expectation key is a metric that wraps its check, the judge
metrics grade with the (scripted) judge, and a run passes, fails or is not exercised.

The checks themselves are tested in test_checks.py; here each metric adapter gets one
passing and one failing run, read back from the test case's metadata as the runner does.
"""

import dataclasses
import json
from typing import Any

import pytest
from tests.unit.evals.conftest import (
    collected,
    deepeval_case,
    evidence,
    llm_span,
    make_case,
    mock_span,
    pending,
    request_span,
    response,
    score,
    tool_call,
    tool_span,
    turn_run,
)

from evals import checks
from evals.cases import CaseExpect, JudgeSpec, TurnExpect, load_cases
from evals.evidence import CaseEvidence
from evals.judges import Judges, ScriptedJudge, scripted_judges
from evals.metrics import (
    CURSOR_NOT_EXERCISED,
    JUDGE_PREAMBLE,
    JUDGE_VERDICT,
    ExpectationMetric,
    expectation_metrics,
    judge_metric,
)
from evals.results import MetricKind, Outcome
from evals.scoring import metric_result
from helpdesk.shared.errors import ErrorCode, format_tool_error
from helpdesk.shared.tickets import WritesReport

# --- one passing and one failing run per expectation key -------------------------------------


def _one_turn(
    resp: Any = None, spans: tuple[Any, ...] = (), *, case_expect: dict[str, Any] | None = None,
    turn_expect: dict[str, Any] | None = None, message: str = "My printer jams.",
    writes: WritesReport | None = None, facts: dict[str, dict[str, str]] | None = None,
    before: int = 0, after: int = 0,
) -> CaseEvidence:  # fmt: skip
    case = make_case(
        turns=[{"message": message, "expect": turn_expect or {}}], expect=case_expect or {}
    )
    run = turn_run(
        resp or response(),
        [request_span(), *spans],
        case=case,
        message=message,
        writes_before=before,
        writes_after=after,
    )
    return evidence([run], case=case, writes=writes, facts=facts)


SEARCH = tool_span("search_tickets", result='{"items":[],"next_cursor":null}')
CURSOR_SEARCH = tool_span("search_tickets", args=json.dumps({"query": "n", "cursor": "abc"}))
PROPOSAL = response(pending_action=pending(tool="create_ticket", args={"title": "Finance: x"}))
NOT_FOUND = format_tool_error(ErrorCode.NOT_FOUND, "no ticket INC-0999", "ask the user")
ONE_WRITE = WritesReport(total=1, by_key={"k": 1})

# key -> (expectation, passing evidence, failing evidence); case keys first, then turn keys.
CASE_ROWS: dict[str, tuple[dict[str, Any], CaseEvidence, CaseEvidence]] = {
    "tools_called_in_order_contains": (
        {"tools_called_in_order_contains": ["search_tickets"]},
        _one_turn(spans=(SEARCH,)),
        _one_turn(),
    ),
    "tools_not_called": (
        {"tools_not_called": ["create_ticket"]},
        _one_turn(spans=(SEARCH,)),
        _one_turn(spans=(tool_span("create_ticket"),)),
    ),
    "no_tool_calls": ({"no_tool_calls": True}, _one_turn(), _one_turn(spans=(SEARCH,))),
    "cursor_followed": (
        {"cursor_followed": True},
        _one_turn(spans=(CURSOR_SEARCH,)),
        _one_turn(spans=(tool_span("search_tickets", result='{"items":[],"next_cursor":"c"}'),)),
    ),
    "proposed_tool": ({"proposed_tool": "create_ticket"}, _one_turn(PROPOSAL), _one_turn()),
    "proposed_args_match": (
        {"proposed_args_match": {"title": "Finance: x"}},
        _one_turn(PROPOSAL),
        _one_turn(response(pending_action=pending(args={"title": "y"}))),
    ),
    "proposed_args_contain": (
        {"proposed_args_contain": {"title": "finance"}},
        _one_turn(PROPOSAL),
        _one_turn(response(pending_action=pending(args={"title": "y"}))),
    ),
    "proposed_text_contains": (
        {"proposed_text_contains": ["finance"]},
        _one_turn(PROPOSAL),
        _one_turn(response(pending_action=pending(args={"title": "y"}))),
    ),
    "max_proposals": ({"max_proposals": 0}, _one_turn(), _one_turn(PROPOSAL)),
    "writes_executed": (
        {"writes_executed": 1},
        _one_turn(writes=ONE_WRITE),
        _one_turn(writes=WritesReport(total=2, by_key={"k": 2})),
    ),
    "no_write_before_approval": (
        {"no_write_before_approval": True},
        _one_turn(PROPOSAL, (SEARCH, mock_span())),
        _one_turn(PROPOSAL, (tool_span("create_ticket"),), after=1),
    ),
    "retries_seen": (
        {"retries_seen": True},
        _one_turn(spans=(tool_span("search_tickets", statuses=(503, 200)),)),
        _one_turn(spans=(SEARCH,)),
    ),
    "tool_error_codes_include": (
        {"tool_error_codes_include": ["NOT_FOUND"]},
        _one_turn(spans=(tool_span("get_ticket", error_code="NOT_FOUND"),)),
        _one_turn(spans=(SEARCH,)),
    ),
    "no_invented_ticket_ids": (
        {"no_invented_ticket_ids": True},
        _one_turn(response(reply="INC-0003 is open."), message="What about INC-0003?"),
        _one_turn(
            response(
                reply="I opened INC-0042.", tool_calls=[tool_call("get_ticket", error=NOT_FOUND)]
            )
        ),
    ),
    "final_status": ({"final_status": "ok"}, _one_turn(), _one_turn(PROPOSAL)),
    "facts": (
        {"facts": {"alice": {"team": "finance"}}},
        _one_turn(facts={"alice": {"team": "Finance"}}),
        _one_turn(facts={"alice": {}}),
    ),
    "no_facts_for": (
        {"no_facts_for": ["bob"]},
        _one_turn(facts={"bob": {}}),
        _one_turn(facts={"bob": {"team": "x"}}),
    ),
}

TURN_ROWS: dict[str, tuple[dict[str, Any], CaseEvidence, CaseEvidence]] = {
    "status": ({"status": "ok"}, _one_turn(), _one_turn(PROPOSAL)),
    "reply_matches": (
        {"reply_matches": "INC-0016"},
        _one_turn(response(reply="Opened INC-0016.")),
        _one_turn(response(reply="Done.")),
    ),
    "reply_not_matches": (
        {"reply_not_matches": "(?i)payroll"},
        _one_turn(response(reply="Nothing on file.")),
        _one_turn(response(reply="You are on Payroll.")),
    ),
    "llm_input_contains": (
        {"llm_input_contains": ["team: finance"]},
        _one_turn(spans=(llm_span([("system", "- team: finance")]),)),
        _one_turn(spans=(llm_span([("system", "none")]),)),
    ),
    "llm_input_not_contains": (
        {"llm_input_not_contains": ["Payroll"]},
        _one_turn(spans=(llm_span([("system", "none")]),)),
        _one_turn(spans=(llm_span([("system", "- team: Payroll")]),)),
    ),
    "tool_calls_include": (
        {"tool_calls_include": ["get_ticket"]},
        _one_turn(response(tool_calls=[tool_call("get_ticket")])),
        _one_turn(response(tool_calls=[tool_call("get_ticket", error=NOT_FOUND)])),
    ),
}


def _with(ev: CaseEvidence, *, case_expect: dict[str, Any] | None = None,
          turn_expect: dict[str, Any] | None = None) -> CaseEvidence:  # fmt: skip
    """`ev` with its case's expectations replaced."""
    turns = [{"message": "My printer jams.", "expect": turn_expect or {}}]
    case = make_case(turns=turns, expect=case_expect or {})
    runs = [dataclasses.replace(run, turn=case.turns[0]) for run in ev.turns]
    return evidence(runs, case=case, writes=ev.writes, facts=ev.facts)


def test_every_expectation_key_has_a_metric_row() -> None:
    assert set(CASE_ROWS) == set(CaseExpect.model_fields) == set(checks.CASE_CHECKS)
    assert set(TURN_ROWS) == set(TurnExpect.model_fields) == set(checks.TURN_CHECKS)


@pytest.mark.parametrize("key", sorted(CASE_ROWS))
def test_case_key_metric_passes_and_fails(key: str) -> None:
    expect, good, bad = CASE_ROWS[key]
    passing = ExpectationMetric(key)
    passing.measure(deepeval_case(_with(good, case_expect=expect)))
    assert (passing.score, passing.success) == (1.0, True), passing.reason
    failing = ExpectationMetric(key)
    failing.measure(deepeval_case(_with(bad, case_expect=expect)))
    assert (failing.score, failing.success) == (0.0, False)
    # The check's own failure text is the reason, and every failure is kept.
    assert failing.failures
    assert failing.reason == "; ".join(failing.failures)
    assert all(f.startswith(f"{key}:") for f in failing.failures), failing.failures


@pytest.mark.parametrize("key", sorted(TURN_ROWS))
def test_turn_key_metric_passes_and_fails(key: str) -> None:
    expect, good, bad = TURN_ROWS[key]
    passing = ExpectationMetric(key, turn=1)
    passing.measure(deepeval_case(_with(good, turn_expect=expect)))
    assert (passing.score, passing.success) == (1.0, True), passing.reason
    failing = ExpectationMetric(key, turn=1)
    failing.measure(deepeval_case(_with(bad, turn_expect=expect)))
    assert (failing.score, failing.success) == (0.0, False)
    assert all(f.startswith(f"turn 1 {key}:") for f in failing.failures), failing.failures


def test_an_unknown_key_has_no_metric() -> None:
    with pytest.raises(ValueError, match="no check"):
        ExpectationMetric("vibes")
    with pytest.raises(ValueError, match="no check"):
        ExpectationMetric("writes_executed", turn=1)


def test_metrics_are_built_for_exactly_the_keys_a_case_sets() -> None:
    case = make_case(
        turns=[{"message": "hi", "expect": {"status": "ok", "reply_matches": "x"}}],
        expect={"writes_executed": 0, "final_status": "ok"},
    )
    names = [metric.__name__ for metric in expectation_metrics(case)]
    assert names == ["turn 1 status", "turn 1 reply_matches", "writes_executed", "final_status"]


def test_every_real_case_key_has_a_metric() -> None:
    for case in load_cases():
        names = {metric.__name__ for metric in expectation_metrics(case)}
        assert set(case.expect.set_keys()) <= names, case.id
        for i, turn in enumerate(case.turns, start=1):
            assert {f"turn {i} {key}" for key in turn.expect.set_keys()} <= names, case.id


def test_a_metric_reads_only_the_test_case_metadata() -> None:
    expect, _, bad = CASE_ROWS["writes_executed"]
    test_case = deepeval_case(_with(bad, case_expect=expect))
    test_case.metadata = None
    with pytest.raises(ValueError, match="no run evidence"):
        ExpectationMetric("writes_executed").measure(test_case)


def test_the_metric_result_is_deepevals_metric_data() -> None:
    expect, _, bad = CASE_ROWS["writes_executed"]
    metric = ExpectationMetric("writes_executed")
    metric.measure(deepeval_case(_with(bad, case_expect=expect)))
    result = metric_result(metric)
    assert result.name == "writes_executed"
    assert result.kind is MetricKind.DETERMINISTIC
    assert result.outcome is Outcome.FAILED
    assert (result.score, result.threshold) == (0.0, 1.0)
    assert result.reason is not None
    assert result.reason.startswith("writes_executed:")
    assert result.evaluation_model == "deterministic"


# --- pagination: pass, fail, not exercised ---------------------------------------------------

PAGE_1_OF_2 = '{"items":[{"id":"INC-0001"}],"next_cursor":"eyJvIjo1fQ"}'
EVERYTHING = '{"items":[{"id":"INC-0001"},{"id":"INC-0015"}],"next_cursor":null}'
NOTHING = '{"items":[],"next_cursor":null}'


def _pagination(*spans: Any, reply: str = "INC-0015 is open.") -> CaseEvidence:
    return _one_turn(
        response(reply=reply),
        spans,
        case_expect={"cursor_followed": True},
        turn_expect={"reply_matches": "INC-0015"},
    )


async def test_following_the_cursor_passes() -> None:
    first = tool_span("search_tickets", args='{"query":"network"}', result=PAGE_1_OF_2)
    second = tool_span(
        "search_tickets", args='{"query":"network","cursor":"eyJvIjo1fQ"}', offset_ms=1
    )
    result = await score(collected(_pagination(first, second)))
    assert result.outcome is Outcome.PASSED, result.reasons


async def test_not_following_an_existing_next_page_fails() -> None:
    first = tool_span("search_tickets", args='{"query":"network"}', result=PAGE_1_OF_2)
    result = await score(collected(_pagination(first)))
    assert result.outcome is Outcome.FAILED
    assert result.reasons == ["cursor_followed: no successful search_tickets call carried a cursor"]


async def test_every_match_on_the_first_page_is_not_exercised() -> None:
    everything = tool_span(
        "search_tickets", args='{"query":"network","page_size":20}', result=EVERYTHING
    )
    result = await score(collected(_pagination(everything)))
    assert result.outcome is Outcome.NOT_EXERCISED
    assert result.reasons == [f"cursor_followed: {CURSOR_NOT_EXERCISED}"]
    by_name = {m.name: m for m in result.metrics}
    assert by_name["cursor_followed"].outcome is Outcome.NOT_EXERCISED
    assert by_name["cursor_followed"].score is None
    # Every other check still applies: the answer must still name the page-2 ticket.
    assert by_name["turn 1 reply_matches"].outcome is Outcome.PASSED


async def test_a_cursor_skipped_on_a_later_search_fails_instead_of_not_exercised() -> None:
    # The narrow first query fits one page; the broader second one returns a cursor that
    # the agent never follows. Pagination was reached, so cursor_followed fails.
    narrow = tool_span("search_tickets", args='{"query":"network ticket"}', result=NOTHING)
    broad = tool_span("search_tickets", args='{"query":"network"}', result=PAGE_1_OF_2, offset_ms=1)
    result = await score(collected(_pagination(narrow, broad, reply="INC-0014 is open.")))
    assert result.outcome is Outcome.FAILED
    by_name = {m.name: m for m in result.metrics}
    assert by_name["cursor_followed"].outcome is Outcome.FAILED
    assert "cursor_followed: no successful search_tickets call carried a cursor" in result.reasons


async def test_a_not_exercised_run_still_fails_its_other_checks() -> None:
    everything = tool_span("search_tickets", result=EVERYTHING)
    result = await score(collected(_pagination(everything, reply="INC-0001 is the newest.")))
    assert result.outcome is Outcome.FAILED
    assert [r.split(":")[0] for r in result.reasons] == [
        "turn 1 reply_matches",
        "cursor_followed",
    ]


@pytest.mark.parametrize(
    "spans",
    [
        (),  # no search at all: the agent never looked
        (tool_span("search_tickets", result=EVERYTHING, error_code="UPSTREAM_UNAVAILABLE"),),
        (tool_span("search_tickets", result=EVERYTHING[:40]),),  # cut short in the span
    ],
)
async def test_not_exercised_needs_proof_from_the_trace(spans: tuple[Any, ...]) -> None:
    result = await score(collected(_pagination(*spans)))
    assert result.outcome is Outcome.FAILED


# --- judge metrics ---------------------------------------------------------------------------

JUDGED = make_case(
    judge=[
        {"name": "refuses", "evaluation_steps": ["Check that it refuses."]},
        {"name": "polite", "tools": True, "evaluation_steps": ["Check that it is polite."]},
    ],
    expect={"writes_executed": 0},
)


def _judged(**overrides: Any) -> CaseEvidence:
    case = JUDGED.model_copy(update=overrides)
    run = turn_run(
        response(reply="Sorry, I can only help with IT.", tool_calls=[tool_call("search_tickets")]),
        [request_span(), tool_span("search_tickets", result='{"items":[],"next_cursor":null}')],
        case=case,
    )
    return evidence([run], case=case)


async def test_judge_metrics_pass_with_the_scripted_verdicts() -> None:
    result = await score(collected(_judged()))
    assert result.outcome is Outcome.PASSED, result.reasons
    judged = [m for m in result.metrics if m.kind is MetricKind.JUDGE]
    assert [m.name for m in judged] == [
        "refuses [Conversational GEval]",
        "polite [Conversational GEval]",
    ]
    assert all(m.evaluation_model == "scripted-judge" for m in judged)
    assert all(m.score == 1.0 for m in judged)


async def test_a_fail_verdict_fails_the_run_on_that_judge_metric() -> None:
    result = await score(collected(_judged(fake_judge={"polite": "fail"})))
    assert result.outcome is Outcome.FAILED
    assert result.reasons == ["judge polite [Conversational GEval]: scripted verdict: fail"]
    outcomes = {m.name: m.outcome for m in result.metrics}
    assert outcomes["refuses [Conversational GEval]"] is Outcome.PASSED
    assert outcomes["writes_executed"] is Outcome.PASSED


async def test_the_judge_sees_the_steps_the_turns_and_only_the_tools_it_asks_for() -> None:
    judges: dict[str, ScriptedJudge] = {}

    def judge_for(case: Any, spec: JudgeSpec) -> ScriptedJudge:
        return judges.setdefault(spec.name, ScriptedJudge(case.fake_verdict(spec.name)))

    base = scripted_judges()
    setup = Judges(judge_for=judge_for, model=base.model, provider="fake", note="")
    await score(collected(_judged()), setup)
    (refuses,) = judges["refuses"].prompts
    (polite,) = judges["polite"].prompts
    for prompt in (refuses, polite):
        assert "Sorry, I can only help with IT." in prompt
        assert "My printer jams." in prompt
        assert JUDGE_PREAMBLE in prompt
        assert JUDGE_VERDICT in prompt
    assert "Check that it refuses." in refuses
    assert "search_tickets" not in refuses
    assert "search_tickets" in polite
    assert '"next_cursor":null' in polite  # the result from the mcp.tool span


async def test_a_judge_error_fails_the_metric_and_says_so() -> None:
    class Broken(ScriptedJudge):
        async def a_generate(self, prompt: str, schema: Any = None) -> Any:
            raise TimeoutError("judge timed out")

    setup = Judges(judge_for=lambda case, spec: Broken("pass"), model="x", provider="x", note="")
    result = await score(collected(_judged()), setup)
    assert result.outcome is Outcome.FAILED
    assert result.reasons[0] == (
        "judge refuses [Conversational GEval]: judge error: TimeoutError: judge timed out"
    )


def test_judge_metrics_are_strict_with_explicit_steps() -> None:
    spec = JUDGED.judge[1]
    metric = judge_metric(spec, ScriptedJudge("pass"))
    assert metric.strict_mode
    assert metric.threshold == 1
    assert metric.evaluation_steps == [JUDGE_PREAMBLE, "Check that it is polite.", JUDGE_VERDICT]
    assert metric.criteria is None


# --- the vacuity guard -----------------------------------------------------------------------


async def test_a_run_without_evidence_fails_before_any_metric() -> None:
    result = await score(collected(None, error="trace incomplete: abc after 15 s (no span)"))
    assert result.outcome is Outcome.FAILED
    assert result.metrics == []
    assert result.reasons == ["trace incomplete: abc after 15 s (no span)"]
    assert result.error == result.reasons[0]


async def test_run_outcome_and_reasons_follow_the_metrics() -> None:
    case = make_case(
        turns=[{"message": "hi", "expect": {"status": "awaiting_approval"}}],
        expect={"writes_executed": 0, "no_tool_calls": False},
    )
    run = turn_run(response(), [request_span()], case=case)
    result = await score(collected(evidence([run], case=case)))
    assert result.outcome is Outcome.FAILED
    assert result.reasons == ["turn 1 status: ok, wanted awaiting_approval"]
    # Turn checks first, then the case's, in CaseExpect's field order.
    assert [m.name for m in result.metrics] == ["turn 1 status", "no_tool_calls", "writes_executed"]
