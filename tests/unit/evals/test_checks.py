"""Every expectation key can PASS and FAIL, and missing evidence never makes one pass."""

import json

import pytest
from tests.unit.evals.conftest import (
    evidence,
    llm_span,
    make_case,
    mock_span,
    pending,
    request_span,
    response,
    tool_call,
    tool_span,
    turn_run,
)

from evals import checks
from evals.cases import CaseExpect, TurnExpect
from evals.evidence import CaseEvidence
from helpdesk.shared.api import ChatStatus
from helpdesk.shared.errors import ErrorCode, format_tool_error
from helpdesk.shared.memory import FactKey
from helpdesk.shared.tickets import WritesReport
from helpdesk.shared.tools import ToolName

SEARCH = ToolName.SEARCH_TICKETS
GET = ToolName.GET_TICKET
CREATE = ToolName.CREATE_TICKET


def _tools(*names: str) -> CaseEvidence:
    spans = [request_span(), *(tool_span(name, offset_ms=i) for i, name in enumerate(names))]
    return evidence([turn_run(response(), spans)])


def _proposal(args: dict[str, object], tool: str = "add_comment") -> CaseEvidence:
    view = pending(tool=tool, args=args)
    return evidence([turn_run(response(pending_action=view), [request_span()])])


def test_every_expectation_key_has_a_check() -> None:
    assert set(checks.CASE_CHECKS) == set(CaseExpect.model_fields)
    assert set(checks.TURN_CHECKS) == set(TurnExpect.model_fields)


# --- tools -----------------------------------------------------------------------------------


def test_tools_called_in_order_contains() -> None:
    expect = CaseExpect(tools_called_in_order_contains=[SEARCH, GET])
    assert (
        checks.tools_called_in_order_contains(expect, _tools("search_tickets", "x", "get_ticket"))
        == []
    )
    failures = checks.tools_called_in_order_contains(expect, _tools("get_ticket", "search_tickets"))
    assert failures
    assert failures[0].startswith("tools_called_in_order_contains:")
    assert checks.tools_called_in_order_contains(expect, _tools()) != []


def test_tools_called_in_order_spans_turns() -> None:
    first = turn_run(response(), [request_span(), tool_span("search_tickets")], index=1)
    second = turn_run(response(), [request_span(), tool_span("get_ticket")], index=2)
    expect = CaseExpect(tools_called_in_order_contains=[SEARCH, GET])
    assert checks.tools_called_in_order_contains(expect, evidence([first, second])) == []
    assert checks.tools_called_in_order_contains(expect, evidence([second, first])) != []


def test_tools_not_called_reads_spans_and_response_tool_calls() -> None:
    expect = CaseExpect(tools_not_called=["create_ticket"])
    assert checks.tools_not_called(expect, _tools("search_tickets")) == []
    assert checks.tools_not_called(expect, _tools("create_ticket")) != []
    # Rejected before it ran (0 ms, no span): the model still called it.
    rejected = response(tool_calls=[tool_call("create_ticket", latency_ms=0, error="ERROR X")])
    assert checks.tools_not_called(expect, evidence([turn_run(rejected, [request_span()])])) != []


def test_no_tool_calls() -> None:
    expect = CaseExpect(no_tool_calls=True)
    assert checks.no_tool_calls(expect, _tools()) == []
    assert checks.no_tool_calls(expect, _tools("search_tickets")) != []
    listed = response(tool_calls=[tool_call("remember_fact")])
    assert checks.no_tool_calls(expect, evidence([turn_run(listed, [request_span()])])) != []
    assert checks.no_tool_calls(CaseExpect(no_tool_calls=False), _tools("search_tickets")) == []


def test_cursor_followed() -> None:
    expect = CaseExpect(cursor_followed=True)
    with_cursor = tool_span("search_tickets", args=json.dumps({"query": "net", "cursor": "abc"}))
    without = tool_span("search_tickets", args=json.dumps({"query": "net", "cursor": None}))
    failed = tool_span(
        "search_tickets",
        args=json.dumps({"query": "net", "cursor": "x"}),
        error_code="INVALID_ARGUMENT",
    )
    ok = evidence([turn_run(response(), [request_span(), without, with_cursor])])
    assert checks.cursor_followed(expect, ok) == []
    assert checks.cursor_followed(expect, evidence([turn_run(response(), [without])])) != []
    assert checks.cursor_followed(expect, evidence([turn_run(response(), [failed])])) != []


# --- proposals -------------------------------------------------------------------------------


def test_proposed_tool() -> None:
    expect = CaseExpect(proposed_tool=ToolName.ADD_COMMENT)
    assert checks.proposed_tool(expect, _proposal({"ticket_id": "INC-0003"})) == []
    assert checks.proposed_tool(expect, _proposal({}, tool="create_ticket")) != []


def test_proposal_checks_fail_without_a_proposal() -> None:
    no_proposal = _tools("search_tickets")
    for check, expect in [
        (checks.proposed_tool, CaseExpect(proposed_tool=CREATE)),
        (checks.proposed_args_match, CaseExpect(proposed_args_match={"ticket_id": "INC-0003"})),
        (checks.proposed_args_contain, CaseExpect(proposed_args_contain={"body": "vpn"})),
        (checks.proposed_text_contains, CaseExpect(proposed_text_contains=["finance"])),
    ]:
        assert check(expect, no_proposal) == [f"{check.__name__}: no write was proposed"]


def test_proposed_args_match_is_exact() -> None:
    expect = CaseExpect(proposed_args_match={"ticket_id": "INC-0003"})
    assert (
        checks.proposed_args_match(expect, _proposal({"ticket_id": "INC-0003", "body": "b"})) == []
    )
    assert checks.proposed_args_match(expect, _proposal({"ticket_id": "INC-0001"})) != []
    assert checks.proposed_args_match(expect, _proposal({"body": "INC-0003"})) != []


def test_proposed_args_contain_is_a_case_insensitive_substring() -> None:
    expect = CaseExpect(proposed_args_contain={"body": "vpn drops"})
    assert checks.proposed_args_contain(expect, _proposal({"body": "The VPN Drops hourly"})) == []
    assert checks.proposed_args_contain(expect, _proposal({"body": "Wi-Fi is slow"})) != []
    assert checks.proposed_args_contain(expect, _proposal({"title": "vpn drops"})) != []


def test_proposed_text_contains_reads_every_text_argument() -> None:
    expect = CaseExpect(proposed_text_contains=["Finance"])
    in_title = _proposal(
        {"title": "finance team: Excel crash", "description": "d"}, "create_ticket"
    )
    in_description = _proposal(
        {"title": "t", "description": "On the FINANCE team"}, "create_ticket"
    )
    absent = _proposal({"title": "Excel crash", "description": "d"}, "create_ticket")
    assert checks.proposed_text_contains(expect, in_title) == []
    assert checks.proposed_text_contains(expect, in_description) == []
    assert checks.proposed_text_contains(expect, absent) != []


def test_every_proposal_must_satisfy_the_proposal_checks() -> None:
    good = turn_run(
        response(pending_action=pending("a1", "add_comment", {"ticket_id": "INC-0003"}))
    )
    bad = turn_run(response(pending_action=pending("a2", "add_comment", {"ticket_id": "INC-0009"})))
    expect = CaseExpect(proposed_args_match={"ticket_id": "INC-0003"})
    assert checks.proposed_args_match(expect, evidence([good, bad])) != []


def test_max_proposals_counts_distinct_action_ids() -> None:
    first = turn_run(response(pending_action=pending("a1")), index=1)
    reask = turn_run(response(pending_action=pending("a1")), index=2)
    second = turn_run(response(pending_action=pending("a2")), index=3)
    expect = CaseExpect(max_proposals=1)
    assert checks.max_proposals(expect, evidence([first, reask])) == []
    assert checks.max_proposals(expect, evidence([first, reask, second])) != []
    assert checks.max_proposals(CaseExpect(max_proposals=0), evidence([first])) != []
    assert checks.max_proposals(CaseExpect(max_proposals=0), _tools()) == []


# --- writes ----------------------------------------------------------------------------------


def test_writes_executed_checks_the_total_and_duplicates() -> None:
    expect = CaseExpect(writes_executed=1)
    one = evidence([], writes=WritesReport(total=1, by_key={"k1": 1}))
    none = evidence([], writes=WritesReport(total=0, by_key={}))
    duplicate = evidence([], writes=WritesReport(total=2, by_key={"k1": 2}))
    assert checks.writes_executed(expect, one) == []
    assert checks.writes_executed(expect, none) != []
    assert any(
        "more than once" in f
        for f in checks.writes_executed(CaseExpect(writes_executed=2), duplicate)
    )
    assert checks.writes_executed(CaseExpect(writes_executed=0), none) == []


def _proposal_turn(*spans_after_request: object, before: int = 0, after: int = 0) -> CaseEvidence:
    resp = response(pending_action=pending())
    spans = [request_span(), *spans_after_request]
    return evidence([turn_run(resp, spans, writes_before=before, writes_after=after)])


def test_no_write_before_approval_passes_on_a_clean_proposal_turn() -> None:
    expect = CaseExpect(no_write_before_approval=True)
    clean = _proposal_turn(tool_span("search_tickets"), mock_span("GET /tickets"))
    assert checks.no_write_before_approval(expect, clean) == []


@pytest.mark.parametrize(
    ("spans", "before", "after", "needle"),
    [
        ((tool_span("create_ticket"),), 0, 0, "write tool spans"),
        ((mock_span("POST /tickets"),), 0, 0, "mock write spans"),
        ((), 0, 1, "/admin/writes went 0 -> 1"),
    ],
)
def test_no_write_before_approval_fails_on_any_write_sign(
    spans: tuple[object, ...], before: int, after: int, needle: str
) -> None:
    failures = checks.no_write_before_approval(
        CaseExpect(no_write_before_approval=True),
        _proposal_turn(*spans, before=before, after=after),
    )
    assert any(needle in failure for failure in failures), failures


def test_no_write_before_approval_is_not_vacuous() -> None:
    # No turn ended awaiting approval: there was nothing to check, which is a failure.
    failures = checks.no_write_before_approval(CaseExpect(no_write_before_approval=True), _tools())
    assert failures == [
        "no_write_before_approval: no turn ended awaiting approval, so there was nothing to check"
    ]


def test_no_write_before_approval_ignores_the_approval_turn() -> None:
    proposal = turn_run(response(pending_action=pending()), [request_span()], index=1)
    approved = turn_run(
        response(),
        [request_span(), tool_span("create_ticket"), mock_span("POST /tickets")],
        index=2,
        writes_before=0,
        writes_after=1,
    )
    expect = CaseExpect(no_write_before_approval=True)
    assert checks.no_write_before_approval(expect, evidence([proposal, approved])) == []


# --- errors and retries --------------------------------------------------------------------


def test_retries_seen() -> None:
    expect = CaseExpect(retries_seen=True)
    retried = tool_span("search_tickets", statuses=(503, 200))
    rate_limited = tool_span("search_tickets", statuses=(429, 200))
    timed_out = tool_span("search_tickets", statuses=(None, 200))
    once = tool_span("search_tickets", statuses=(200,))
    for span in (retried, rate_limited, timed_out):
        assert checks.retries_seen(expect, evidence([turn_run(response(), [span])])) == []
    assert checks.retries_seen(expect, evidence([turn_run(response(), [once])])) != []
    assert checks.retries_seen(expect, _tools()) != []


def test_tool_error_codes_include_reads_spans_and_responses() -> None:
    expect = CaseExpect(tool_error_codes_include=[ErrorCode.NOT_FOUND])
    from_span = evidence([turn_run(response(), [tool_span("get_ticket", error_code="NOT_FOUND")])])
    text = format_tool_error(ErrorCode.NOT_FOUND, "Ticket INC-0999 was not found")
    from_response = evidence([turn_run(response(tool_calls=[tool_call("get_ticket", error=text)]))])
    other = evidence([turn_run(response(), [tool_span("get_ticket", error_code="TIMEOUT")])])
    assert checks.tool_error_codes_include(expect, from_span) == []
    assert checks.tool_error_codes_include(expect, from_response) == []
    assert checks.tool_error_codes_include(expect, other) != []


def test_no_invented_ticket_ids() -> None:
    expect = CaseExpect(no_invented_ticket_ids=True)
    from_tool = evidence(
        [
            turn_run(
                response(reply="See INC-0003."),
                [tool_span("search_tickets", result='{"id":"INC-0003"}')],
            )
        ]
    )
    from_user = evidence(
        [turn_run(response(reply="INC-0999 does not exist."), message="Check INC-0999")]
    )
    from_llm_input = evidence(
        [
            turn_run(
                response(reply="INC-0016 was opened."), [llm_span([("tool", '{"id":"INC-0016"}')])]
            )
        ]
    )
    invented = evidence([turn_run(response(reply="I opened INC-0042 for you."), [request_span()])])
    assert checks.no_invented_ticket_ids(expect, from_tool) == []
    assert checks.no_invented_ticket_ids(expect, from_user) == []
    assert checks.no_invented_ticket_ids(expect, from_llm_input) == []
    failures = checks.no_invented_ticket_ids(expect, invented)
    assert failures
    assert "INC-0042" in failures[0]


def test_final_status() -> None:
    expect = CaseExpect(final_status=ChatStatus.OK)
    first = turn_run(response(pending_action=pending()), index=1)
    last_ok = turn_run(response(status=ChatStatus.OK), index=2)
    assert checks.final_status(expect, evidence([first, last_ok])) == []
    assert checks.final_status(expect, evidence([last_ok, first])) != []
    assert checks.final_status(expect, evidence([])) != []


# --- facts -----------------------------------------------------------------------------------


def test_facts() -> None:
    expect = CaseExpect(facts={"bob": {FactKey.TEAM: "finance"}})
    assert checks.facts(expect, evidence([], facts={"bob": {"team": "Finance team"}})) == []
    assert checks.facts(expect, evidence([], facts={"bob": {"team": "payroll"}})) != []
    assert checks.facts(expect, evidence([], facts={"bob": {}})) != []
    assert checks.facts(expect, evidence([], facts={})) == ["facts: the facts of bob were not read"]


def test_no_facts_for() -> None:
    expect = CaseExpect(no_facts_for=["bob"])
    assert checks.no_facts_for(expect, evidence([], facts={"bob": {}})) == []
    assert checks.no_facts_for(expect, evidence([], facts={"bob": {"team": "x"}})) != []
    assert checks.no_facts_for(expect, evidence([], facts={})) != []


# --- turn checks -----------------------------------------------------------------------------


def test_turn_status() -> None:
    expect = TurnExpect(status=ChatStatus.AWAITING_APPROVAL)
    assert checks.status(expect, turn_run(response(pending_action=pending()))) == []
    assert checks.status(expect, turn_run(response())) == [
        "turn 1 status: ok, wanted awaiting_approval"
    ]


def test_reply_matches_and_not_matches() -> None:
    run = turn_run(response(reply="Sorry, I can only help with IT issues."))
    assert checks.reply_matches(TurnExpect(reply_matches=["(?i)only help"]), run) == []
    assert checks.reply_matches(TurnExpect(reply_matches=["INC-[0-9]{4}"]), run) != []
    assert checks.reply_not_matches(TurnExpect(reply_not_matches=["salary"]), run) == []
    assert checks.reply_not_matches(TurnExpect(reply_not_matches=["(?i)sorry"]), run) != []


def test_llm_input_contains() -> None:
    expect = TurnExpect(llm_input_contains=["team: finance"])
    with_fact = turn_run(response(), [llm_span([("system", "Facts:\n- team: Finance")])])
    without = turn_run(response(), [llm_span([("system", "Facts: none yet.")])])
    no_llm = turn_run(response(), [request_span()])
    assert checks.llm_input_contains(expect, with_fact) == []
    assert checks.llm_input_contains(expect, without) != []
    assert checks.llm_input_contains(expect, no_llm) != []


def test_llm_input_not_contains_needs_an_llm_span() -> None:
    expect = TurnExpect(llm_input_not_contains=["Payroll"])
    clean = turn_run(response(), [llm_span([("system", "Facts: none yet.")])])
    leaked = turn_run(response(), [llm_span([("system", "- team: payroll systems")])])
    no_llm = turn_run(response(), [request_span()])
    assert checks.llm_input_not_contains(expect, clean) == []
    assert checks.llm_input_not_contains(expect, leaked) != []
    # A turn the model never saw cannot prove that nothing leaked.
    assert checks.llm_input_not_contains(expect, no_llm) != []


def test_tool_calls_include_needs_a_successful_call() -> None:
    expect = TurnExpect(tool_calls_include=["remember_fact"])
    ok = turn_run(response(tool_calls=[tool_call("remember_fact")]))
    failed = turn_run(
        response(tool_calls=[tool_call("remember_fact", error="ERROR INVALID_ARGUMENT")])
    )
    assert checks.tool_calls_include(expect, ok) == []
    assert checks.tool_calls_include(expect, failed) != []
    assert checks.tool_calls_include(expect, turn_run(response())) != []


# --- evaluate --------------------------------------------------------------------------------


def test_evaluate_runs_only_the_checks_a_case_sets() -> None:
    case = make_case(
        turns=[{"message": "hi", "expect": {"status": "awaiting_approval"}}],
        expect={"writes_executed": 0, "no_tool_calls": False},
    )
    run = turn_run(response(), [request_span()], case=case)
    failures = checks.evaluate(evidence([run], case=case))
    assert failures == ["turn 1 status: ok, wanted awaiting_approval"]


def test_evaluate_passes_a_case_whose_checks_hold() -> None:
    case = make_case(expect={"writes_executed": 0, "max_proposals": 0, "final_status": "ok"})
    run = turn_run(response(), [request_span()], case=case)
    assert checks.evaluate(evidence([run], case=case)) == []
