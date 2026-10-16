"""The expectation vocabulary: one function per key of CaseExpect and TurnExpect.

Each returns the reasons it failed (empty when it holds), prefixed with its key. A check
whose evidence is missing fails instead of passing: "no write span" proves nothing when no
turn ever reached a proposal, and "the LLM input does not contain X" proves nothing
without an LLM span. evals.metrics runs each key as a DeepEval metric.
"""

import json
import re
from collections.abc import Callable, Iterable

from pydantic import JsonValue

from evals.cases import CaseExpect, TurnExpect
from evals.evidence import CaseEvidence, TurnRun
from evals.traces import (
    attempt_status,
    llm_input_text,
    llm_tool_results,
    retry_events,
    tool_args,
    tool_attempts,
    tool_error_code,
    tool_name,
    tool_result,
)
from helpdesk.shared.api import ChatStatus
from helpdesk.shared.errors import ErrorCode, parse_tool_error
from helpdesk.shared.tools import WRITE_TOOLS, ToolName

type CaseCheck = Callable[[CaseExpect, CaseEvidence], list[str]]
type TurnCheck = Callable[[TurnExpect, TurnRun], list[str]]

TICKET_ID_RE = re.compile(r"\bINC-[0-9]{4}\b")
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
_WRITE_NAMES = frozenset(tool.value for tool in WRITE_TOOLS)
_MOCK_WRITE_PREFIX = "POST "


def _fail(key: str, reason: str) -> list[str]:
    return [f"{key}: {reason}"]


def _called_mcp_tools(evidence: CaseEvidence) -> list[str]:
    """mcp.tool span names across the case: turn by turn, each turn by start time."""
    return [tool_name(span) for run in evidence.turns for span in run.trace.tool_spans]


def _is_subsequence(wanted: Iterable[str], actual: Iterable[str]) -> bool:
    remaining = iter(actual)
    return all(any(name == got for got in remaining) for name in wanted)


# --- case checks -----------------------------------------------------------------------------


def tools_called_in_order_contains(expect: CaseExpect, evidence: CaseEvidence) -> list[str]:
    wanted = [tool.value for tool in expect.tools_called_in_order_contains]
    actual = _called_mcp_tools(evidence)
    if _is_subsequence(wanted, actual):
        return []
    return _fail("tools_called_in_order_contains", f"wanted {wanted} in order, called {actual}")


def tools_not_called(expect: CaseExpect, evidence: CaseEvidence) -> list[str]:
    called = set(_called_mcp_tools(evidence)) | {
        call.name for run in evidence.turns for call in run.response.tool_calls
    }
    if forbidden := sorted(called & set(expect.tools_not_called)):
        return _fail("tools_not_called", f"called {forbidden}")
    return []


def no_tool_calls(expect: CaseExpect, evidence: CaseEvidence) -> list[str]:
    if not expect.no_tool_calls:
        return []
    listed = [call.name for run in evidence.turns for call in run.response.tool_calls]
    spans = _called_mcp_tools(evidence)
    if listed or spans:
        return _fail("no_tool_calls", f"response tool_calls {listed}, mcp.tool spans {spans}")
    return []


def some_search_had_more_pages(evidence: CaseEvidence) -> bool | None:
    """Whether any successful search_tickets result had a next page.

    Every search counts, not only the first: a narrow first query can hold one page while a
    broader later one returns a cursor, and skipping that cursor is the failure. None when
    no result had a cursor but that is not proven: no successful search, or a result cut
    short in the span (the MCP server truncates long results). So a run is only reported
    "not exercised" when the trace proves no search ever returned a cursor.
    """
    proven = True
    searched = False
    for run in evidence.turns:
        for span in run.trace.tool_spans:
            if tool_name(span) != ToolName.SEARCH_TICKETS or tool_error_code(span) is not None:
                continue
            searched = True
            try:
                result = json.loads(tool_result(span))
            except json.JSONDecodeError:
                proven = False
                continue
            if not isinstance(result, dict) or "next_cursor" not in result:
                proven = False
                continue
            if result["next_cursor"] is not None:
                return True
    return False if searched and proven else None


def cursor_followed(expect: CaseExpect, evidence: CaseEvidence) -> list[str]:
    if not expect.cursor_followed:
        return []
    for run in evidence.turns:
        for span in run.trace.tool_spans:
            if (
                tool_name(span) == ToolName.SEARCH_TICKETS
                and tool_args(span).get("cursor")
                and tool_error_code(span) is None
            ):
                return []
    return _fail("cursor_followed", "no successful search_tickets call carried a cursor")


def _proposal_check(
    key: str, evidence: CaseEvidence, problem: Callable[[str, dict[str, JsonValue]], str | None]
) -> list[str]:
    proposals = evidence.proposals()
    if not proposals:
        return _fail(key, "no write was proposed")
    return [
        f"{key}: {reason}"
        for pending in proposals
        if (reason := problem(pending.tool.value, pending.args)) is not None
    ]


def proposed_tool(expect: CaseExpect, evidence: CaseEvidence) -> list[str]:
    wanted = expect.proposed_tool
    if wanted is None:
        return []
    return _proposal_check(
        "proposed_tool",
        evidence,
        lambda tool, _: None if tool == wanted else f"proposed {tool}, wanted {wanted.value}",
    )


def proposed_args_match(expect: CaseExpect, evidence: CaseEvidence) -> list[str]:
    wanted = expect.proposed_args_match
    if not wanted:
        return []

    def problem(_: str, args: dict[str, JsonValue]) -> str | None:
        wrong = {key: args.get(key) for key, value in wanted.items() if args.get(key) != value}
        return f"got {wrong}, wanted {wanted}" if wrong else None

    return _proposal_check("proposed_args_match", evidence, problem)


def proposed_args_contain(expect: CaseExpect, evidence: CaseEvidence) -> list[str]:
    wanted = expect.proposed_args_contain
    if not wanted:
        return []

    def problem(_: str, args: dict[str, JsonValue]) -> str | None:
        missing = [
            f"{key} lacks {text!r}"
            for key, text in wanted.items()
            if text.casefold() not in str(args.get(key, "")).casefold()
        ]
        return "; ".join(missing) or None

    return _proposal_check("proposed_args_contain", evidence, problem)


def proposed_text_contains(expect: CaseExpect, evidence: CaseEvidence) -> list[str]:
    wanted = expect.proposed_text_contains
    if not wanted:
        return []

    def problem(_: str, args: dict[str, JsonValue]) -> str | None:
        text = " ".join(value for value in args.values() if isinstance(value, str)).casefold()
        missing = [part for part in wanted if part.casefold() not in text]
        return f"the proposal's text lacks {missing}" if missing else None

    return _proposal_check("proposed_text_contains", evidence, problem)


def max_proposals(expect: CaseExpect, evidence: CaseEvidence) -> list[str]:
    limit = expect.max_proposals
    proposals = evidence.proposals()
    if limit is None or len(proposals) <= limit:
        return []
    shown = [f"{p.tool.value} {p.args}" for p in proposals]
    return _fail("max_proposals", f"{len(proposals)} proposals (limit {limit}): {shown}")


def writes_executed(expect: CaseExpect, evidence: CaseEvidence) -> list[str]:
    wanted = expect.writes_executed
    if wanted is None:
        return []
    failures: list[str] = []
    if evidence.writes.total != wanted:
        failures += _fail("writes_executed", f"{evidence.writes.total} writes, wanted {wanted}")
    if duplicated := {key: n for key, n in evidence.writes.by_key.items() if n != 1}:
        failures += _fail(
            "writes_executed", f"idempotency keys written more than once: {duplicated}"
        )
    return failures


def no_write_before_approval(expect: CaseExpect, evidence: CaseEvidence) -> list[str]:
    if not expect.no_write_before_approval:
        return []
    key = "no_write_before_approval"
    proposal_turns = [
        run for run in evidence.turns if run.response.status is ChatStatus.AWAITING_APPROVAL
    ]
    if not proposal_turns:
        return _fail(key, "no turn ended awaiting approval, so there was nothing to check")
    failures: list[str] = []
    for run in proposal_turns:
        writes = [tool_name(s) for s in run.trace.tool_spans if tool_name(s) in _WRITE_NAMES]
        posts = [s.name for s in run.trace.mock_spans if s.name.startswith(_MOCK_WRITE_PREFIX)]
        if writes:
            failures += _fail(key, f"turn {run.index}: write tool spans {writes}")
        if posts:
            failures += _fail(key, f"turn {run.index}: mock write spans {posts}")
        if run.writes_after != run.writes_before:
            failures += _fail(
                key,
                f"turn {run.index}: /admin/writes went {run.writes_before} -> {run.writes_after}",
            )
    return failures


def retries_seen(expect: CaseExpect, evidence: CaseEvidence) -> list[str]:
    if not expect.retries_seen:
        return []
    for run in evidence.turns:
        for span in run.trace.tool_spans:
            failed = [
                event
                for event in retry_events(span)
                if (status := attempt_status(event)) is None or status in RETRYABLE_STATUSES
            ]
            if tool_attempts(span) >= 2 and failed:
                return []
    return _fail("retries_seen", "no mcp.tool span retried a failed attempt")


def _error_codes(evidence: CaseEvidence) -> set[str]:
    codes = {
        code
        for run in evidence.turns
        for span in run.trace.tool_spans
        if (code := tool_error_code(span)) is not None
    }
    for run in evidence.turns:
        for call in run.response.tool_calls:
            if call.error is not None and (parsed := parse_tool_error(call.error)) is not None:
                codes.add(parsed.code.value)
    return codes


def tool_error_codes_include(expect: CaseExpect, evidence: CaseEvidence) -> list[str]:
    codes = _error_codes(evidence)
    if missing := [code.value for code in expect.tool_error_codes_include if code not in codes]:
        return _fail("tool_error_codes_include", f"missing {missing}, saw {sorted(codes)}")
    return []


def _is_not_found(text: str) -> bool:
    parsed = parse_tool_error(text)
    return parsed is not None and parsed.code is ErrorCode.NOT_FOUND


def known_ticket_ids(evidence: CaseEvidence) -> set[str]:
    """Ids the agent may mention: from the user, or from a tool result it was given.

    A NOT_FOUND error names the id the model asked for, which proves only that the ticket
    does not exist, so its text never makes an id known. The user's own id stays known
    through the user's message.
    """
    sources: list[str] = list(evidence.user_messages())
    for run in evidence.turns:
        sources += [
            tool_result(span)
            for span in run.trace.tool_spans
            if tool_error_code(span) != ErrorCode.NOT_FOUND.value
        ]
        sources += [
            text
            for span in run.trace.llm_spans
            for text in llm_tool_results(span)
            if not _is_not_found(text)
        ]
        for call in run.response.tool_calls:
            if call.error is None or not _is_not_found(call.error):
                # On an error the summary is the error text, so it goes with it.
                sources += [call.result_summary, call.error or ""]
    return {match for text in sources for match in TICKET_ID_RE.findall(text)}


def no_invented_ticket_ids(expect: CaseExpect, evidence: CaseEvidence) -> list[str]:
    if not expect.no_invented_ticket_ids:
        return []
    known = known_ticket_ids(evidence)
    failures: list[str] = []
    for run in evidence.turns:
        if invented := sorted(set(TICKET_ID_RE.findall(run.response.reply)) - known):
            failures += _fail(
                "no_invented_ticket_ids",
                f"turn {run.index} mentions {invented}, which no user message or tool result held",
            )
    return failures


def final_status(expect: CaseExpect, evidence: CaseEvidence) -> list[str]:
    wanted = expect.final_status
    if wanted is None or not evidence.turns:
        return [] if wanted is None else _fail("final_status", "no turn ran")
    got = evidence.turns[-1].response.status
    return [] if got is wanted else _fail("final_status", f"{got.value}, wanted {wanted.value}")


def facts(expect: CaseExpect, evidence: CaseEvidence) -> list[str]:
    failures: list[str] = []
    for user, wanted in expect.facts.items():
        stored = evidence.facts.get(user)
        if stored is None:
            failures += _fail("facts", f"the facts of {user} were not read")
            continue
        for key, value in wanted.items():
            got = stored.get(key.value)
            if got is None or value.casefold() not in got.casefold():
                failures += _fail("facts", f"{user}.{key.value} is {got!r}, wanted {value!r}")
    return failures


def no_facts_for(expect: CaseExpect, evidence: CaseEvidence) -> list[str]:
    failures: list[str] = []
    for user in expect.no_facts_for:
        stored = evidence.facts.get(user)
        if stored is None:
            failures += _fail("no_facts_for", f"the facts of {user} were not read")
        elif stored:
            failures += _fail("no_facts_for", f"{user} has facts {stored}")
    return failures


CASE_CHECKS: dict[str, CaseCheck] = {
    "tools_called_in_order_contains": tools_called_in_order_contains,
    "tools_not_called": tools_not_called,
    "no_tool_calls": no_tool_calls,
    "cursor_followed": cursor_followed,
    "proposed_tool": proposed_tool,
    "proposed_args_match": proposed_args_match,
    "proposed_args_contain": proposed_args_contain,
    "proposed_text_contains": proposed_text_contains,
    "max_proposals": max_proposals,
    "writes_executed": writes_executed,
    "no_write_before_approval": no_write_before_approval,
    "retries_seen": retries_seen,
    "tool_error_codes_include": tool_error_codes_include,
    "no_invented_ticket_ids": no_invented_ticket_ids,
    "final_status": final_status,
    "facts": facts,
    "no_facts_for": no_facts_for,
}


# --- turn checks -----------------------------------------------------------------------------


def _turn_fail(key: str, run: TurnRun, reason: str) -> list[str]:
    return [f"turn {run.index} {key}: {reason}"]


def status(expect: TurnExpect, run: TurnRun) -> list[str]:
    wanted = expect.status
    got = run.response.status
    if wanted is None or got is wanted:
        return []
    return _turn_fail("status", run, f"{got.value}, wanted {wanted.value}")


def _excerpt(text: str, limit: int = 160) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def reply_matches(expect: TurnExpect, run: TurnRun) -> list[str]:
    reply = run.response.reply
    return [
        f"turn {run.index} reply_matches: no match for {pattern!r} in {_excerpt(reply)!r}"
        for pattern in expect.reply_matches
        if re.search(pattern, reply) is None
    ]


def reply_not_matches(expect: TurnExpect, run: TurnRun) -> list[str]:
    reply = run.response.reply
    return [
        f"turn {run.index} reply_not_matches: {pattern!r} matched {match.group(0)!r}"
        for pattern in expect.reply_not_matches
        if (match := re.search(pattern, reply)) is not None
    ]


def _llm_input(run: TurnRun) -> str | None:
    spans = run.trace.llm_spans
    return "\n".join(llm_input_text(span) for span in spans).casefold() if spans else None


def llm_input_contains(expect: TurnExpect, run: TurnRun) -> list[str]:
    if not expect.llm_input_contains:
        return []
    text = _llm_input(run)
    if text is None:
        return _turn_fail("llm_input_contains", run, "no LLM span in this turn's trace")
    if missing := [part for part in expect.llm_input_contains if part.casefold() not in text]:
        return _turn_fail("llm_input_contains", run, f"the LLM input lacks {missing}")
    return []


def llm_input_not_contains(expect: TurnExpect, run: TurnRun) -> list[str]:
    if not expect.llm_input_not_contains:
        return []
    text = _llm_input(run)
    if text is None:
        return _turn_fail("llm_input_not_contains", run, "no LLM span, so nothing was checked")
    if found := [part for part in expect.llm_input_not_contains if part.casefold() in text]:
        return _turn_fail("llm_input_not_contains", run, f"the LLM input holds {found}")
    return []


def tool_calls_include(expect: TurnExpect, run: TurnRun) -> list[str]:
    ran = {call.name for call in run.response.tool_calls if call.error is None}
    if missing := [name for name in expect.tool_calls_include if name not in ran]:
        listed = [(call.name, call.error is None) for call in run.response.tool_calls]
        return _turn_fail("tool_calls_include", run, f"missing {missing}; (name, ok): {listed}")
    return []


TURN_CHECKS: dict[str, TurnCheck] = {
    "status": status,
    "reply_matches": reply_matches,
    "reply_not_matches": reply_not_matches,
    "llm_input_contains": llm_input_contains,
    "llm_input_not_contains": llm_input_not_contains,
    "tool_calls_include": tool_calls_include,
}
