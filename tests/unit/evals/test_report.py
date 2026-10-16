"""latest.json / latest.md shape (pass / fail / not exercised / judge error, metrics, both
token kinds), and the --list table."""

import json
from pathlib import Path

from tests.unit.evals.conftest import TRACE_ID, make_case

from evals.case_table import render_case_table
from evals.cases import load_cases
from evals.report import (
    Report,
    RunMetadata,
    build_report,
    percentile,
    rate_text,
    render_markdown,
    tally,
    write_report,
)
from evals.results import CaseRunResult, MetricKind, MetricResult, Outcome, TurnResult
from evals.stack import Tier
from evals.traces import TokenUsage
from helpdesk.shared.api import ChatStatus

METADATA = RunMetadata(
    started_at="2026-10-16T12:00:00+00:00",
    finished_at="2026-10-16T12:05:00+00:00",
    git_sha="abc123def456",
    git_dirty=True,
    tier=Tier.LIVE,
    models=["claude-sonnet-5-5"],
    providers=["anthropic"],
    requested_model=None,
    repeats=2,
    chaos_only=False,
    case_filter=[],
    judge_model="claude-haiku-5-5",
    judge_provider="anthropic",
    judge_note="Haiku to save budget.",
)


def _turn(latency_ms: float, tokens: TokenUsage) -> TurnResult:
    return TurnResult(
        index=1,
        user_id="alice",
        session="main",
        kind="message",
        text="hi",
        status=ChatStatus.OK,
        latency_ms=latency_ms,
        trace_id=TRACE_ID,
        tokens=tokens,
    )


def _metric(
    name: str, outcome: Outcome, kind: MetricKind = MetricKind.DETERMINISTIC
) -> MetricResult:
    judge = kind is MetricKind.JUDGE
    return MetricResult(
        name=name,
        kind=kind,
        outcome=outcome,
        score=None if outcome is Outcome.NOT_EXERCISED else float(outcome is Outcome.PASSED),
        threshold=1.0,
        reason="r",
        evaluation_model="claude-haiku-5-5" if judge else "deterministic",
        tokens=TokenUsage(input=50, output=5) if judge else TokenUsage(),
    )


REASONS = {
    Outcome.PASSED: [],
    Outcome.FAILED: ["writes_executed: 2 writes, wanted 1 | pipe"],
    Outcome.NOT_EXERCISED: ["cursor_followed: not exercised: every match on one page"],
    Outcome.JUDGE_ERROR: ["judge refuses [Conversational GEval]: judge error: APITimeoutError"],
}


def _run(
    run: int, *, outcome: Outcome, latency_ms: float, case_id: str = "unit_case"
) -> CaseRunResult:
    tokens = TokenUsage(input=100, output=10, cache_read=60, cache_write=5)
    writes = Outcome.FAILED if outcome is Outcome.FAILED else Outcome.PASSED
    cursor = Outcome.NOT_EXERCISED if outcome is Outcome.NOT_EXERCISED else Outcome.PASSED
    judged = Outcome.JUDGE_ERROR if outcome is Outcome.JUDGE_ERROR else Outcome.PASSED
    return CaseRunResult(
        case_id=case_id,
        plan_case=[1],
        run=run,
        outcome=outcome,
        reasons=REASONS[outcome],
        turns=[_turn(latency_ms, tokens)],
        duration_s=1.0,
        tokens=tokens,
        models=["claude-sonnet-5-5"],
        providers=["anthropic"],
        metrics=[
            _metric("writes_executed", writes),
            _metric("cursor_followed", cursor),
            _metric("refuses [Conversational GEval]", judged, MetricKind.JUDGE),
        ],
        judge_tokens=TokenUsage(input=50, output=5),
    )


def _report() -> Report:
    cases = [make_case(), make_case(id="other_case", plan_case=[2])]
    results = [
        _run(1, outcome=Outcome.PASSED, latency_ms=100),
        _run(2, outcome=Outcome.FAILED, latency_ms=300),
        _run(3, outcome=Outcome.NOT_EXERCISED, latency_ms=150),
        _run(1, outcome=Outcome.PASSED, latency_ms=200, case_id="other_case"),
    ]
    return build_report(METADATA, cases, results)


def test_percentile_is_nearest_rank() -> None:
    assert percentile([], 50) == 0.0
    assert percentile([5.0], 95) == 5.0
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile(list(range(1, 101)), 95) == 95


def test_the_pass_rate_leaves_out_the_runs_that_were_not_exercised() -> None:
    t = tally([Outcome.PASSED, Outcome.FAILED, Outcome.NOT_EXERCISED, Outcome.PASSED])
    assert (t.runs, t.passes, t.failures, t.not_exercised, t.exercised) == (4, 2, 1, 1, 3)
    assert t.pass_rate == 0.667
    assert rate_text(t) == "2/3 (67%), 1 not exercised"
    assert rate_text(tally([Outcome.PASSED])) == "1/1 (100%)"
    nothing = tally([Outcome.NOT_EXERCISED])
    assert (nothing.pass_rate, rate_text(nothing)) == (0.0, "0/0 (n/a), 1 not exercised")


def test_the_pass_rate_leaves_out_judge_errors_and_counts_them_next_to_it() -> None:
    outcomes = [Outcome.PASSED] * 9 + [Outcome.FAILED, Outcome.NOT_EXERCISED]
    t = tally([*outcomes, Outcome.JUDGE_ERROR, Outcome.JUDGE_ERROR])
    assert (t.runs, t.passes, t.failures, t.not_exercised, t.judge_errors) == (13, 9, 1, 1, 2)
    assert (t.exercised, t.pass_rate) == (10, 0.9)
    assert rate_text(t) == "9/10 (90%), 1 not exercised, 2 judge errors"
    one = tally([Outcome.PASSED, Outcome.JUDGE_ERROR])
    assert (one.failures, rate_text(one)) == (0, "1/1 (100%), 1 judge error")


def _report_with_a_judge_error() -> Report:
    cases = [make_case(), make_case(id="other_case", plan_case=[2])]
    results = [
        _run(1, outcome=Outcome.PASSED, latency_ms=100),
        _run(1, outcome=Outcome.JUDGE_ERROR, latency_ms=200, case_id="other_case"),
    ]
    return build_report(METADATA, cases, results)


def test_a_judge_error_is_reported_apart_from_failures() -> None:
    report = _report_with_a_judge_error()
    assert (report.overall.failures, report.overall.judge_errors) == (0, 1)
    assert rate_text(report.overall) == "1/1 (100%), 1 judge error"
    _, other = report.cases
    assert other.failures == []
    assert [n.run for n in other.judge_errors] == [1]
    judge = next(
        m for m in report.metrics if m.case_id == "other_case" and m.kind is MetricKind.JUDGE
    )
    assert (judge.tally.failures, judge.tally.judge_errors) == (0, 1)
    md = render_markdown(report)
    for needle in (
        "**1/1 (100%), 1 judge error**",
        "| `other_case` | 2 | 1 | 0 | 0 | 0 | 1 | 0/0 (n/a), 1 judge error |",
        "| `other_case` | refuses [Conversational GEval] | judge | claude-haiku-5-5"
        " | 0 | 0 | 0 | 1 |",
        "## Judge errors",
        "- `other_case` run 1:",
        "judge error: APITimeoutError",
    ):
        assert needle in md, needle
    assert md.split("## Failures")[1].split("##")[0].strip() == "None."


def test_build_report_summarises_cases_metrics_tokens_and_latency() -> None:
    report = _report()
    overall = report.overall
    assert (overall.runs, overall.passes, overall.failures, overall.not_exercised) == (4, 2, 1, 1)
    assert overall.pass_rate == 0.667
    unit, other = report.cases
    assert (unit.tally.runs, unit.tally.passes, unit.tally.pass_rate) == (3, 1, 0.5)
    assert [f.run for f in unit.failures] == [2]
    assert [n.run for n in unit.not_exercised] == [3]
    assert other.failures == []
    cursor = next(
        m for m in report.metrics if m.case_id == "unit_case" and m.name == "cursor_followed"
    )
    assert (cursor.tally.passes, cursor.tally.not_exercised) == (2, 1)
    judge = next(m for m in report.metrics if m.kind is MetricKind.JUDGE)
    assert judge.evaluation_model == "claude-haiku-5-5"
    assert report.tokens == TokenUsage(input=400, output=40, cache_read=240, cache_write=20)
    assert report.judge_tokens == TokenUsage(input=200, output=20)
    assert (report.latency.turns, report.latency.max_ms) == (4, 300)
    assert report.latency.p50_ms == 150
    assert report.latency.per_case[0].mean_total_ms == 183.3


def test_markdown_has_metadata_cases_metrics_failures_tokens_and_latency() -> None:
    md = render_markdown(_report())
    for needle in (
        "`abc123def456` (dirty)",
        "| Model (from LLM spans) | claude-sonnet-5-5 |",
        "| Judge model | claude-haiku-5-5 (anthropic) |",
        "| Judge note | Haiku to save budget. |",
        "**2/3 (67%), 1 not exercised**",
        "| `unit_case` | 1 | 3 | 1 | 1 | 1 | 0 | 1/2 (50%), 1 not exercised |",
        "| `unit_case` | cursor_followed | deterministic | deterministic | 2 | 0 | 1 | 0 |",
        "| `unit_case` | refuses [Conversational GEval] | judge | claude-haiku-5-5"
        " | 3 | 0 | 0 | 0 |",
        "## Failures",
        "- `unit_case` run 2:",
        "wanted 1 \\| pipe",  # escaped for the table-free list, still safe in markdown
        "## Not exercised",
        "- `unit_case` run 3:",
        "## Judge errors\n\nNone.",
        "| Agent | 400 | 40 | 240 | 20 | 440 |",
        "| Judge | 200 | 20 | - | - | 220 |",
        "| 4 | 150.0 | 300.0 | 300.0 |",
    ):
        assert needle in md, needle


def test_write_report_writes_both_files(tmp_path: Path) -> None:
    report = _report()
    json_path, md_path = write_report(report, tmp_path / "out")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(data) == {
        "metadata", "overall", "cases", "metrics", "tokens", "judge_tokens", "latency", "runs"
    }  # fmt: skip
    assert data["metadata"]["judge_model"] == "claude-haiku-5-5"
    assert data["runs"][2]["outcome"] == "not_exercised"
    assert data["metadata"]["tier"] == "live"
    assert data["metadata"]["git_sha"] == "abc123def456"
    assert Report.model_validate(data) == report
    assert md_path.read_text(encoding="utf-8") == render_markdown(report)


def test_case_table_lists_every_case_with_its_turns_and_checks() -> None:
    cases = load_cases()
    table = render_case_table(cases)
    lines = table.splitlines()
    assert lines[0].startswith("| Case | PLAN # |")
    assert len(lines) == 2 + len(cases)
    assert '"yes"' in table  # the text approval
    assert "[approve]" in table
    assert "[decline]" in table
    assert "(chaos: moderate)" in table
    assert "(chaos: all_5xx)" in table
    assert "no_write_before_approval" in table
    assert "(alice)" in table  # memory_isolation's other user
    assert "judge: refuses_out_of_scope" in table
