"""latest.json / latest.md shape, and the --list table."""

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
    render_markdown,
    write_report,
)
from evals.results import CaseRunResult, TurnResult
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


def _run(run: int, *, passed: bool, latency_ms: float, case_id: str = "unit_case") -> CaseRunResult:
    tokens = TokenUsage(input=100, output=10, cache_read=60, cache_write=5)
    return CaseRunResult(
        case_id=case_id,
        plan_case=[1],
        run=run,
        passed=passed,
        reasons=[] if passed else ["writes_executed: 2 writes, wanted 1 | pipe"],
        turns=[_turn(latency_ms, tokens)],
        duration_s=1.0,
        tokens=tokens,
        models=["claude-sonnet-5-5"],
        providers=["anthropic"],
    )


def _report() -> Report:
    cases = [make_case(), make_case(id="other_case", plan_case=[2])]
    results = [
        _run(1, passed=True, latency_ms=100),
        _run(2, passed=False, latency_ms=300),
        _run(1, passed=True, latency_ms=200, case_id="other_case"),
    ]
    return build_report(METADATA, cases, results)


def test_percentile_is_nearest_rank() -> None:
    assert percentile([], 50) == 0.0
    assert percentile([5.0], 95) == 5.0
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile(list(range(1, 101)), 95) == 95


def test_build_report_summarises_cases_tokens_and_latency() -> None:
    report = _report()
    assert (report.overall.runs, report.overall.passes) == (3, 2)
    assert report.overall.pass_rate == 0.667
    unit, other = report.cases
    assert (unit.runs, unit.passes, unit.pass_rate) == (2, 1, 0.5)
    assert unit.failures[0].run == 2
    assert other.failures == []
    assert report.tokens == TokenUsage(input=300, output=30, cache_read=180, cache_write=15)
    assert (report.latency.turns, report.latency.max_ms) == (3, 300)
    assert report.latency.p50_ms == 200
    assert report.latency.per_case[0].mean_total_ms == 200


def test_markdown_has_metadata_cases_failures_tokens_and_latency() -> None:
    md = render_markdown(_report())
    for needle in (
        "`abc123def456` (dirty)",
        "| Model (from LLM spans) | claude-sonnet-5-5 |",
        "**2/3 (67%)**",
        "| `unit_case` | 1 | 2 | 1 | 50% |",
        "## Failures",
        "- `unit_case` run 2:",
        "wanted 1 \\| pipe",  # escaped for the table-free list, still safe in markdown
        "| 300 | 30 | 180 | 15 | 330 |",
        "| 3 | 200.0 | 300.0 | 300.0 |",
    ):
        assert needle in md, needle


def test_write_report_writes_both_files(tmp_path: Path) -> None:
    report = _report()
    json_path, md_path = write_report(report, tmp_path / "out")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(data) == {"metadata", "overall", "cases", "tokens", "latency", "runs"}
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
