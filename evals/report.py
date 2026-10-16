"""latest.json and latest.md: run metadata, pass rate per case and per metric, tokens and
latency.

A run passes, fails, is not exercised (it never reached what a check is about), or is a
judge error (no check failed, but a judge metric could not be scored). The pass rate is
over the other runs, with the not-exercised and judge-error counts next to it.
"""

import math
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from evals.cases import Case
from evals.results import CaseRunResult, MetricKind, Outcome
from evals.stack import Tier
from evals.traces import TokenUsage

JSON_NAME = "latest.json"
MD_NAME = "latest.md"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunMetadata(_Model):
    started_at: str  # ISO-8601, UTC
    finished_at: str
    git_sha: str
    git_dirty: bool
    tier: Tier
    # As recorded by the LLM spans (llm.model_name / llm.provider), not as configured.
    models: list[str]
    providers: list[str]
    requested_model: str | None
    repeats: int
    chaos_only: bool
    case_filter: list[str]
    # The model behind the judge metrics (JUDGE_MODEL_*), and why that one.
    judge_model: str = "-"
    judge_provider: str = "-"
    judge_note: str = ""
    # Set when an infrastructure error ended the run early: the report holds only the
    # runs that finished before it.
    aborted: str | None = None


class RunNote(_Model):
    run: int
    reasons: list[str]


class Tally(_Model):
    runs: int
    passes: int
    failures: int
    not_exercised: int
    judge_errors: int
    # passes / (runs - not_exercised - judge_errors); 0 when no run is left.
    pass_rate: float

    @property
    def exercised(self) -> int:
        """The runs that were exercised and fully scored: the pass rate's denominator."""
        return self.runs - self.not_exercised - self.judge_errors


class CaseSummary(_Model):
    id: str
    title: str
    plan_case: list[int]
    tally: Tally
    failures: list[RunNote]
    not_exercised: list[RunNote]
    judge_errors: list[RunNote]


class MetricSummary(_Model):
    case_id: str
    name: str
    kind: MetricKind
    evaluation_model: str | None
    tally: Tally


class CaseLatency(_Model):
    id: str
    mean_total_ms: float
    max_total_ms: float


class LatencySummary(_Model):
    turns: int
    p50_ms: float
    p95_ms: float
    max_ms: float
    per_case: list[CaseLatency]


class Report(_Model):
    metadata: RunMetadata
    overall: Tally
    cases: list[CaseSummary]
    metrics: list[MetricSummary]
    # The agent's tokens (LLM spans) and the judge's (DeepEval metric data), apart.
    tokens: TokenUsage
    judge_tokens: TokenUsage
    latency: LatencySummary
    runs: list[CaseRunResult]


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile; 0 for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return round(ordered[rank - 1], 1)


def tally(outcomes: Sequence[Outcome]) -> Tally:
    passes = sum(outcome is Outcome.PASSED for outcome in outcomes)
    skipped = sum(outcome is Outcome.NOT_EXERCISED for outcome in outcomes)
    unscored = sum(outcome is Outcome.JUDGE_ERROR for outcome in outcomes)
    exercised = len(outcomes) - skipped - unscored
    return Tally(
        runs=len(outcomes),
        passes=passes,
        failures=exercised - passes,
        not_exercised=skipped,
        judge_errors=unscored,
        pass_rate=round(passes / exercised, 3) if exercised else 0.0,
    )


def _notes(runs: Sequence[CaseRunResult], outcome: Outcome) -> list[RunNote]:
    return [RunNote(run=r.run, reasons=r.reasons) for r in runs if r.outcome is outcome]


def _metric_summaries(case: Case, runs: Sequence[CaseRunResult]) -> list[MetricSummary]:
    names: dict[str, MetricSummary] = {}
    for run in runs:
        for metric in run.metrics:
            if metric.name not in names:
                outcomes = [m.outcome for r in runs for m in r.metrics if m.name == metric.name]
                names[metric.name] = MetricSummary(
                    case_id=case.id,
                    name=metric.name,
                    kind=metric.kind,
                    evaluation_model=metric.evaluation_model,
                    tally=tally(outcomes),
                )
    return list(names.values())


def build_report(
    metadata: RunMetadata, cases: Sequence[Case], results: Sequence[CaseRunResult]
) -> Report:
    summaries: list[CaseSummary] = []
    metrics: list[MetricSummary] = []
    per_case: list[CaseLatency] = []
    for case in cases:
        runs = [r for r in results if r.case_id == case.id]
        summaries.append(
            CaseSummary(
                id=case.id,
                title=case.title,
                plan_case=case.plan_case,
                tally=tally([r.outcome for r in runs]),
                failures=_notes(runs, Outcome.FAILED),
                not_exercised=_notes(runs, Outcome.NOT_EXERCISED),
                judge_errors=_notes(runs, Outcome.JUDGE_ERROR),
            )
        )
        metrics += _metric_summaries(case, runs)
        totals = [sum(t.latency_ms for t in r.turns) for r in runs]
        if totals:
            per_case.append(
                CaseLatency(
                    id=case.id,
                    mean_total_ms=round(sum(totals) / len(totals), 1),
                    max_total_ms=round(max(totals), 1),
                )
            )
    latencies = [turn.latency_ms for r in results for turn in r.turns]
    return Report(
        metadata=metadata,
        overall=tally([r.outcome for r in results]),
        cases=summaries,
        metrics=metrics,
        tokens=sum((r.tokens for r in results), TokenUsage()),
        judge_tokens=sum((r.judge_tokens for r in results), TokenUsage()),
        latency=LatencySummary(
            turns=len(latencies),
            p50_ms=percentile(latencies, 50),
            p95_ms=percentile(latencies, 95),
            max_ms=round(max(latencies, default=0.0), 1),
            per_case=per_case,
        ),
        runs=list(results),
    )


def md_cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def rate_text(t: Tally) -> str:
    """`passes/exercised (rate)`, with the not-exercised runs and judge errors next to it."""
    rate = f"{t.pass_rate * 100:.0f}%" if t.exercised else "n/a"
    parts = [f"{t.passes}/{t.exercised} ({rate})"]
    if t.not_exercised:
        parts.append(f"{t.not_exercised} not exercised")
    if t.judge_errors:
        parts.append(f"{t.judge_errors} judge error{'s' if t.judge_errors != 1 else ''}")
    return ", ".join(parts)


def _run_notes(
    title: str, cases: Sequence[CaseSummary], pick: Callable[[CaseSummary], list[RunNote]]
) -> list[str]:
    lines = ["", f"## {title}", ""]
    listed = [(c, pick(c)) for c in cases]
    if not any(notes for _, notes in listed):
        return [*lines, "None."]
    for case, notes in listed:
        for note in notes:
            lines.append(f"- `{case.id}` run {note.run}:")
            lines += [f"  - {md_cell(reason)}" for reason in note.reasons]
    return lines


def render_markdown(report: Report) -> str:
    meta = report.metadata
    lines = [
        "# Eval results",
        "",
        "Generated by `python -m evals.run`. Do not edit by hand.",
        "",
    ]
    if meta.aborted is not None:
        lines += [
            f"**Aborted: {md_cell(meta.aborted)}.** Only the runs that finished are below.",
            "",
        ]
    lines += [
        "| | |",
        "|---|---|",
        f"| Started (UTC) | {meta.started_at} |",
        f"| Finished (UTC) | {meta.finished_at} |",
        f"| Git | `{meta.git_sha}`{' (dirty)' if meta.git_dirty else ''} |",
        f"| Tier | {meta.tier.value} |",
        f"| Model (from LLM spans) | {', '.join(meta.models) or '-'} |",
        f"| Provider (from LLM spans) | {', '.join(meta.providers) or '-'} |",
        f"| Judge model | {meta.judge_model} ({meta.judge_provider}) |",
        f"| Judge note | {md_cell(meta.judge_note) or '-'} |",
        f"| Repeats | {meta.repeats} |",
        f"| Chaos cases only | {'yes' if meta.chaos_only else 'no'} |",
        f"| Case filter | {', '.join(meta.case_filter) or 'all'} |",
        f"| **Overall pass rate** | **{rate_text(report.overall)}** |",
        "",
        "Runs pass, fail, are not exercised (the run never reached what a check is about),",
        "or are judge errors (no check failed, but a judge metric could not be scored: an API",
        "error after the SDK's retries, or no verdict in its answer). A failed check always",
        "fails the run. The pass rate leaves out the not-exercised runs and the judge errors.",
        "",
        "## Cases",
        "",
        "| Case | PLAN # | Runs | Passes | Failures | Not exercised | Judge errors | Pass rate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| `{c.id}` | {', '.join(map(str, c.plan_case))} | {c.tally.runs} | {c.tally.passes}"
        f" | {c.tally.failures} | {c.tally.not_exercised} | {c.tally.judge_errors}"
        f" | {rate_text(c.tally)} |"
        for c in report.cases
    ]
    lines += [
        "",
        "## Metrics (DeepEval)",
        "",
        "Deterministic metrics run the checks in `evals/checks.py` on the trace evidence;",
        "judge metrics (`ConversationalGEval`, strict) grade only what a reply means.",
        "",
        "| Case | Metric | Kind | Evaluated by | Passes | Failures | Not exercised"
        " | Judge errors |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| `{m.case_id}` | {md_cell(m.name)} | {m.kind.value} | {m.evaluation_model or '-'}"
        f" | {m.tally.passes} | {m.tally.failures} | {m.tally.not_exercised}"
        f" | {m.tally.judge_errors} |"
        for m in report.metrics
    ]
    lines += _run_notes("Failures", report.cases, lambda c: c.failures)
    lines += _run_notes("Not exercised", report.cases, lambda c: c.not_exercised)
    lines += _run_notes("Judge errors", report.cases, lambda c: c.judge_errors)
    tokens = report.tokens
    judge = report.judge_tokens
    latency = report.latency
    lines += [
        "",
        "## Tokens",
        "",
        "Agent: from the LLM spans. Input counts every prompt token, the cached ones",
        "included (OpenInference's `llm.token_count.prompt`); cache read and cache write are",
        "the parts of it served from or written to the prompt cache. Judge: from DeepEval's",
        "metric data (the judge model's usage).",
        "",
        "| Model | Input | Output | Cache read | Cache write | Input + output |",
        "|---|---|---|---|---|---|",
        f"| Agent | {tokens.input} | {tokens.output} | {tokens.cache_read} | {tokens.cache_write}"
        f" | {tokens.total} |",
        f"| Judge | {judge.input} | {judge.output} | - | - | {judge.total} |",
        "",
        "## Latency (client side, per /chat request)",
        "",
        "| Turns | p50 ms | p95 ms | max ms |",
        "|---|---|---|---|",
        f"| {latency.turns} | {latency.p50_ms} | {latency.p95_ms} | {latency.max_ms} |",
        "",
        "| Case | Mean total ms per run | Max total ms per run |",
        "|---|---|---|",
    ]
    lines += [f"| `{c.id}` | {c.mean_total_ms} | {c.max_total_ms} |" for c in latency.per_case]
    return "\n".join(lines) + "\n"


def write_report(report: Report, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / JSON_NAME, out_dir / MD_NAME
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
