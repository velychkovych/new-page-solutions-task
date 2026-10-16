"""Score the collected runs through DeepEval: one test case per run, every metric measured.

The metrics are measured with `await metric.a_measure(test_case)`, which is the call
`deepeval.evaluate()` makes for each metric, rather than through `evaluate()` itself.
`evaluate()` runs its own event loop (patching ours with nest_asyncio when one is
running), prints banners to stdout, writes the test cases with their inputs and outputs
to `.deepeval/` even with caching off, and keeps a process-global test run. Measuring
directly keeps our loop, our stdout summary and our files, and bounds the judge calls with
one semaphore. Each result is still DeepEval's own: `create_metric_data` is the function
`evaluate()` uses to turn a measured metric into its MetricData.
"""

import asyncio
from collections.abc import Sequence
from typing import cast

from deepeval.dataset import ConversationalGolden, EvaluationDataset
from deepeval.evaluate.utils import create_metric_data
from deepeval.metrics import BaseConversationalMetric, BaseMetric
from deepeval.test_case import ConversationalTestCase

from evals.dataset import build_test_case
from evals.driver import CollectedRun
from evals.judges import JudgeError, JudgeFor, cost_tokens
from evals.metrics import ExpectationMetric, expectation_metrics, judge_metrics
from evals.results import CaseRunResult, MetricKind, MetricResult, Outcome
from evals.traces import TokenUsage

# Judge calls in flight at once, across all runs.
JUDGE_CONCURRENCY = 4


def _outcome(metric: BaseConversationalMetric) -> Outcome:
    if isinstance(metric, ExpectationMetric):
        if metric.not_exercised is not None:
            return Outcome.NOT_EXERCISED
    elif metric.error is not None:
        return Outcome.JUDGE_ERROR
    return Outcome.PASSED if metric.is_successful() else Outcome.FAILED


def metric_result(metric: BaseConversationalMetric) -> MetricResult:
    # create_metric_data is annotated for single-turn metrics; evaluate() passes
    # conversational ones to it too, and it reads only attributes both kinds have.
    data = create_metric_data(cast(BaseMetric, metric))
    return MetricResult(
        name=data.name,
        kind=MetricKind.DETERMINISTIC
        if isinstance(metric, ExpectationMetric)
        else MetricKind.JUDGE,
        outcome=_outcome(metric),
        score=data.score,
        threshold=data.threshold,
        reason=data.reason,
        evaluation_model=data.evaluation_model,
        error=data.error,
        tokens=cost_tokens(data.input_tokens, data.output_tokens),
    )


def _reasons(metric: BaseConversationalMetric, result: MetricResult) -> list[str]:
    if isinstance(metric, ExpectationMetric):
        if metric.not_exercised is not None:
            return [f"{result.name}: {metric.not_exercised}"]
        return metric.failures if result.outcome is Outcome.FAILED else []
    if result.outcome is Outcome.JUDGE_ERROR:
        return [f"judge {result.name}: {result.error}"]
    if result.outcome is Outcome.FAILED:
        return [f"judge {result.name}: {result.reason}"]
    return []


async def _measure_judge(
    metric: BaseConversationalMetric, test_case: ConversationalTestCase, limit: asyncio.Semaphore
) -> None:
    async with limit:
        try:
            await metric.a_measure(test_case, _show_indicator=False)
        except Exception as exc:
            # The judge failed (API error after the SDK's retries, an answer without the
            # schema): the metric is a judge error and says so, instead of ending the whole
            # run.
            if isinstance(exc, JudgeError):
                # A call that answered unusably was still paid for; DeepEval only counts
                # the tokens of a call that returns.
                metric._accrue_tokens(exc.cost.input_tokens, exc.cost.output_tokens)
            metric.error = f"judge error: {type(exc).__name__}: {exc}"
            metric.success = False


def _run_outcome(metrics: Sequence[MetricResult]) -> Outcome:
    """A failed metric fails the run, whatever else happened: a judge that could not be
    scored never hides a check that failed. A judge error comes before not exercised, as
    the run that has both was not fully scored."""
    outcomes = {metric.outcome for metric in metrics}
    for outcome in (Outcome.FAILED, Outcome.JUDGE_ERROR, Outcome.NOT_EXERCISED):
        if outcome in outcomes:
            return outcome
    return Outcome.PASSED


def _unscored(collected: CollectedRun, error: str) -> CaseRunResult:
    return CaseRunResult(
        case_id=collected.case.id,
        plan_case=collected.case.plan_case,
        run=collected.run,
        outcome=Outcome.FAILED,
        reasons=[error],
        turns=collected.turns,
        duration_s=collected.duration_s,
        tokens=collected.tokens,
        models=collected.models,
        providers=collected.providers,
        error=error,
    )


async def score_run(
    collected: CollectedRun,
    golden: ConversationalGolden,
    judge_for: JudgeFor,
    limit: asyncio.Semaphore,
    dataset: EvaluationDataset | None = None,
) -> CaseRunResult:
    if collected.evidence is None:
        return _unscored(collected, collected.error or "the run collected no evidence")
    test_case = build_test_case(golden, collected.evidence, collected.run)
    if dataset is not None:
        dataset.add_test_case(test_case)
    deterministic = expectation_metrics(collected.case)
    judges = judge_metrics(collected.case, judge_for)
    for metric in deterministic:
        await metric.a_measure(test_case)
    await asyncio.gather(*(_measure_judge(metric, test_case, limit) for metric in judges))
    measured: list[BaseConversationalMetric] = [*deterministic, *judges]
    results = [metric_result(metric) for metric in measured]
    failed = [
        reason
        for metric, result in zip(measured, results, strict=True)
        for reason in _reasons(metric, result)
    ]
    return CaseRunResult(
        case_id=collected.case.id,
        plan_case=collected.case.plan_case,
        run=collected.run,
        outcome=_run_outcome(results),
        reasons=failed,
        turns=collected.turns,
        duration_s=collected.duration_s,
        tokens=collected.tokens,
        models=collected.models,
        providers=collected.providers,
        metrics=results,
        judge_tokens=sum((r.tokens for r in results), TokenUsage()),
    )


async def score_runs(
    collected: Sequence[CollectedRun],
    dataset: EvaluationDataset,
    judge_for: JudgeFor,
    *,
    concurrency: int = JUDGE_CONCURRENCY,
) -> list[CaseRunResult]:
    """Every run's result, in the order the runs were collected."""
    goldens = {
        golden.name: golden
        for golden in dataset.goldens
        if isinstance(golden, ConversationalGolden)
    }
    limit = asyncio.Semaphore(concurrency)
    return list(
        await asyncio.gather(
            *(score_run(run, goldens[run.case.id], judge_for, limit, dataset) for run in collected)
        )
    )
