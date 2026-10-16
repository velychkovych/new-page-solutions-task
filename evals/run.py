"""`python -m evals.run`: run the eval cases and write latest.json / latest.md.

    python -m evals.run [--repeats N] [--chaos] [--model NAME] [--case ID ...] [--fake]
                        [--out DIR] [--list]

Live tier (default): the compose service `agent-eval` with the real model, and the judge
model (JUDGE_MODEL_*, the only values read from .env, through Settings) for the judge
metrics; costs money.
`--fake`: local processes with the scripted model and a scripted judge, deterministic, no
key needed; its results go to evals/results/fake/ and never replace the committed live
results.

The runs are collected first; then every run is scored through DeepEval (evals.scoring).

Exit status: 0 no run failed (a run that was not exercised does not fail), 1 a run failed
(results are written either way),
2 an infrastructure error (stack not reachable, a command failed) or bad arguments. An
infrastructure error after some runs finished still writes their results, marked aborted.
143 terminated by SIGTERM, after cleaning up the stack; nothing is written.
"""

import argparse
import asyncio
import contextlib
import signal
import sys
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import ValidationError

from evals.case_table import render_case_table
from evals.cases import CASES_DIR, Case, CaseFileError, load_cases, select_cases
from evals.chaos_profiles import CHAOS_OFF
from evals.dataset import build_dataset
from evals.driver import CollectedRun, run_case
from evals.fake_stack import FakeStack
from evals.judges import Judges, judge_settings, model_judges, no_judges, scripted_judges
from evals.live_stack import LiveStack
from evals.report import Report, RunMetadata, build_report, rate_text, write_report
from evals.results import CaseRunResult, Outcome
from evals.scoring import score_runs
from evals.stack import ROOT, Stack, StackError, Tier, run_command
from helpdesk.agent.model import ModelConfigError
from helpdesk.shared.settings import MissingSecretError

# The judge's settings come from here, not from the working directory (evals.judges).
ENV_FILE = ROOT / ".env"
LIVE_OUT = ROOT / "evals" / "results"
FAKE_OUT = LIVE_OUT / "fake"
RESULTS_PATHSPEC = LIVE_OUT.relative_to(ROOT).as_posix()
LIVE_REPEATS = 3
FAKE_REPEATS = 1

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INFRA = 2
EXIT_TERMINATED = 128 + signal.SIGTERM

type Progress = Callable[[str], None]


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__.split("\n")[0])
    parser.add_argument("--repeats", type=int, help=f"runs per case (live {LIVE_REPEATS}, fake 1)")
    parser.add_argument("--chaos", action="store_true", help="only cases that turn chaos on")
    parser.add_argument("--model", help="MODEL_NAME for agent-eval (live tier only)")
    parser.add_argument("--case", action="append", default=[], metavar="ID", help="repeatable")
    parser.add_argument("--fake", action="store_true", help="deterministic tier, fake model")
    parser.add_argument("--out", type=Path, help=f"default {LIVE_OUT} (live), {FAKE_OUT} (fake)")
    parser.add_argument("--list", action="store_true", help="print the cases as markdown")
    args = parser.parse_args(argv)
    if args.repeats is not None and args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.fake and args.model:
        parser.error("--model applies to the live tier only")
    out: Path = (args.out or (FAKE_OUT if args.fake else LIVE_OUT)).resolve()
    if args.fake and out == LIVE_OUT.resolve():
        parser.error(f"the fake tier never writes the live results in {LIVE_OUT}")
    args.out = out
    return args


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


async def git_state(repo: Path = ROOT) -> tuple[str, bool]:
    """The commit and whether the tree differs from it. evals/results is left out: the
    runner writes there, and a run must not report its own (or an earlier run's) output."""
    sha = await run_command(["git", "rev-parse", "--short=12", "HEAD"], cwd=repo)
    status = await run_command(
        ["git", "status", "--porcelain", "--", ".", f":(exclude){RESULTS_PATHSPEC}"], cwd=repo
    )
    commit = sha.stdout.strip() if sha.returncode == 0 else "unknown"
    return commit, sha.returncode == 0 and bool(status.stdout.strip())


@asynccontextmanager
async def open_stack(fake: bool, model: str | None) -> AsyncIterator[Stack]:
    if fake:
        with tempfile.TemporaryDirectory(prefix="helpdesk-evals-") as tmp:
            fake_stack = FakeStack(Path(tmp))
            try:
                await fake_stack.start()
                yield fake_stack
            finally:
                await fake_stack.close()
        return
    live = LiveStack()
    try:
        await live.start(model)
        try:
            yield live
        finally:
            # Leave the shared mock without chaos, whatever happened; a failure here must
            # not hide the error that ended the run.
            with contextlib.suppress(StackError):
                await live.set_chaos(CHAOS_OFF)
    finally:
        await live.close()


async def run_cases(
    stack: Stack,
    cases: Sequence[Case],
    repeats: int,
    progress: Progress,
    collected: list[CollectedRun],
) -> None:
    """Appends each run as it finishes, so the caller keeps them if one fails."""
    total = len(cases) * repeats
    async with httpx.AsyncClient() as client:
        for case in cases:
            for run in range(1, repeats + 1):
                result = await run_case(case, stack, client, run=run)
                collected.append(result)
                ended = "collected" if result.error is None else f"ended early: {result.error}"
                progress(
                    f"[{len(collected)}/{total}] {case.id} run {run}: {ended}"
                    f" ({result.duration_s:.1f} s)"
                )


VERDICTS = {Outcome.PASSED: "PASS", Outcome.FAILED: "FAIL", Outcome.NOT_EXERCISED: "NOT EXERCISED"}


def report_verdicts(results: Sequence[CaseRunResult], progress: Progress) -> None:
    for result in results:
        progress(f"{result.case_id} run {result.run}: {VERDICTS[result.outcome]}")
        for reason in result.reasons:
            progress(f"    - {reason}")


async def evaluate_cases(
    cases: Sequence[Case],
    *,
    fake: bool,
    model: str | None,
    repeats: int,
    chaos_only: bool,
    case_filter: Sequence[str],
    progress: Progress,
    judges: Judges,
) -> Report:
    started = utc_now()
    # Before the run: what was checked out when it started is what it measured.
    sha, dirty = await git_state()
    collected: list[CollectedRun] = []
    aborted: str | None = None
    try:
        async with open_stack(fake, model) as stack:
            await run_cases(stack, cases, repeats, progress, collected)
    except StackError as exc:
        if not collected:
            raise
        # Hours of finished runs are worth keeping; the report says they are not all.
        aborted = (
            f"infrastructure error after {len(collected)} of {len(cases) * repeats} runs: {exc}"
        )
    progress(f"Scoring {len(collected)} runs through DeepEval (judge: {judges.model})")
    results = await score_runs(collected, build_dataset(cases), judges.judge_for)
    report_verdicts(results, progress)
    metadata = RunMetadata(
        started_at=started,
        finished_at=utc_now(),
        git_sha=sha,
        git_dirty=dirty,
        tier=Tier.FAKE if fake else Tier.LIVE,
        models=sorted({m for r in results for m in r.models}),
        providers=sorted({p for r in results for p in r.providers}),
        requested_model=model,
        repeats=repeats,
        chaos_only=chaos_only,
        case_filter=list(case_filter),
        judge_model=judges.model,
        judge_provider=judges.provider,
        judge_note=judges.note,
        aborted=aborted,
    )
    return build_report(metadata, cases, results)


async def cancel_on_sigterm[T](work: Awaitable[T]) -> T:
    """Await `work`, cancelling it on SIGTERM as asyncio.run does on Ctrl-C.

    The default SIGTERM action ends the process at once: no finally block runs, so the
    fake tier's processes and temporary directory, or the live tier's chaos, are left
    behind. Cancelling runs every cleanup on the way out.
    """
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    if task is None:  # pragma: no cover - always awaited inside asyncio.run
        return await work
    with contextlib.suppress(NotImplementedError):  # no signal handlers on Windows
        loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        return await work
    finally:
        with contextlib.suppress(NotImplementedError):
            loop.remove_signal_handler(signal.SIGTERM)


def summary_lines(report: Report, out: Path) -> list[str]:
    tokens = report.tokens
    judge = report.judge_tokens
    meta = report.metadata
    return [
        f"Pass rate: {rate_text(report.overall)}",
        f"Model: {', '.join(meta.models) or '-'} (provider: {', '.join(meta.providers) or '-'})",
        f"Judge: {meta.judge_model} (provider: {meta.judge_provider}). {meta.judge_note}",
        f"Tokens: {tokens.total} (input {tokens.input}, output {tokens.output};"
        f" of the input, cache read {tokens.cache_read}, cache write {tokens.cache_write})",
        f"Judge tokens: {judge.total} (input {judge.input}, output {judge.output})",
        f"Results: {out}",
    ]


def needs_judge(cases: Sequence[Case]) -> bool:
    return any(case.judge for case in cases)


def judges_for(fake: bool, cases: Sequence[Case]) -> Judges:
    """The fake tier's scripted judge, or the judge model from the judge's settings (live).

    Built before any case runs, so a missing judge key costs no agent tokens.
    """
    if fake:
        return scripted_judges()
    if not needs_judge(cases):
        return no_judges()
    return model_judges(judge_settings(ENV_FILE))


def settings_errors(exc: ValidationError) -> str:
    """Which settings are wrong, without their values: those may be keys."""
    return "; ".join(
        f"{'.'.join(map(str, error['loc'])) or 'settings'}: {error['msg']}"
        for error in exc.errors(include_input=False, include_url=False)
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cases = select_cases(load_cases(CASES_DIR), ids=args.case, chaos_only=args.chaos)
    except (CaseFileError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INFRA
    if args.list:
        print(render_case_table(cases), end="")
        return EXIT_OK
    if not cases:
        print("error: no case matches the filters", file=sys.stderr)
        return EXIT_INFRA
    repeats = args.repeats or (FAKE_REPEATS if args.fake else LIVE_REPEATS)
    try:
        judges = judges_for(args.fake, cases)
    except ValidationError as exc:
        print(f"error: the judge model cannot be built: {settings_errors(exc)}", file=sys.stderr)
        return EXIT_INFRA
    except (MissingSecretError, ModelConfigError) as exc:
        print(f"error: the judge model cannot be built: {exc}", file=sys.stderr)
        return EXIT_INFRA
    return run_and_report(args, cases, repeats, judges)


def run_and_report(
    args: argparse.Namespace, cases: Sequence[Case], repeats: int, judges: Judges
) -> int:
    try:
        report = asyncio.run(
            cancel_on_sigterm(
                evaluate_cases(
                    cases,
                    fake=args.fake,
                    model=args.model,
                    repeats=repeats,
                    chaos_only=args.chaos,
                    case_filter=args.case,
                    progress=print,
                    judges=judges,
                )
            )
        )
    except StackError as exc:
        print(f"infrastructure error: {exc}", file=sys.stderr)
        return EXIT_INFRA
    except asyncio.CancelledError:
        print("terminated: the stack was cleaned up, no results written", file=sys.stderr)
        return EXIT_TERMINATED
    write_report(report, args.out)
    for line in summary_lines(report, args.out):
        print(line)
    if report.metadata.aborted is not None:
        print(f"aborted: {report.metadata.aborted}", file=sys.stderr)
        return EXIT_INFRA
    return EXIT_OK if report.overall.failures == 0 else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
