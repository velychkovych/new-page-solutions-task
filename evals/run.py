"""`python -m evals.run`: run the eval cases and write latest.json / latest.md.

    python -m evals.run [--repeats N] [--chaos] [--model NAME] [--case ID ...] [--fake]
                        [--out DIR] [--list]

Live tier (default): the compose service `agent-eval` with the real model; costs money.
`--fake`: local processes with the scripted model, deterministic, no key needed; its
results go to evals/results/fake/ and never replace the committed live results.

Exit status: 0 every run passed, 1 a run failed (results are written either way),
2 an infrastructure error (stack not reachable, a command failed) or bad arguments.
"""

import argparse
import asyncio
import contextlib
import sys
import tempfile
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx

from evals.case_table import render_case_table
from evals.cases import CASES_DIR, Case, CaseFileError, load_cases, select_cases
from evals.chaos_profiles import CHAOS_OFF
from evals.driver import run_case
from evals.fake_stack import FakeStack
from evals.live_stack import LiveStack
from evals.report import Report, RunMetadata, build_report, write_report
from evals.results import CaseRunResult
from evals.stack import ROOT, Stack, StackError, Tier, run_command

LIVE_OUT = ROOT / "evals" / "results"
FAKE_OUT = LIVE_OUT / "fake"
LIVE_REPEATS = 3
FAKE_REPEATS = 1

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INFRA = 2

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


async def git_state() -> tuple[str, bool]:
    sha = await run_command(["git", "rev-parse", "--short=12", "HEAD"])
    status = await run_command(["git", "status", "--porcelain"])
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
    stack: Stack, cases: Sequence[Case], repeats: int, progress: Progress
) -> list[CaseRunResult]:
    results: list[CaseRunResult] = []
    total = len(cases) * repeats
    async with httpx.AsyncClient() as client:
        for case in cases:
            for run in range(1, repeats + 1):
                result = await run_case(case, stack, client, run=run)
                results.append(result)
                verdict = "PASS" if result.passed else "FAIL"
                progress(
                    f"[{len(results)}/{total}] {case.id} run {run}: {verdict}"
                    f" ({result.duration_s:.1f} s)"
                )
                for reason in result.reasons:
                    progress(f"    - {reason}")
    return results


async def evaluate_cases(
    cases: Sequence[Case],
    *,
    fake: bool,
    model: str | None,
    repeats: int,
    chaos_only: bool,
    case_filter: Sequence[str],
    progress: Progress,
) -> Report:
    started = utc_now()
    async with open_stack(fake, model) as stack:
        results = await run_cases(stack, cases, repeats, progress)
    sha, dirty = await git_state()
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
    )
    return build_report(metadata, cases, results)


def summary_lines(report: Report, out: Path) -> list[str]:
    tokens = report.tokens
    overall = report.overall
    return [
        f"Pass rate: {overall.passes}/{overall.runs} ({overall.pass_rate * 100:.0f}%)",
        f"Model: {', '.join(report.metadata.models) or '-'}"
        f" (provider: {', '.join(report.metadata.providers) or '-'})",
        f"Tokens: {tokens.total} (input {tokens.input}, output {tokens.output},"
        f" cache read {tokens.cache_read}, cache write {tokens.cache_write})",
        f"Results: {out}",
    ]


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
        report = asyncio.run(
            evaluate_cases(
                cases,
                fake=args.fake,
                model=args.model,
                repeats=repeats,
                chaos_only=args.chaos,
                case_filter=args.case,
                progress=print,
            )
        )
    except StackError as exc:
        print(f"infrastructure error: {exc}", file=sys.stderr)
        return EXIT_INFRA
    write_report(report, args.out)
    for line in summary_lines(report, args.out):
        print(line)
    return EXIT_OK if report.overall.passes == report.overall.runs else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
