"""The CLI: arguments, --list, exit codes and where results go (no stack is started)."""

import subprocess
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from tests.unit.evals.conftest import collected, make_case

from evals import run as run_module
from evals.judges import scripted_judges
from evals.report import Report, RunMetadata, build_report
from evals.results import CaseRunResult, Outcome
from evals.run import (
    EXIT_FAILED,
    EXIT_INFRA,
    EXIT_OK,
    FAKE_OUT,
    LIVE_OUT,
    git_state,
    main,
    parse_args,
)
from evals.stack import StackError, Tier, mask_secrets
from evals.traces import TokenUsage
from helpdesk.shared.settings import MissingSecretError


def test_defaults() -> None:
    args = parse_args([])
    assert args.out == LIVE_OUT.resolve()
    assert not args.fake
    assert parse_args(["--fake"]).out == FAKE_OUT.resolve()


@pytest.mark.parametrize(
    "argv",
    [
        ["--fake", "--model", "claude-haiku-5-5"],
        ["--fake", "--out", str(LIVE_OUT)],
        ["--repeats", "0"],
    ],
)
def test_bad_arguments_exit_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        parse_args(argv)
    assert exc.value.code == 2


def test_list_prints_the_case_table(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list"]) == EXIT_OK
    out = capsys.readouterr().out
    assert out.startswith("| Case | PLAN # |")
    assert "`status_question`" in out


def test_list_with_chaos_lists_only_chaos_cases(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list", "--chaos"]) == EXIT_OK
    rows = [line for line in capsys.readouterr().out.splitlines() if line.startswith("| `")]
    assert len(rows) == 2


def test_an_unknown_case_id_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--case", "nope", "--list"]) == EXIT_INFRA
    assert "unknown case id" in capsys.readouterr().err


def _report(*outcomes: Outcome) -> Report:
    metadata = RunMetadata(
        started_at="s",
        finished_at="f",
        git_sha="sha",
        git_dirty=False,
        tier=Tier.FAKE,
        models=[],
        providers=["scriptedchatmodel"],
        requested_model=None,
        repeats=1,
        chaos_only=False,
        case_filter=[],
    )
    results = [
        CaseRunResult(
            case_id="unit_case",
            plan_case=[1],
            run=run,
            outcome=outcome,
            reasons=[] if outcome is Outcome.PASSED else ["x"],
            turns=[],
            duration_s=0.1,
            tokens=TokenUsage(input=7, output=3),
            models=[],
            providers=[],
            judge_tokens=TokenUsage(input=20, output=2),
        )
        for run, outcome in enumerate(outcomes, start=1)
    ]
    return build_report(metadata, [make_case()], results)


@pytest.mark.parametrize(
    "expected",
    [
        (Outcome.PASSED, EXIT_OK, "Pass rate: 1/1 (100%)"),
        (Outcome.FAILED, EXIT_FAILED, "Pass rate: 0/1 (0%)"),
        # Not exercised is not a failure, and the pass rate says so.
        (Outcome.NOT_EXERCISED, EXIT_OK, "Pass rate: 0/0 (n/a), 1 not exercised"),
        # Nor is a judge error, but the run was not fully scored: no pass either.
        (Outcome.JUDGE_ERROR, EXIT_INFRA, "Pass rate: 0/0 (n/a), 1 judge error"),
    ],
)
def test_exit_code_follows_the_runs_and_results_are_written_either_way(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    expected: tuple[Outcome, int, str],
) -> None:
    outcome, code, rate = expected

    async def fake_evaluate(*_: Any, **__: Any) -> Report:
        return _report(outcome)

    monkeypatch.setattr(run_module, "evaluate_cases", fake_evaluate)
    assert main(["--fake", "--case", "status_question", "--out", str(tmp_path)]) == code
    assert (tmp_path / "latest.json").exists()
    assert (tmp_path / "latest.md").exists()
    out = capsys.readouterr().out
    assert rate in out
    assert "Tokens: 10 (input 7, output 3" in out
    assert "Judge tokens: 22 (input 20, output 2)" in out
    assert "Model: - (provider: scriptedchatmodel)" in out


@pytest.mark.parametrize(
    ("outcomes", "code"),
    [
        # A failed run is the agent's failure, whatever the judge did on another run.
        ((Outcome.FAILED, Outcome.JUDGE_ERROR), EXIT_FAILED),
        ((Outcome.PASSED, Outcome.JUDGE_ERROR, Outcome.NOT_EXERCISED), EXIT_INFRA),
    ],
)
def test_a_failure_wins_over_judge_errors_and_judge_errors_are_not_a_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    outcomes: tuple[Outcome, ...],
    code: int,
) -> None:
    async def fake_evaluate(*_: Any, **__: Any) -> Report:
        return _report(*outcomes)

    monkeypatch.setattr(run_module, "evaluate_cases", fake_evaluate)
    assert main(["--fake", "--case", "status_question", "--out", str(tmp_path)]) == code
    assert (tmp_path / "latest.md").exists()
    err = capsys.readouterr().err
    assert ("judge errors: 1 run(s) could not be fully scored" in err) is (code == EXIT_INFRA)


def test_the_live_tier_needs_the_judge_before_any_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_key(*_: Any) -> Any:
        raise MissingSecretError("ANTHROPIC_API_KEY")

    async def must_not_run(*_: Any, **__: Any) -> Report:
        raise AssertionError("no case may run without a judge")

    monkeypatch.setattr(run_module, "model_judges", no_key)
    monkeypatch.setattr(run_module, "judge_settings", lambda _: None)
    monkeypatch.setattr(run_module, "evaluate_cases", must_not_run)
    assert main(["--case", "status_question", "--out", str(tmp_path)]) == EXIT_INFRA
    assert (
        "the judge model cannot be built: ANTHROPIC_API_KEY is not set" in capsys.readouterr().err
    )


def test_a_settings_error_names_the_setting_and_never_prints_a_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("MOCK_API_TOKEN=mock-FAKEPROBE456\n", encoding="utf-8")
    monkeypatch.setattr(run_module, "ENV_FILE", env_file)
    # A model-level error: pydantic's own message repeats the head and tail of every
    # input, which here are a mock token and the key.
    monkeypatch.setenv("CHAOS_RATE_LIMIT_RATE", "0.7")
    monkeypatch.setenv("CHAOS_ERROR_RATE", "0.7")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-FAKEPROBE123")
    assert main(["--case", "status_question", "--out", str(tmp_path)]) == EXIT_INFRA
    err = capsys.readouterr().err
    assert "the judge model cannot be built" in err
    assert "CHAOS_RATE_LIMIT_RATE + CHAOS_ERROR_RATE must be <= 1" in err
    assert "FAKEPROBE" not in err


def test_judges_per_tier() -> None:
    judged = [c for c in run_module.load_cases() if c.judge]
    unjudged = [c for c in run_module.load_cases() if not c.judge]
    assert run_module.judges_for(True, judged).model == "scripted-judge"
    assert run_module.judges_for(False, unjudged).model == "-"


def test_an_infrastructure_error_exits_2_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    async def broken(*_: Any, **__: Any) -> Report:
        raise StackError("agent-eval is not healthy")

    monkeypatch.setattr(run_module, "evaluate_cases", broken)
    assert main(["--fake", "--out", str(tmp_path)]) == EXIT_INFRA
    assert "infrastructure error: agent-eval is not healthy" in capsys.readouterr().err
    assert not (tmp_path / "latest.json").exists()


def test_mask_secrets_hides_keys_and_named_secrets() -> None:
    text = "key sk-ant-api03-abcDEF_123 token s3cr3t-value and " + "a" * 40
    masked = mask_secrets(text, ["s3cr3t-value"])
    assert "sk-ant" not in masked
    assert "s3cr3t-value" not in masked
    assert "a" * 40 not in masked


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "evals" / "results").mkdir(parents=True)
    (repo / "tracked.txt").write_text("a\n")
    (repo / "evals" / "results" / "latest.md").write_text("old\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "x")
    return repo


async def test_git_state_ignores_the_results_directory(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "evals" / "results" / "latest.md").write_text("new\n")
    (repo / "evals" / "results" / "fake").mkdir()
    (repo / "evals" / "results" / "fake" / "latest.json").write_text("{}\n")
    sha, dirty = await git_state(repo)
    assert len(sha) == 12
    assert not dirty


async def test_git_state_sees_any_other_change(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "tracked.txt").write_text("b\n")
    assert (await git_state(repo))[1]


async def test_the_git_state_is_taken_before_the_stack_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    async def fake_git_state() -> tuple[str, bool]:
        order.append("git")
        return "abc", False

    @asynccontextmanager
    async def fake_open_stack(*_: Any) -> AsyncIterator[None]:
        order.append("stack")
        yield None

    async def no_cases(*_: Any) -> None:
        return None

    monkeypatch.setattr(run_module, "git_state", fake_git_state)
    monkeypatch.setattr(run_module, "open_stack", fake_open_stack)
    monkeypatch.setattr(run_module, "run_cases", no_cases)
    report = await run_module.evaluate_cases(
        [make_case()],
        fake=True,
        model=None,
        repeats=1,
        chaos_only=False,
        case_filter=[],
        progress=lambda _: None,
        judges=scripted_judges(),
    )
    assert order == ["git", "stack"]
    assert report.metadata.git_sha == "abc"


async def _evaluate_breaking_after(monkeypatch: pytest.MonkeyPatch, finished: int) -> Report:
    async def fake_git_state() -> tuple[str, bool]:
        return "abc", False

    @asynccontextmanager
    async def fake_open_stack(*_: Any) -> AsyncIterator[None]:
        yield None

    async def breaks(*args: Any) -> None:
        runs: list[Any] = args[-1]
        runs += [collected(None, error="x", run=i) for i in range(1, finished + 1)]
        raise StackError("mock GET /admin/writes returned 502")

    monkeypatch.setattr(run_module, "git_state", fake_git_state)
    monkeypatch.setattr(run_module, "open_stack", fake_open_stack)
    monkeypatch.setattr(run_module, "run_cases", breaks)
    return await run_module.evaluate_cases(
        [make_case()],
        fake=True,
        model=None,
        repeats=3,
        chaos_only=False,
        case_filter=[],
        progress=lambda _: None,
        judges=scripted_judges(),
    )


async def test_an_infrastructure_error_mid_run_keeps_the_finished_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = await _evaluate_breaking_after(monkeypatch, finished=2)
    assert report.overall.runs == 2
    assert report.metadata.aborted == (
        "infrastructure error after 2 of 3 runs: mock GET /admin/writes returned 502"
    )


async def test_an_infrastructure_error_before_any_run_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(StackError):
        await _evaluate_breaking_after(monkeypatch, finished=0)


def test_an_aborted_run_writes_its_results_marked_aborted_and_exits_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    aborted = "infrastructure error after 1 of 3 runs: agent-eval is not healthy"
    report = _report(Outcome.PASSED)
    report = report.model_copy(
        update={"metadata": report.metadata.model_copy(update={"aborted": aborted})}
    )

    async def fake_evaluate(*_: Any, **__: Any) -> Report:
        return report

    monkeypatch.setattr(run_module, "evaluate_cases", fake_evaluate)
    assert main(["--fake", "--out", str(tmp_path)]) == EXIT_INFRA
    assert f"aborted: {aborted}" in capsys.readouterr().err
    written = Report.model_validate_json((tmp_path / "latest.json").read_text())
    assert written.metadata.aborted == aborted
    assert f"**Aborted: {aborted}.**" in (tmp_path / "latest.md").read_text()
