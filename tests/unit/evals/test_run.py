"""The CLI: arguments, --list, exit codes and where results go (no stack is started)."""

from pathlib import Path
from typing import Any

import pytest
from tests.unit.evals.conftest import make_case

from evals import run as run_module
from evals.report import Report, RunMetadata, build_report
from evals.results import CaseRunResult
from evals.run import EXIT_FAILED, EXIT_INFRA, EXIT_OK, FAKE_OUT, LIVE_OUT, main, parse_args
from evals.stack import StackError, Tier, mask_secrets
from evals.traces import TokenUsage


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


def _report(passed: bool) -> Report:
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
    result = CaseRunResult(
        case_id="unit_case",
        plan_case=[1],
        run=1,
        passed=passed,
        reasons=[] if passed else ["x"],
        turns=[],
        duration_s=0.1,
        tokens=TokenUsage(input=7, output=3),
        models=[],
        providers=[],
    )
    return build_report(metadata, [make_case()], [result])


@pytest.mark.parametrize(("passed", "code"), [(True, EXIT_OK), (False, EXIT_FAILED)])
def test_exit_code_follows_the_runs_and_results_are_written_either_way(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    passed: bool,
    code: int,
) -> None:
    async def fake_evaluate(*_: Any, **__: Any) -> Report:
        return _report(passed)

    monkeypatch.setattr(run_module, "evaluate_cases", fake_evaluate)
    assert main(["--fake", "--case", "status_question", "--out", str(tmp_path)]) == code
    assert (tmp_path / "latest.json").exists()
    assert (tmp_path / "latest.md").exists()
    out = capsys.readouterr().out
    assert "Tokens: 10 (input 7, output 3" in out
    assert "Model: - (provider: scriptedchatmodel)" in out


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
