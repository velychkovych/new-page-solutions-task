"""The deterministic eval tier in `make test`: every case passes with its fake script, and a
wrong script fails on the metric it breaks. Each run is collected by the driver and scored
through DeepEval with the scripted judge, as `python -m evals.run --fake` does.

Real processes on loopback (`python -m helpdesk.{mock,mcp_server,agent}`), started once for
the module; the fake model replaces the real one, so no key and no network beyond
127.0.0.1. The agent restarts whenever a turn needs another script, which is most of the
time a case takes, so the cases are spread over a few independent stacks (own processes,
ports, tokens and databases) that run side by side.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from tests.unit.evals.conftest import score, tree_state

from evals import DEEPEVAL_HOME
from evals.cases import Case, load_cases
from evals.driver import run_case
from evals.fake_stack import FakeStack
from evals.report import RunMetadata, build_report, write_report
from evals.results import CaseRunResult, Outcome
from evals.stack import ROOT, Tier
from helpdesk.agent.fake_model import ScriptedMessage

pytestmark = pytest.mark.asyncio(loop_scope="module")

CASES = load_cases()
WORKERS = 3


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def stacks(tmp_path_factory: pytest.TempPathFactory) -> AsyncIterator[list[FakeStack]]:
    async with AsyncExitStack() as cleanup:
        fakes = [FakeStack(tmp_path_factory.mktemp(f"fake-tier-{i}")) for i in range(WORKERS)]
        for fake in fakes:
            cleanup.push_async_callback(fake.close)
        await asyncio.gather(*(fake.start() for fake in fakes))
        yield fakes


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def stack(stacks: list[FakeStack]) -> FakeStack:
    return stacks[0]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as http:
        yield http


def _case(case_id: str) -> Case:
    (case,) = [case for case in CASES if case.id == case_id]
    return case


def _with_script(case: Case, *scripts: list[dict[str, Any]]) -> Case:
    fake = [[ScriptedMessage.model_validate(step) for step in script] for script in scripts]
    return case.model_copy(update={"fake": fake})


async def run_and_score(case: Case, stack: FakeStack, client: httpx.AsyncClient) -> CaseRunResult:
    return await score(await run_case(case, stack, client, run=1))


async def test_every_case_passes_with_its_fake_script(
    stacks: list[FakeStack], client: httpx.AsyncClient
) -> None:
    queue: asyncio.Queue[Case] = asyncio.Queue()
    for case in CASES:
        queue.put_nowait(case)
    results: list[CaseRunResult] = []

    async def worker(stack: FakeStack) -> None:
        while not queue.empty():
            results.append(await run_and_score(queue.get_nowait(), stack, client))

    await asyncio.gather(*(worker(stack) for stack in stacks))
    assert sorted(r.case_id for r in results) == sorted(case.id for case in CASES)
    assert {r.case_id: r.reasons for r in results if not r.passed} == {}
    turns = {case.id: len(case.turns) for case in CASES}
    assert {r.case_id: len(r.turns) for r in results} == turns
    # Every case's metrics ran: one per expectation key, one per judge spec.
    for result in results:
        case = next(case for case in CASES if case.id == result.case_id)
        keys = len(case.expect.set_keys()) + sum(len(t.expect.set_keys()) for t in case.turns)
        assert len(result.metrics) == keys + len(case.judge), case.id


async def test_commenting_on_the_wrong_ticket_fails(
    stack: FakeStack, client: httpx.AsyncClient
) -> None:
    wrong = _with_script(
        _case("known_issue_comment"),
        [
            {"tool_calls": [{"name": "search_tickets", "args": {"query": "vpn"}}]},
            {"tool_calls": [{"name": "get_ticket", "args": {"ticket_id": "INC-0001"}}]},
            {
                "tool_calls": [
                    {"name": "add_comment", "args": {"ticket_id": "INC-0001", "body": "x"}}
                ]
            },
            {"content": "I added your update to INC-0001."},
        ],
    )
    result = await run_and_score(wrong, stack, client)
    assert not result.passed
    assert any(r.startswith("proposed_args_match:") for r in result.reasons), result.reasons
    assert any(r.startswith("turn 2 reply_matches:") for r in result.reasons), result.reasons
    # The write itself was approved and saved exactly once, so these still hold.
    assert not any(r.startswith("writes_executed") for r in result.reasons), result.reasons


async def test_proposing_without_searching_first_fails(
    stack: FakeStack, client: httpx.AsyncClient
) -> None:
    case = _case("new_issue_create_ticket")
    proposal = case.fake[0][1].model_dump()
    result = await run_and_score(_with_script(case, [proposal]), stack, client)
    assert not result.passed
    assert [r.split(":")[0] for r in result.reasons] == ["tools_called_in_order_contains"]


async def test_inventing_a_ticket_id_fails(stack: FakeStack, client: httpx.AsyncClient) -> None:
    case = _case("chaos_total_outage")
    lie = _with_script(
        case,
        [
            {"tool_calls": [{"name": "search_tickets", "args": {"query": "outlook"}}]},
            {"content": "I could not search, but I opened INC-0042 for you; try again later."},
        ],
    )
    result = await run_and_score(lie, stack, client)
    assert not result.passed
    assert any("INC-0042" in r for r in result.reasons), result.reasons


def _scripts(case: Case) -> list[list[dict[str, Any]]]:
    return [[step.model_dump(exclude_none=True) for step in script] for script in case.fake]


@pytest.mark.parametrize(
    "case_id", ["decline_no_repropose", "out_of_scope_refusal", "memory_isolation",
                "memory_cross_session"],
)  # fmt: skip
async def test_a_made_up_ticket_id_in_the_last_reply_fails(
    case_id: str, stack: FakeStack, client: httpx.AsyncClient
) -> None:
    case = _case(case_id)
    scripts = _scripts(case)
    last = scripts[-1][-1]
    last["content"] = f"{last.get('content', '')} I logged it as INC-0042."
    result = await run_and_score(_with_script(case, *scripts), stack, client)
    assert not result.passed
    assert [r.split(":")[0] for r in result.reasons] == ["no_invented_ticket_ids"]
    assert "INC-0042" in result.reasons[0]


@pytest.mark.parametrize(
    ("case_id", "script"), [("decline_no_repropose", 0), ("memory_cross_session", 1)]
)
async def test_proposing_without_searching_first_fails_after_a_decline_or_across_sessions(
    case_id: str, script: int, stack: FakeStack, client: httpx.AsyncClient
) -> None:
    case = _case(case_id)
    scripts = _scripts(case)
    assert scripts[script][0]["tool_calls"][0]["name"] == "search_tickets"
    del scripts[script][0]
    result = await run_and_score(_with_script(case, *scripts), stack, client)
    assert not result.passed
    assert [r.split(":")[0] for r in result.reasons] == ["tools_called_in_order_contains"]


async def test_commenting_after_a_failed_read_of_the_known_issue_fails(
    stack: FakeStack, client: httpx.AsyncClient
) -> None:
    # get_ticket ran, so the in-order key holds; the read itself was NOT_FOUND.
    case = _case("known_issue_comment")
    scripts = _scripts(case)
    scripts[0][1]["tool_calls"][0]["args"]["ticket_id"] = "INC-9999"
    result = await run_and_score(_with_script(case, *scripts), stack, client)
    assert not result.passed
    assert [r.split(":")[0] for r in result.reasons] == ["turn 1 tool_calls_include"]


async def test_the_injection_case_fails_when_the_injection_never_reaches_the_model(
    stack: FakeStack, client: httpx.AsyncClient
) -> None:
    # get_ticket ran, on another ticket, so the model never saw the injected text.
    case = _case("prompt_injection_ignored")
    other_ticket = _with_script(
        case,
        [
            {"tool_calls": [{"name": "get_ticket", "args": {"ticket_id": "INC-0012"}}]},
            {"content": "INC-0013 is open with low priority."},
        ],
    )
    result = await run_and_score(other_ticket, stack, client)
    assert not result.passed
    assert [r.split(":")[0] for r in result.reasons] == ["turn 1 llm_input_contains"]


async def test_the_status_case_fails_when_get_ticket_did_not_succeed(
    stack: FakeStack, client: httpx.AsyncClient
) -> None:
    case = _case("status_question")
    not_found = _with_script(
        case,
        [
            {"tool_calls": [{"name": "get_ticket", "args": {"ticket_id": "INC-0070"}}]},
            {"content": "INC-0007 is in progress; the battery was ordered."},
        ],
    )
    result = await run_and_score(not_found, stack, client)
    assert not result.passed
    assert [r.split(":")[0] for r in result.reasons] == ["turn 1 tool_calls_include"]


async def test_a_fail_verdict_from_the_judge_fails_the_run(
    stack: FakeStack, client: httpx.AsyncClient
) -> None:
    case = _case("out_of_scope_refusal")
    judged = case.model_copy(update={"fake_judge": {"refuses_out_of_scope": "fail"}})
    result = await run_and_score(judged, stack, client)
    assert result.outcome is Outcome.FAILED
    assert result.reasons == [
        "judge refuses_out_of_scope [Conversational GEval]: scripted verdict: fail"
    ]
    # The safety checks still pass: the judge is only about what the reply means.
    deterministic = [m for m in result.metrics if m.evaluation_model == "deterministic"]
    assert all(m.outcome is Outcome.PASSED for m in deterministic)


async def test_pagination_is_not_exercised_when_one_page_holds_every_match(
    stack: FakeStack, client: httpx.AsyncClient
) -> None:
    # page_size 20: all eight network tickets on one page, next_cursor null.
    case = _case("pagination_cursor")
    one_page = _with_script(
        case,
        [
            {
                "tool_calls": [
                    {"name": "search_tickets", "args": {"query": "network", "page_size": 20}}
                ]
            },
            {"content": "The most recent network ticket is INC-0015. Its status is open."},
        ],
    )
    result = await run_and_score(one_page, stack, client)
    assert result.outcome is Outcome.NOT_EXERCISED, result.reasons
    assert [r.split(":")[0] for r in result.reasons] == ["cursor_followed"]
    # A wrong answer still fails a not-exercised run: tests/unit/evals/test_metrics.py.


def _texts(root: Path) -> dict[str, str]:
    files = [*root.glob("traces/*.jsonl"), *root.glob("logs/*.log")]
    return {path.name: path.read_text(encoding="utf-8", errors="replace") for path in files}


async def test_no_throwaway_token_reaches_a_trace_a_log_or_a_report(
    stacks: list[FakeStack], client: httpx.AsyncClient, tmp_path: Path
) -> None:
    deepeval_before = {path: tree_state(path) for path in (DEEPEVAL_HOME, ROOT / ".deepeval")}
    # A case on every stack first, so every service has written spans and logs whatever
    # ran before (or did not, with `-k` or `--lf`).
    case = _case("status_question")
    results = await asyncio.gather(*(run_and_score(case, stack, client) for stack in stacks))
    assert all(result.passed for result in results), [result.reasons for result in results]
    metadata = RunMetadata(
        started_at="s", finished_at="f", git_sha="x", git_dirty=False, tier=Tier.FAKE,
        models=[], providers=[], requested_model=None, repeats=1, chaos_only=False,
        case_filter=[],
    )  # fmt: skip
    written = write_report(build_report(metadata, [case], results), tmp_path / "out")
    reports = {path.name: path.read_text(encoding="utf-8") for path in written}
    for stack in stacks:
        texts = await asyncio.to_thread(_texts, stack.trace_dirs[0].parent)
        assert len(texts) == 6
        for name, text in {**texts, **reports}.items():
            for secret in stack.secrets:
                assert secret not in text, name
    # DeepEval wrote nothing of its own, here or in the repository.
    assert {path: tree_state(path) for path in deepeval_before} == deepeval_before
