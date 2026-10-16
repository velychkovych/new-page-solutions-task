"""The deterministic eval tier in `make test`: every case passes with its fake script, and a
wrong script fails on the expectation it breaks.

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

from evals.cases import Case, load_cases
from evals.driver import run_case
from evals.fake_stack import FakeStack
from evals.results import CaseRunResult
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


async def test_every_case_passes_with_its_fake_script(
    stacks: list[FakeStack], client: httpx.AsyncClient
) -> None:
    queue: asyncio.Queue[Case] = asyncio.Queue()
    for case in CASES:
        queue.put_nowait(case)
    results: list[CaseRunResult] = []

    async def worker(stack: FakeStack) -> None:
        while not queue.empty():
            results.append(await run_case(queue.get_nowait(), stack, client, run=1))

    await asyncio.gather(*(worker(stack) for stack in stacks))
    assert sorted(r.case_id for r in results) == sorted(case.id for case in CASES)
    assert {r.case_id: r.reasons for r in results if not r.passed} == {}
    turns = {case.id: len(case.turns) for case in CASES}
    assert {r.case_id: len(r.turns) for r in results} == turns


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
    result = await run_case(wrong, stack, client, run=1)
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
    result = await run_case(_with_script(case, [proposal]), stack, client, run=1)
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
    result = await run_case(lie, stack, client, run=1)
    assert not result.passed
    assert any("INC-0042" in r for r in result.reasons), result.reasons


def _texts(root: Path) -> dict[str, str]:
    files = [*root.glob("traces/*.jsonl"), *root.glob("logs/*.log")]
    return {path.name: path.read_text(encoding="utf-8", errors="replace") for path in files}


async def test_no_throwaway_token_reaches_a_trace_or_a_log(stacks: list[FakeStack]) -> None:
    # Runs after the cases above (file order), so every service has written spans and logs.
    for stack in stacks:
        texts = await asyncio.to_thread(_texts, stack.trace_dirs[0].parent)
        assert len(texts) == 6
        for name, text in texts.items():
            for secret in stack.secrets:
                assert secret not in text, name
