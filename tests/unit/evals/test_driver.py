"""The driver against an in-memory stack and an httpx MockTransport agent (no processes)."""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from tests.unit.evals.conftest import (
    make_case,
    mock_span,
    pending,
    request_span,
    response,
    tool_call,
    tool_span,
)

from evals.chaos_profiles import CHAOS_OFF, MODERATE
from evals.driver import run_case
from evals.facts_command import FactsCommand, FactsResult
from evals.stack import Tier
from helpdesk.agent.fake_model import ScriptedMessage
from helpdesk.shared.api import ChatResponse
from helpdesk.shared.tickets import ChaosConfig, WritesReport
from helpdesk.telemetry.records import SpanRecord

AGENT_URL = "http://agent.test"


@dataclass
class MemoryStack:
    traces: Path
    calls: list[str] = field(default_factory=list)
    writes_total: int = 0
    stored: dict[str, dict[str, str]] = field(default_factory=dict)
    scripts: list[list[ScriptedMessage]] = field(default_factory=list)

    @property
    def tier(self) -> Tier:
        return Tier.FAKE

    @property
    def agent_url(self) -> str:
        return AGENT_URL

    @property
    def trace_dirs(self) -> tuple[Path, ...]:
        return (self.traces,)

    async def reset_mock(self) -> None:
        self.calls.append("reset")
        self.writes_total = 0

    async def set_chaos(self, config: ChaosConfig) -> None:
        self.calls.append(f"chaos:{'on' if config.enabled else 'off'}")

    async def writes(self) -> WritesReport:
        return WritesReport(
            total=self.writes_total, by_key={f"k{i}": 1 for i in range(self.writes_total)}
        )

    async def facts(self, command: FactsCommand) -> FactsResult:
        self.calls.append(
            f"facts:clear={len(command.clear)},set={sorted(command.set)},read={command.read}"
        )
        for user in command.clear:
            self.stored.pop(user, None)
        for user, values in command.set.items():
            self.stored[user] = {key.value: value for key, value in values.items()}
        return FactsResult(facts={user: self.stored.get(user, {}) for user in command.read})

    async def use_script(self, script: Sequence[ScriptedMessage]) -> None:
        self.scripts.append(list(script))


type Reply = Callable[[dict[str, Any]], tuple[ChatResponse, Sequence[SpanRecord]]]


def agent(
    stack: MemoryStack, replies: list[Reply], seen: list[dict[str, Any]]
) -> httpx.AsyncClient:
    """An agent that answers each /chat with the next reply and writes its spans."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        resp, spans = replies.pop(0)(body)
        stack.traces.mkdir(parents=True, exist_ok=True)
        with (stack.traces / "spans.jsonl").open("a", encoding="utf-8") as file:
            for span in spans:
                file.write(span.model_dump_json() + "\n")
        return httpx.Response(200, text=resp.model_dump_json())

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def trace_id(n: int) -> str:
    return f"{n:032x}"


def proposal(n: int, action_id: str = "act-1") -> Reply:
    def reply(_: dict[str, Any]) -> tuple[ChatResponse, Sequence[SpanRecord]]:
        tid = trace_id(n)
        resp = response(
            pending_action=pending(action_id),
            tool_calls=[tool_call("search_tickets")],
            trace_id=tid,
        )
        return resp, [
            request_span(tid),
            tool_span("search_tickets", trace_id=tid),
            mock_span(trace_id=tid),
        ]

    return reply


def approved_write(stack: MemoryStack, n: int) -> Reply:
    def reply(_: dict[str, Any]) -> tuple[ChatResponse, Sequence[SpanRecord]]:
        stack.writes_total += 1
        tid = trace_id(n)
        resp = response(
            reply="Created INC-0016.", tool_calls=[tool_call("create_ticket")], trace_id=tid
        )
        result = '{"id":"INC-0016"}'
        return resp, [
            request_span(tid),
            tool_span("create_ticket", result=result, trace_id=tid),
            mock_span("POST /tickets", trace_id=tid),
        ]

    return reply


def plain(n: int, *, spans: bool = True, reply_text: str = "Hello.") -> Reply:
    def reply(_: dict[str, Any]) -> tuple[ChatResponse, Sequence[SpanRecord]]:
        tid = trace_id(n)
        return response(reply=reply_text, trace_id=tid), [request_span(tid)] if spans else []

    return reply


APPROVAL_CASE: dict[str, Any] = {
    "setup": {"chaos": "moderate", "facts": {"team": "finance"}},
    "turns": [{"message": "My printer jams."}, {"decision": "approve"}],
    "fake": [[{"content": "scripted"}]],
    "expect": {
        "writes_executed": 1,
        "no_write_before_approval": True,
        "max_proposals": 1,
        "no_invented_ticket_ids": True,
        "final_status": "ok",
    },
}


async def test_an_approval_case_passes_and_sends_the_pending_action_id(tmp_path: Path) -> None:
    stack = MemoryStack(tmp_path / "traces")
    seen: list[dict[str, Any]] = []
    case = make_case(**APPROVAL_CASE)
    async with agent(stack, [proposal(1, "act-42"), approved_write(stack, 2)], seen) as client:
        result = await run_case(case, stack, client, run=1, trace_wait_s=1)
    assert result.reasons == []
    assert result.passed
    assert seen[0]["message"] == "My printer jams."
    assert seen[1]["decision"] == {"action_id": "act-42", "approve": True}
    assert seen[0]["session_id"] == seen[1]["session_id"]
    assert [t.status.value for t in result.turns] == ["awaiting_approval", "ok"]
    # Only the message turn starts a script; the decision continues it.
    assert stack.scripts == [[ScriptedMessage(content="scripted")]]


async def test_setup_order_and_chaos_off_before_the_final_reads(tmp_path: Path) -> None:
    stack = MemoryStack(tmp_path / "traces")
    case = make_case(**APPROVAL_CASE)
    async with agent(stack, [proposal(1), approved_write(stack, 2)], []) as client:
        await run_case(case, stack, client, run=1, trace_wait_s=1)
    assert stack.calls == [
        "chaos:off",
        "reset",
        "facts:clear=3,set=['alice'],read=[]",
        "chaos:on",
        "chaos:off",
    ]
    assert MODERATE.enabled
    assert not CHAOS_OFF.enabled


async def test_a_write_during_the_proposal_turn_fails_the_case(tmp_path: Path) -> None:
    stack = MemoryStack(tmp_path / "traces")

    def sneaky(body: dict[str, Any]) -> tuple[ChatResponse, Sequence[SpanRecord]]:
        stack.writes_total += 1
        return proposal(1)(body)

    case = make_case(**APPROVAL_CASE)
    async with agent(stack, [sneaky, approved_write(stack, 2)], []) as client:
        result = await run_case(case, stack, client, run=1, trace_wait_s=1)
    assert not result.passed
    assert any("/admin/writes went 0 -> 1" in reason for reason in result.reasons)


async def test_a_decision_without_a_pending_action_fails_the_case(tmp_path: Path) -> None:
    stack = MemoryStack(tmp_path / "traces")
    case = make_case(**APPROVAL_CASE)
    async with agent(stack, [plain(1)], []) as client:
        result = await run_case(case, stack, client, run=1, trace_wait_s=1)
    assert result.reasons == ["turn 2: a approve turn, but no action is pending"]


async def test_an_incomplete_trace_fails_the_case_before_any_check_runs(tmp_path: Path) -> None:
    stack = MemoryStack(tmp_path / "traces")
    # With no spans at all, "no tool calls" and "no writes" would hold vacuously.
    case = make_case(expect={"no_tool_calls": True, "writes_executed": 0})
    async with agent(stack, [plain(1, spans=False)], []) as client:
        result = await run_case(case, stack, client, run=1, trace_wait_s=0.05)
    assert not result.passed
    assert len(result.reasons) == 1
    assert result.reasons[0].startswith(f"trace incomplete: {trace_id(1)}")
    assert "no agent.request span" in result.reasons[0]


async def test_an_http_error_fails_the_case(tmp_path: Path) -> None:
    stack = MemoryStack(tmp_path / "traces")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_case(make_case(), stack, client, run=1, trace_wait_s=0.05)
    assert result.reasons == ["turn 1: HTTP 500: Internal Server Error"]


async def test_each_run_and_each_session_label_gets_a_fresh_session_id(tmp_path: Path) -> None:
    stack = MemoryStack(tmp_path / "traces")
    seen: list[dict[str, Any]] = []
    case = make_case(
        turns=[{"message": "a", "session": "one"}, {"message": "b", "session": "two"}],
        fake=[[{"content": "x"}], [{"content": "x"}]],
    )
    replies = [plain(1), plain(2), plain(3), plain(4)]
    async with agent(stack, replies, seen) as client:
        await run_case(case, stack, client, run=1, trace_wait_s=1)
        await run_case(case, stack, client, run=2, trace_wait_s=1)
    sessions = [body["session_id"] for body in seen]
    assert len(set(sessions)) == 4
    assert all(s.startswith("ev-unit_case-") for s in sessions)


async def test_facts_are_read_for_the_fact_checks(tmp_path: Path) -> None:
    stack = MemoryStack(tmp_path / "traces")
    case = make_case(
        setup={"facts": {"team": "finance"}},
        expect={"facts": {"alice": {"team": "finance"}}, "no_facts_for": ["bob"]},
    )
    async with agent(stack, [plain(1)], []) as client:
        result = await run_case(case, stack, client, run=1, trace_wait_s=1)
    assert result.passed, result.reasons
    assert stack.calls[-1] == "facts:clear=0,set=[],read=['alice', 'bob']"


@pytest.mark.parametrize("reply_text", ["I opened INC-0042 for you."])
async def test_tokens_and_failures_are_reported(tmp_path: Path, reply_text: str) -> None:
    stack = MemoryStack(tmp_path / "traces")
    case = make_case(expect={"no_invented_ticket_ids": True})
    async with agent(stack, [plain(1, reply_text=reply_text)], []) as client:
        result = await run_case(case, stack, client, run=3, trace_wait_s=1)
    assert result.run == 3
    assert not result.passed
    assert "INC-0042" in result.reasons[0]
