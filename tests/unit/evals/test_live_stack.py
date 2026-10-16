"""How the live tier starts agent-eval: never by starting or recreating mock and mcp.

docker is faked: every command the stack runs is recorded and answered here.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from evals import live_stack
from evals.live_stack import SERVICE, SHARED_SERVICES, LiveStack, unhealthy_services
from evals.stack import CommandResult, StackError


def _row(service: str, state: str = "running", health: str = "healthy") -> dict[str, str]:
    return {"Service": service, "State": state, "Health": health, "Name": f"x-{service}-1"}


def _lines(*rows: dict[str, str]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)


def test_healthy_services_one_object_per_line() -> None:
    assert unhealthy_services(_lines(_row("mock"), _row("mcp")), SHARED_SERVICES) == []


def test_healthy_services_as_one_array() -> None:
    # docker compose before v2.21 printed an array.
    assert unhealthy_services(json.dumps([_row("mock"), _row("mcp")]), SHARED_SERVICES) == []


@pytest.mark.parametrize(
    ("output", "down"),
    [
        ("", ["mock", "mcp"]),
        (_lines(_row("mock")), ["mcp"]),
        (_lines(_row("mock"), _row("mcp", health="starting")), ["mcp"]),
        (_lines(_row("mock", state="exited", health=""), _row("mcp")), ["mock"]),
    ],
)
def test_unhealthy_or_missing_services(output: str, down: list[str]) -> None:
    assert unhealthy_services(output, SHARED_SERVICES) == down


class _Docker:
    def __init__(self, ps_output: str) -> None:
        self.ps_output = ps_output
        self.calls: list[list[str]] = []

    async def __call__(
        self, args: Sequence[str], *, env: Mapping[str, str] | None = None, **_: Any
    ) -> CommandResult:
        self.calls.append(list(args))
        if "ps" in args:
            return CommandResult(0, self.ps_output, "")
        return CommandResult(0, "", "")


async def _start(monkeypatch: pytest.MonkeyPatch, ps_output: str) -> _Docker:
    docker = _Docker(ps_output)
    monkeypatch.setattr(live_stack, "run_command", docker)

    async def healthy(_: LiveStack) -> None:
        return None

    monkeypatch.setattr(LiveStack, "_wait_healthy", healthy)
    stack = LiveStack()
    try:
        await stack.start(None)
    finally:
        await stack.close()
    return docker


async def test_agent_eval_is_started_without_its_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = await _start(monkeypatch, _lines(_row("mock"), _row("mcp")))
    (up,) = [call for call in docker.calls if "up" in call]
    assert "--no-deps" in up
    assert up[-1] == SERVICE
    assert not any(name in up for name in SHARED_SERVICES)


async def test_a_stack_that_is_not_up_is_an_error_naming_make_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(StackError, match=r"mcp not running and healthy.*`make up`"):
        await _start(monkeypatch, _lines(_row("mock")))


async def test_nothing_is_started_when_the_stack_is_not_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = _Docker("")
    monkeypatch.setattr(live_stack, "run_command", docker)
    with pytest.raises(StackError):
        await LiveStack().start(None)
    assert not any("up" in call for call in docker.calls)
