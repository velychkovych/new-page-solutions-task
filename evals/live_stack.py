"""The live tier: compose service `agent-eval` (profile `eval`) with the real model.

Its own container, port (127.0.0.1:8082), data volume and span file
(traces/eval/agent.jsonl), so resetting facts for the seed users or switching `--model`
never touches the browser agent. The mock publishes no port: its admin API is called
from inside the mock container, and facts are set and read inside `agent-eval`, so the
tokens never reach this process (CONTRACTS §7).
"""

import asyncio
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path

import httpx

from evals import facts_command, mock_command
from evals.facts_command import FactsCommand, FactsResult
from evals.stack import ROOT, StackError, Tier, checked, run_command
from helpdesk.agent.fake_model import ScriptedMessage
from helpdesk.shared.tickets import ChaosConfig, CreateTicketRequest, Ticket, WritesReport

SERVICE = "agent-eval"
PROFILE = "eval"
# Shared with the browser agent and started by `make up`, maybe with other settings
# (`make up OBS=1`): the runner never starts or recreates them.
SHARED_SERVICES = ("mock", "mcp")
AGENT_URL = "http://127.0.0.1:8082"
TRACE_DIRS = (ROOT / "traces", ROOT / "traces" / "eval")
HEALTH_TIMEOUT_S = 60.0
UP_TIMEOUT_S = 300.0

_MOCK_SOURCE = Path(mock_command.__file__).read_text(encoding="utf-8")
_FACTS_SOURCE = Path(facts_command.__file__).read_text(encoding="utf-8")


def compose_args(*args: str) -> list[str]:
    return ["docker", "compose", "--profile", PROFILE, *args]


def up_env(model: str | None) -> dict[str, str]:
    """Compose interpolates MODEL_NAME from here before .env, so `--model` recreates the
    container with that model, and running without it switches back to the default."""
    env = dict(os.environ)
    if model is not None:
        env["MODEL_NAME"] = model
    return env


def unhealthy_services(ps_json: str, wanted: Sequence[str]) -> list[str]:
    """The services of `wanted` that `docker compose ps --format json` does not show as
    running and healthy. Compose prints one object per line (an array before v2.21)."""
    text = ps_json.strip()
    rows: list[object] = []
    if text.startswith("["):
        parsed = json.loads(text)
        rows = parsed if isinstance(parsed, list) else []
    elif text:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    healthy = {
        row.get("Service")
        for row in rows
        if isinstance(row, dict)
        and row.get("State") == "running"
        and row.get("Health") == "healthy"
    }
    return [name for name in wanted if name not in healthy]


class LiveStack:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=5)

    @property
    def tier(self) -> Tier:
        return Tier.LIVE

    @property
    def agent_url(self) -> str:
        return AGENT_URL

    @property
    def trace_dirs(self) -> tuple[Path, ...]:
        return TRACE_DIRS

    async def start(self, model: str | None) -> None:
        await self._check_shared_services()
        # --no-deps: `up` would otherwise recreate mock and mcp whenever their settings
        # differ from how `make up` started them, under the browser agent's feet.
        result = await run_command(
            compose_args("up", "-d", "--wait", "--no-deps", SERVICE),
            env=up_env(model),
            timeout_s=UP_TIMEOUT_S,
        )
        checked(result, f"docker compose up {SERVICE} (is the stack built? run `make up`)")
        await self._wait_healthy()

    async def _check_shared_services(self) -> None:
        out = checked(
            await run_command(compose_args("ps", "--format", "json", *SHARED_SERVICES)),
            "docker compose ps",
        )
        if down := unhealthy_services(out, SHARED_SERVICES):
            raise StackError(
                f"{' and '.join(down)} not running and healthy; start the stack with"
                " `make up` first (the evals never start or recreate it)"
            )

    async def close(self) -> None:
        await self._client.aclose()

    async def _wait_healthy(self) -> None:
        deadline = time.monotonic() + HEALTH_TIMEOUT_S
        while True:
            try:
                if (await self._client.get(f"{AGENT_URL}/health")).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                raise StackError(f"{SERVICE} is not healthy at {AGENT_URL}")
            await asyncio.sleep(0.5)

    async def _mock(
        self, method: str, path: str, body: str = "", *, key: str = "", expected: int = 200
    ) -> object:
        command = ("python", "-c", _MOCK_SOURCE, method, path, body, *((key,) if key else ()))
        out = checked(
            await run_command(compose_args("exec", "-T", "mock", *command)),
            f"mock {method} {path}",
        )
        reply = json.loads(out)
        if reply.get("status") != expected:
            raise StackError(f"mock {method} {path} returned {reply.get('status')}")
        return reply.get("body")

    async def reset_mock(self) -> None:
        checked(
            await run_command(
                compose_args("exec", "-T", "mock", "python", "-m", "helpdesk.mock.seed", "--reset")
            ),
            "mock reset",
        )

    async def set_chaos(self, config: ChaosConfig) -> None:
        await self._mock("POST", "/admin/chaos", config.model_dump_json())

    async def create_ticket(self, ticket: CreateTicketRequest, idempotency_key: str) -> Ticket:
        body = ticket.model_dump_json()
        created = await self._mock("POST", "/tickets", body, key=idempotency_key, expected=201)
        return Ticket.model_validate(created)

    async def writes(self) -> WritesReport:
        return WritesReport.model_validate(await self._mock("GET", "/admin/writes"))

    async def facts(self, command: FactsCommand) -> FactsResult:
        out = checked(
            await run_command(
                compose_args(
                    "exec", "-T", SERVICE, "python", "-c", _FACTS_SOURCE, command.model_dump_json()
                )
            ),
            "facts command",
        )
        return FactsResult.model_validate_json(out)

    async def use_script(self, script: Sequence[ScriptedMessage]) -> None:
        return None
