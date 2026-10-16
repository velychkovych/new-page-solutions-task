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
from helpdesk.shared.tickets import ChaosConfig, WritesReport

SERVICE = "agent-eval"
PROFILE = "eval"
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
        result = await run_command(
            compose_args("up", "-d", "--wait", SERVICE), env=up_env(model), timeout_s=UP_TIMEOUT_S
        )
        checked(result, f"docker compose up {SERVICE} (is the stack built? run `make up`)")
        await self._wait_healthy()

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

    async def _mock(self, method: str, path: str, body: str = "") -> object:
        out = checked(
            await run_command(
                compose_args("exec", "-T", "mock", "python", "-c", _MOCK_SOURCE, method, path, body)
            ),
            f"mock {method} {path}",
        )
        reply = json.loads(out)
        if reply.get("status") != 200:
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
