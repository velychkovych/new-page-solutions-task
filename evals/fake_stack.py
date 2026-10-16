"""The deterministic tier: mock, MCP server and agent as local processes, fake model.

The services run as `python -m helpdesk.{mock,mcp_server,agent}` on free loopback ports,
with throwaway tokens generated here and an environment holding only what they need (no
.env, no model key, no OTLP endpoint). Databases and spans live in a temporary directory.

The fake model reads its script once at startup, so the agent is restarted whenever the
next turn needs a different script. Checkpoints and facts are on disk, so a conversation
carries on across the restart.
"""

import asyncio
import os
import secrets
import socket
import sys
import time
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import IO

import httpx
from pydantic import TypeAdapter

from evals import facts_command
from evals.facts_command import FactsCommand, FactsResult
from evals.stack import StackError, Tier, checked, mask_secrets, run_command
from helpdesk.agent.fake_model import ScriptedMessage
from helpdesk.mock.seed import seed_database
from helpdesk.shared.tickets import ADMIN_TOKEN_HEADER, ChaosConfig, WritesReport

STARTUP_TIMEOUT_S = 30.0
STOP_TIMEOUT_S = 20.0
_KEEP_ENV = ("PATH", "HOME", "SYSTEMROOT", "TMPDIR")
_SCRIPT = TypeAdapter(list[ScriptedMessage])
_FACTS_SOURCE = Path(facts_command.__file__).read_text(encoding="utf-8")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _Service:
    def __init__(self, name: str, port: int, process: asyncio.subprocess.Process) -> None:
        self.name = name
        self.url = f"http://127.0.0.1:{port}"
        self.process = process


class FakeStack:
    def __init__(self, workdir: Path) -> None:
        self._dir = workdir
        self._api_token = secrets.token_urlsafe(24)
        self._admin_token = secrets.token_urlsafe(24)
        self._services: dict[str, _Service] = {}
        self._logs = ExitStack()
        self._script: list[ScriptedMessage] | None = None
        self._scripts_written = 0
        self._client = httpx.AsyncClient(timeout=10)
        self._mock_db = workdir / "data" / "mock.sqlite"
        self._mcp_port = 0

    @property
    def tier(self) -> Tier:
        return Tier.FAKE

    @property
    def agent_url(self) -> str:
        return self._services["agent"].url if "agent" in self._services else ""

    @property
    def trace_dirs(self) -> tuple[Path, ...]:
        return (self._dir / "traces",)

    @property
    def secrets(self) -> tuple[str, str]:
        """The throwaway tokens, so callers can prove none of them leaked."""
        return self._api_token, self._admin_token

    # --- lifecycle -----------------------------------------------------------------------

    def _env(self, **extra: str) -> dict[str, str]:
        base = {key: os.environ[key] for key in _KEEP_ENV if key in os.environ}
        return {**base, "TRACES_DIR": str(self._dir / "traces"), "LOG_LEVEL": "INFO", **extra}

    async def _spawn(self, name: str, module: str, port: int, **env: str) -> _Service:
        (self._dir / "logs").mkdir(parents=True, exist_ok=True)
        log: IO[bytes] = self._logs.enter_context((self._dir / "logs" / f"{name}.log").open("ab"))
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            module,
            cwd=self._dir,  # no .env here
            env=self._env(**env),
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
        )
        service = _Service(name, port, process)
        self._services[name] = service
        await self._wait_healthy(service)
        return service

    def _log_tail(self, name: str) -> str:
        path = self._dir / "logs" / f"{name}.log"
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        return mask_secrets(text[-2000:], self.secrets)

    async def _wait_healthy(self, service: _Service) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while True:
            if service.process.returncode is not None:
                raise StackError(f"{service.name} exited early:\n{self._log_tail(service.name)}")
            try:
                if (
                    await self._client.get(f"{service.url}/health", timeout=0.5)
                ).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                raise StackError(f"{service.name} did not become healthy")
            await asyncio.sleep(0.05)

    async def _stop(self, name: str) -> None:
        service = self._services.pop(name, None)
        if service is None or service.process.returncode is not None:
            return
        # SIGTERM: uvicorn runs the lifespan shutdown, which flushes the spans.
        service.process.terminate()
        try:
            await asyncio.wait_for(service.process.wait(), STOP_TIMEOUT_S)
        except TimeoutError:
            service.process.kill()
            await service.process.wait()

    async def start(self) -> None:
        mock_port, self._mcp_port = free_port(), free_port()
        await self._spawn(
            "mock",
            "helpdesk.mock",
            mock_port,
            MOCK_HOST="127.0.0.1",
            MOCK_PORT=str(mock_port),
            MOCK_DB_PATH=str(self._mock_db),
            MOCK_API_TOKEN=self._api_token,
            MOCK_ADMIN_TOKEN=self._admin_token,
        )
        await self._spawn(
            "mcp",
            "helpdesk.mcp_server",
            self._mcp_port,
            MCP_HOST="127.0.0.1",
            MCP_PORT=str(self._mcp_port),
            MOCK_URL=self._services["mock"].url,
            MOCK_API_TOKEN=self._api_token,
        )

    async def close(self) -> None:
        for name in ("agent", "mcp", "mock"):
            await self._stop(name)
        await self._client.aclose()
        self._logs.close()

    async def use_script(self, script: Sequence[ScriptedMessage]) -> None:
        wanted = list(script)
        if wanted == self._script and "agent" in self._services:
            return
        await self._stop("agent")
        self._scripts_written += 1
        path = self._dir / "scripts" / f"script-{self._scripts_written}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_SCRIPT.dump_json(wanted))
        port = free_port()
        await self._spawn(
            "agent",
            "helpdesk.agent",
            port,
            AGENT_HOST="127.0.0.1",
            AGENT_PORT=str(port),
            MCP_URL=f"http://127.0.0.1:{self._mcp_port}/mcp",
            MODEL_PROVIDER="fake",
            FAKE_MODEL_SCRIPT=str(path),
            CHECKPOINT_DB_PATH=str(self._dir / "data" / "checkpoints.sqlite"),
            STORE_DB_PATH=str(self._dir / "data" / "store.sqlite"),
        )
        self._script = wanted

    # --- admin -----------------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_token}",
            ADMIN_TOKEN_HEADER: self._admin_token,
        }

    async def _admin(self, method: str, path: str, body: str | None = None) -> object:
        url = self._services["mock"].url + path
        try:
            response = await self._client.request(
                method,
                url,
                content=body,
                headers={**self._headers(), "Content-Type": "application/json"},
            )
        except httpx.TransportError as exc:
            raise StackError(f"mock {method} {path}: {type(exc).__name__}") from exc
        if response.status_code != 200:
            raise StackError(f"mock {method} {path} returned {response.status_code}")
        body_json: object = response.json()
        return body_json

    async def reset_mock(self) -> None:
        await seed_database(self._mock_db, reset=True)

    async def set_chaos(self, config: ChaosConfig) -> None:
        await self._admin("POST", "/admin/chaos", config.model_dump_json())

    async def writes(self) -> WritesReport:
        return WritesReport.model_validate(await self._admin("GET", "/admin/writes"))

    async def facts(self, command: FactsCommand) -> FactsResult:
        result = await run_command(
            [sys.executable, "-c", _FACTS_SOURCE, command.model_dump_json()],
            cwd=self._dir,
            env=self._env(STORE_DB_PATH=str(self._dir / "data" / "store.sqlite")),
        )
        return FactsResult.model_validate_json(checked(result, "facts command"))
