"""How the runner reaches the agent, the mock's admin API and the facts store, per tier.

- `LiveStack` (evals.live_stack): the compose service `agent-eval` with the real model;
  mock admin and facts through `docker compose exec`, so no token reaches this process.
- `FakeStack` (evals.fake_stack): local processes with the scripted fake model.
"""

import asyncio
import contextlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from evals.facts_command import FactsCommand, FactsResult
from helpdesk.agent.fake_model import ScriptedMessage
from helpdesk.shared.tickets import ChaosConfig, WritesReport

ROOT = Path(__file__).resolve().parents[1]
COMMAND_TIMEOUT_S = 120.0
# Anything shaped like a credential is masked before a command's output is shown. The
# services never print one; this is a second guard for error output we pass on.
_SECRET_LIKE = re.compile(r"sk-ant-[A-Za-z0-9_-]+|[A-Za-z0-9_-]{32,}")


class Tier(StrEnum):
    LIVE = "live"
    FAKE = "fake"


class StackError(RuntimeError):
    """The stack cannot be used: not running, not healthy, or a command failed (exit 2)."""


def mask_secrets(text: str, secrets: Sequence[str] = ()) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return _SECRET_LIKE.sub("***", text)


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


async def _kill(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        await process.wait()


async def run_command(
    args: Sequence[str],
    *,
    cwd: Path = ROOT,
    env: Mapping[str, str] | None = None,
    timeout_s: float = COMMAND_TIMEOUT_S,
) -> CommandResult:
    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            env=dict(env) if env is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise StackError(f"cannot run {args[0]}: {exc.strerror}") from exc
    try:
        out, err = await asyncio.wait_for(process.communicate(), timeout_s)
    except TimeoutError as exc:
        await _kill(process)
        raise StackError(f"{' '.join(args[:4])} timed out after {timeout_s:.0f} s") from exc
    except asyncio.CancelledError:
        # The runner is going away (SIGTERM, Ctrl-C); the command must not outlive it.
        await _kill(process)
        raise
    return CommandResult(
        returncode=process.returncode if process.returncode is not None else -1,
        stdout=out.decode("utf-8", errors="replace"),
        stderr=err.decode("utf-8", errors="replace"),
    )


def checked(result: CommandResult, what: str) -> str:
    if result.returncode != 0:
        tail = mask_secrets(result.stderr.strip()[-1500:] or result.stdout.strip()[-1500:])
        raise StackError(f"{what} failed (exit {result.returncode}): {tail}")
    return result.stdout


class Stack(Protocol):
    @property
    def tier(self) -> Tier: ...

    @property
    def agent_url(self) -> str: ...

    @property
    def trace_dirs(self) -> tuple[Path, ...]: ...

    async def reset_mock(self) -> None:
        """Back to the seed tickets, with no idempotency records and no write counts."""

    async def set_chaos(self, config: ChaosConfig) -> None: ...

    async def writes(self) -> WritesReport: ...

    async def facts(self, command: FactsCommand) -> FactsResult: ...

    async def use_script(self, script: Sequence[ScriptedMessage]) -> None:
        """The model script for the next turns (fake tier); the live tier ignores it."""
