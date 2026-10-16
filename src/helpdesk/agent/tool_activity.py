"""What a running tool call tells a streaming client (CONTRACTS §4 `tool_start`, `status`).

The tools and execute_write nodes report through LangGraph's custom stream: that the call
started, the MCP server's retry progress, and, once, that it is slow. A run without
"custom" in its stream modes (every /chat run) gets LangGraph's no-op writer, so these
signals cost nothing there and change nothing in the graph's state.
"""

import asyncio
import contextvars
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from langgraph.config import get_stream_writer

from helpdesk.agent.tool_provider import ProgressCallback


@dataclass(frozen=True, slots=True)
class ToolStarted:
    """The call is about to reach the tool provider."""

    call_id: str
    name: str


@dataclass(frozen=True, slots=True)
class ToolStatus:
    message: str


def progress_message(name: str, message: str) -> str:
    # Parallel reads can each report progress, so the tool is named.
    return f"{name}: {message}"


def slow_tool_message(name: str) -> str:
    return f"Still working on {name}…"


def _ignore(_: Any) -> None:
    return None


def _writer() -> Callable[[Any], None]:
    try:
        return get_stream_writer()
    except RuntimeError, KeyError:
        # Outside a graph run (a node called directly): nobody is listening.
        return _ignore


@contextmanager
def tool_activity(call_id: str, name: str, *, slow_after_s: float) -> Iterator[ProgressCallback]:
    """Signal the start now and a slow notice once after `slow_after_s`; yield the callback
    that forwards the provider's progress messages for this call.

    The writer reads the run config from context variables, while progress arrives on the
    MCP session's receive task and the notice on a loop timer. So every signal is sent in
    a copy of the node's context, whichever task delivers it.
    """
    write = _writer()
    context = contextvars.copy_context()

    def emit(signal: ToolStarted | ToolStatus) -> None:
        context.run(write, signal)

    emit(ToolStarted(call_id=call_id, name=name))
    timer = asyncio.get_running_loop().call_later(
        slow_after_s, emit, ToolStatus(slow_tool_message(name))
    )

    async def on_progress(message: str) -> None:
        emit(ToolStatus(progress_message(name, message)))

    try:
        yield on_progress
    finally:
        timer.cancel()
