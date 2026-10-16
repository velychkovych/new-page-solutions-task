"""A turn's spans from the JSONL trace files, and the guard that they are all there.

The checks read traces, so a missing span must never look like "no write happened".
Before a turn is checked, its trace must hold the agent's request span, one mcp.tool span
for every tool call in the response that reached the MCP server, and one mock span for
every attempt the MCP server got an HTTP status for. Each service flushes its JSONL every
200 ms, so the runner polls until that holds, within a bound.
"""

import asyncio
import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, JsonValue

from helpdesk.shared import trace_attributes as ta
from helpdesk.shared.api import ChatResponse
from helpdesk.shared.tools import ToolName
from helpdesk.telemetry.openinference_keys import (
    LLM_MODEL_NAME,
    LLM_TOKENS_CACHE_READ,
    LLM_TOKENS_CACHE_WRITE,
    LLM_TOKENS_COMPLETION,
    LLM_TOKENS_PROMPT,
    SPAN_KIND,
    SPAN_KIND_LLM,
)
from helpdesk.telemetry.records import SpanEventRecord, SpanRecord
from helpdesk.telemetry.show_trace import load_trace

MCP_TOOLS: frozenset[str] = frozenset(ToolName)
LLM_PROVIDER = "llm.provider"
_LLM_INPUT_PREFIX = "llm.input_messages."
_ROLE_SUFFIX = ".message.role"
_CONTENT_SUFFIX = ".message.content"
AGENT_SERVICE = "agent"
MOCK_SERVICE = "mock"

TRACE_WAIT_S = 15.0
TRACE_POLL_S = 0.2


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input=self.input + other.input,
            output=self.output + other.output,
            cache_read=self.cache_read + other.cache_read,
            cache_write=self.cache_write + other.cache_write,
        )

    @property
    def total(self) -> int:
        return self.input + self.output


def _int(value: JsonValue | None) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _str(value: JsonValue | None) -> str:
    return value if isinstance(value, str) else ""


# --- single spans ------------------------------------------------------------------------


def is_llm(span: SpanRecord) -> bool:
    return span.attributes.get(SPAN_KIND) == SPAN_KIND_LLM


def is_tool(span: SpanRecord) -> bool:
    return span.name == ta.SPAN_TOOL


def tool_name(span: SpanRecord) -> str:
    return _str(span.attributes.get(ta.TOOL_NAME))


def tool_args(span: SpanRecord) -> dict[str, JsonValue]:
    try:
        args = json.loads(_str(span.attributes.get(ta.TOOL_ARGS)) or "{}")
    except json.JSONDecodeError:
        return {}
    return args if isinstance(args, dict) else {}


def tool_result(span: SpanRecord) -> str:
    return _str(span.attributes.get(ta.TOOL_RESULT))


def tool_error_code(span: SpanRecord) -> str | None:
    return _str(span.attributes.get(ta.TOOL_ERROR_CODE)) or None


def tool_attempts(span: SpanRecord) -> int:
    return _int(span.attributes.get(ta.TOOL_ATTEMPTS))


def retry_events(span: SpanRecord) -> list[SpanEventRecord]:
    return [event for event in span.events if event.name == ta.EVENT_RETRY]


def attempt_status(event: SpanEventRecord) -> int | None:
    """The HTTP status of one attempt; None for a timeout or connect error."""
    value = event.attributes.get(ta.RETRY_HTTP_STATUS)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def llm_input_text(span: SpanRecord) -> str:
    """Every input message the LLM span recorded (system prompt, history, tool results)."""
    return "\n".join(
        value
        for key, value in span.attributes.items()
        if key.startswith(_LLM_INPUT_PREFIX) and isinstance(value, str)
    )


def llm_tool_results(span: SpanRecord) -> list[str]:
    """The contents of the input messages whose role is `tool`."""
    results: list[str] = []
    for key, value in span.attributes.items():
        if key.startswith(_LLM_INPUT_PREFIX) and key.endswith(_ROLE_SUFFIX) and value == "tool":
            prefix = key.removesuffix(_ROLE_SUFFIX)
            results.append(_str(span.attributes.get(prefix + _CONTENT_SUFFIX)))
    return results


def llm_tokens(span: SpanRecord) -> TokenUsage:
    attrs = span.attributes
    return TokenUsage(
        input=_int(attrs.get(LLM_TOKENS_PROMPT)),
        output=_int(attrs.get(LLM_TOKENS_COMPLETION)),
        cache_read=_int(attrs.get(LLM_TOKENS_CACHE_READ)),
        cache_write=_int(attrs.get(LLM_TOKENS_CACHE_WRITE)),
    )


# --- one turn's trace ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TurnTrace:
    trace_id: str
    spans: tuple[SpanRecord, ...]

    def _sorted(self, spans: Iterable[SpanRecord]) -> list[SpanRecord]:
        return sorted(spans, key=lambda span: (span.start_time, span.span_id))

    @property
    def request_spans(self) -> list[SpanRecord]:
        return [s for s in self.spans if s.name == ta.SPAN_REQUEST and s.service == AGENT_SERVICE]

    @property
    def tool_spans(self) -> list[SpanRecord]:
        return self._sorted(span for span in self.spans if is_tool(span))

    @property
    def llm_spans(self) -> list[SpanRecord]:
        return self._sorted(span for span in self.spans if is_llm(span))

    @property
    def mock_spans(self) -> list[SpanRecord]:
        return self._sorted(span for span in self.spans if span.service == MOCK_SERVICE)

    @property
    def tokens(self) -> TokenUsage:
        return sum((llm_tokens(span) for span in self.llm_spans), TokenUsage())

    @property
    def models(self) -> set[str]:
        return {
            name for span in self.llm_spans if (name := _str(span.attributes.get(LLM_MODEL_NAME)))
        }

    @property
    def providers(self) -> set[str]:
        return {
            name for span in self.llm_spans if (name := _str(span.attributes.get(LLM_PROVIDER)))
        }


def calls_reaching_mcp(response: ChatResponse) -> int:
    """Response tool calls that ran against the MCP server.

    A call rejected before it ran (router, propose, unparseable) is listed with 0 ms, and
    remember_fact runs inside the agent: neither has an mcp.tool span.
    """
    return sum(call.name in MCP_TOOLS and call.latency_ms > 0 for call in response.tool_calls)


def missing_parts(trace: TurnTrace, response: ChatResponse) -> list[str]:
    """What the trace still lacks before the checks may read it; empty when complete."""
    problems: list[str] = []
    if not trace.request_spans:
        problems.append(f"no {ta.SPAN_REQUEST} span")
    tools = trace.tool_spans
    expected_tools = calls_reaching_mcp(response)
    if len(tools) < expected_tools:
        problems.append(f"{len(tools)} of {expected_tools} {ta.SPAN_TOOL} spans")
    # Every attempt with an HTTP status reached the mock, so the mock has a span for it.
    attempts = sum(
        attempt_status(event) is not None for span in tools for event in retry_events(span)
    )
    if len(trace.mock_spans) < attempts:
        problems.append(f"{len(trace.mock_spans)} of {attempts} mock spans")
    return problems


def read_trace(trace_dirs: Sequence[Path], trace_id: str) -> TurnTrace:
    """The trace's spans from every `*.jsonl` file in `trace_dirs`, without duplicates."""
    spans: dict[str, SpanRecord] = {}
    for directory in trace_dirs:
        for span in load_trace(directory, trace_id).spans:
            spans[span.span_id] = span
    return TurnTrace(trace_id=trace_id, spans=tuple(spans.values()))


class TraceIncompleteError(Exception):
    def __init__(self, trace_id: str, problems: Sequence[str], waited_s: float) -> None:
        super().__init__(
            f"trace incomplete: {trace_id} after {waited_s:.0f} s ({'; '.join(problems)})"
        )
        self.problems = list(problems)


async def wait_for_trace(
    trace_dirs: Sequence[Path],
    response: ChatResponse,
    *,
    timeout_s: float = TRACE_WAIT_S,
    poll_s: float = TRACE_POLL_S,
) -> TurnTrace:
    """Poll until the turn's trace is complete; raise TraceIncompleteError at the bound."""
    started = time.monotonic()
    while True:
        trace = await asyncio.to_thread(read_trace, trace_dirs, response.trace_id)
        problems = missing_parts(trace, response)
        if not problems:
            return trace
        waited = time.monotonic() - started
        if waited >= timeout_s:
            raise TraceIncompleteError(response.trace_id, problems, waited)
        await asyncio.sleep(poll_s)
