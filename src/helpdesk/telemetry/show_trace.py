"""Print one trace's span tree from every service's JSONL file (`make trace ID=<trace_id>`)."""

import argparse
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TextIO

from pydantic import JsonValue, ValidationError

from helpdesk.shared import trace_attributes as ta
from helpdesk.shared.settings import Settings, get_settings
from helpdesk.telemetry.openinference_keys import (
    LLM_MODEL_NAME,
    LLM_TOKENS_CACHE_READ,
    LLM_TOKENS_CACHE_WRITE,
    LLM_TOKENS_COMPLETION,
    LLM_TOKENS_PROMPT,
)
from helpdesk.telemetry.records import SpanEventRecord, SpanRecord

_EXCEPTION_EVENT = "exception"
_MAX_DETAIL_CHARS = 160

_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_INDENT = "  "


class ExitCode(IntEnum):
    OK = 0
    NOT_FOUND = 1
    INVALID_TRACE_ID = 2


@dataclass(frozen=True, slots=True)
class LoadedTrace:
    spans: list[SpanRecord]
    files_read: int
    skipped_lines: int


def normalise_trace_id(raw: str) -> str | None:
    """Lower-case 32-hex id, or None. The all-zero id is invalid in W3C trace context."""
    candidate = raw.strip().lower().removeprefix("0x")
    if not _TRACE_ID_RE.fullmatch(candidate) or candidate == "0" * 32:
        return None
    return candidate


def load_trace(traces_dir: Path, trace_id: str) -> LoadedTrace:
    """Every span of `trace_id` from all `*.jsonl` files, de-duplicated by span id."""
    spans: dict[str, SpanRecord] = {}
    files = sorted(traces_dir.glob("*.jsonl")) if traces_dir.is_dir() else []
    skipped = 0
    needle = trace_id.encode("ascii")
    for path in files:
        # Binary, so a write cut inside a multi-byte character spoils only its own line
        # instead of raising for the whole file.
        with path.open("rb") as file:
            for line in file:
                # Cheap pre-filter: trace files grow large and most lines are other traces.
                if needle not in line:
                    continue
                try:
                    record = SpanRecord.model_validate_json(line.decode("utf-8"))
                except UnicodeDecodeError, ValidationError:
                    # A line a service is still writing, or a corrupt one: skip, but say so.
                    skipped += 1
                    continue
                if record.trace_id == trace_id:
                    spans[record.span_id] = record
    return LoadedTrace(spans=list(spans.values()), files_read=len(files), skipped_lines=skipped)


def _short(value: JsonValue) -> str:
    text = value if isinstance(value, str) else str(value)
    text = " ".join(text.split())
    return text if len(text) <= _MAX_DETAIL_CHARS else text[: _MAX_DETAIL_CHARS - 1] + "…"


def _int_attr(attributes: dict[str, JsonValue], key: str) -> int:
    value = attributes.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _span_details(span: SpanRecord) -> list[str]:
    attrs = span.attributes
    details: list[str] = []
    if LLM_TOKENS_PROMPT in attrs or LLM_TOKENS_COMPLETION in attrs:
        if LLM_MODEL_NAME in attrs:
            details.append(f"model={_short(attrs[LLM_MODEL_NAME])}")
        details.append(
            f"tokens in={_int_attr(attrs, LLM_TOKENS_PROMPT)}"
            f" out={_int_attr(attrs, LLM_TOKENS_COMPLETION)}"
            f" cache_read={_int_attr(attrs, LLM_TOKENS_CACHE_READ)}"
            f" cache_write={_int_attr(attrs, LLM_TOKENS_CACHE_WRITE)}"
        )
    labelled = (
        ("tool", ta.TOOL_NAME),
        ("attempts", ta.TOOL_ATTEMPTS),
        ("error_code", ta.TOOL_ERROR_CODE),
        ("request_status", ta.REQUEST_STATUS),
        ("pending_action", ta.PENDING_ACTION_ID),
        ("decision", ta.APPROVAL_DECISION),
    )
    details.extend(f"{label}={_short(attrs[key])}" for label, key in labelled if key in attrs)
    return details


def _event_line(event: SpanEventRecord) -> str | None:
    attrs = event.attributes
    if event.name == ta.EVENT_RETRY:
        keys = (ta.RETRY_ATTEMPT, ta.RETRY_HTTP_STATUS, ta.RETRY_WAIT_MS)
        parts = [f"{key}={_short(attrs[key])}" for key in keys if key in attrs]
        return " ".join([event.name, *parts])
    if event.name == _EXCEPTION_EVENT:
        kind = attrs.get("exception.type", "")
        message = attrs.get("exception.message", "")
        return f"exception {_short(kind)}: {_short(message)}"
    return None


def _span_line(span: SpanRecord, orphan: bool) -> str:
    parts = [f"{span.name} [{span.service}] {span.duration_ms:.1f} ms {span.status}"]
    parts.extend(_span_details(span))
    if orphan:
        parts.append(f"(orphan: parent {span.parent_span_id} is in no trace file)")
    return "  ".join(parts)


def render_tree(trace_id: str, spans: Sequence[SpanRecord]) -> str:
    by_id = {span.span_id: span for span in spans}
    children: defaultdict[str | None, list[SpanRecord]] = defaultdict(list)
    for span in spans:
        # A parent that no file contains (lost batch, service without telemetry) makes the
        # span a root, so it is still shown instead of silently disappearing.
        parent = span.parent_span_id if span.parent_span_id in by_id else None
        children[parent].append(span)
    for siblings in children.values():
        siblings.sort(key=lambda s: (s.start_time, s.name))

    services = sorted({span.service for span in spans})
    lines = [f"trace {trace_id}: {len(spans)} spans, services: {', '.join(services)}"]

    def walk(nodes: Iterable[SpanRecord], depth: int) -> None:
        for span in nodes:
            orphan = span.parent_span_id is not None and span.parent_span_id not in by_id
            lines.append(_INDENT * depth + _span_line(span, orphan))
            for event in span.events:
                if (text := _event_line(event)) is not None:
                    lines.append(_INDENT * (depth + 1) + "· " + text)
            walk(children.get(span.span_id, []), depth + 1)

    walk(children.get(None, []), 0)
    return "\n".join(lines) + "\n"


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> int:
    stdout = out or sys.stdout
    stderr = err or sys.stderr
    parser = argparse.ArgumentParser(
        prog="show_trace", description="Print a trace's span tree from the JSONL trace files."
    )
    parser.add_argument("trace_id", help="32 hex characters, as returned in every agent response")
    args = parser.parse_args(argv)

    trace_id = normalise_trace_id(args.trace_id)
    if trace_id is None:
        stderr.write(f"error: invalid trace id {args.trace_id!r}: expected 32 hex characters\n")
        return ExitCode.INVALID_TRACE_ID

    traces_dir = (settings or get_settings()).traces_dir
    loaded = load_trace(traces_dir, trace_id)
    if loaded.skipped_lines:
        stderr.write(f"warning: skipped {loaded.skipped_lines} unreadable line(s)\n")
    if not loaded.spans:
        where = f"{loaded.files_read} *.jsonl file(s) in {traces_dir}"
        if not traces_dir.is_dir():
            where = f"{traces_dir} (directory does not exist)"
        stderr.write(f"error: trace {trace_id} not found in {where}\n")
        return ExitCode.NOT_FOUND

    stdout.write(render_tree(trace_id, loaded.spans))
    return ExitCode.OK
