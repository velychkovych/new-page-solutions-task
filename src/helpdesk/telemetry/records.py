"""The JSONL span record (CONTRACTS §6): written by the exporter, read by show_trace and evals."""

import base64
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Literal

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode, format_span_id, format_trace_id
from opentelemetry.util.types import AnyValue
from pydantic import BaseModel, ConfigDict, Field, JsonValue

type SpanStatus = Literal["ok", "error"]

TRACE_ID_PATTERN = r"^[0-9a-f]{32}$"
SPAN_ID_PATTERN = r"^[0-9a-f]{16}$"

_NS_PER_S = 1_000_000_000
_NS_PER_MS = 1_000_000
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class SpanEventRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    timestamp: datetime
    attributes: dict[str, JsonValue]


class SpanRecord(BaseModel):
    """One line of `traces/<service>.jsonl`. The field set is fixed by CONTRACTS §6."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: str = Field(pattern=TRACE_ID_PATTERN)
    span_id: str = Field(pattern=SPAN_ID_PATTERN)
    parent_span_id: str | None = Field(pattern=SPAN_ID_PATTERN)
    name: str
    service: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    status: SpanStatus
    attributes: dict[str, JsonValue]
    events: list[SpanEventRecord]


def ns_to_datetime(ns: int) -> datetime:
    # Integer arithmetic: `ns / 1e9` loses microseconds on current epoch values.
    seconds, rest = divmod(ns, _NS_PER_S)
    return _EPOCH + timedelta(seconds=seconds, microseconds=rest // 1_000)


def attribute_to_json(value: AnyValue) -> JsonValue:
    """Map an OTel attribute value (recursive since opentelemetry 1.4x) to plain JSON."""
    match value:
        case None | str() | bool() | int() | float():
            return value
        case bytes():
            # Lossless, and matches how OTLP/JSON encodes bytes values.
            return base64.b64encode(value).decode("ascii")
        case Mapping():
            return {str(k): attribute_to_json(v) for k, v in value.items()}
        case Sequence():
            return [attribute_to_json(v) for v in value]


def attributes_to_json(attributes: Mapping[str, AnyValue] | None) -> dict[str, JsonValue]:
    return {key: attribute_to_json(value) for key, value in (attributes or {}).items()}


def span_to_record(span: ReadableSpan, service: str) -> SpanRecord | None:
    """Convert a finished span; None for a span the SDK could not have finished properly."""
    context = span.context
    if context is None or span.start_time is None or span.end_time is None:
        return None
    parent = span.parent
    return SpanRecord(
        trace_id=format_trace_id(context.trace_id),
        span_id=format_span_id(context.span_id),
        parent_span_id=format_span_id(parent.span_id) if parent is not None else None,
        name=span.name,
        service=service,
        start_time=ns_to_datetime(span.start_time),
        end_time=ns_to_datetime(span.end_time),
        duration_ms=round((span.end_time - span.start_time) / _NS_PER_MS, 3),
        # UNSET means "nobody reported a failure", which the contract calls ok.
        status="error" if span.status.status_code is StatusCode.ERROR else "ok",
        attributes=attributes_to_json(span.attributes),
        events=[
            SpanEventRecord(
                name=event.name,
                timestamp=ns_to_datetime(event.timestamp),
                attributes=attributes_to_json(event.attributes),
            )
            for event in span.events
        ],
    )
