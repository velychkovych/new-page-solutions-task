"""Cache-write token counts on LLM spans from LangChain's `usage_metadata` (streaming) path.

langchain-anthropic 1.7.6 reports prompt-cache writes in
`usage_metadata.input_token_details.ephemeral_5m_input_tokens` / `ephemeral_1h_input_tokens`
and then sets `cache_creation` to 0. OpenInference 0.1.79 reads only `cache_creation` there,
so a streamed Claude call, which is how the agent calls the model, gets no
`llm.token_count.prompt_details.cache_write`. Without streaming, OpenInference reads the
raw Anthropic usage from `llm_output` instead, and that count is already right.

Where the attribute can be set: opentelemetry-sdk 1.45.1 `Span.end()` stores the end time
and freezes the attributes *before* it calls `SpanProcessor._on_ending`, and
`set_attribute` on an ended span only logs a warning. So no processor hook can still add an
attribute to the span itself. This processor therefore sits in front of the batch
processors and passes each of them a copy of the finished span with the derived count added.
"""

import json
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.util.types import AnyValue

from helpdesk.telemetry.openinference_keys import (
    LLM_TOKENS_CACHE_WRITE,
    OUTPUT_VALUE,
    SPAN_KIND,
    SPAN_KIND_LLM,
)

logger = logging.getLogger(__name__)

# cache_creation is the total when langchain-anthropic has no TTL breakdown, and 0 when it
# has one, so the sum never counts a write twice.
_CACHE_WRITE_DETAILS = ("cache_creation", "ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")
# The keys OpenInference prefers in `llm_output`; when one is set it never reads usage_metadata.
_NON_STREAMING_USAGE_KEYS = ("token_usage", "usage")
_NS_PER_MS = 1_000_000


def _first_generation_usage(outputs: Mapping[str, Any]) -> Any:
    """`generations[0][0].message.kwargs.usage_metadata`, the path OpenInference reads."""
    try:
        return outputs["generations"][0][0]["message"]["kwargs"]["usage_metadata"]
    except LookupError, TypeError:
        return None


def _used_non_streaming_usage(outputs: Mapping[str, Any]) -> bool:
    llm_output = outputs.get("llm_output")
    if not isinstance(llm_output, Mapping):
        return False
    usage = next(
        (v for key in _NON_STREAMING_USAGE_KEYS if (v := llm_output.get(key)) is not None), None
    )
    return bool(usage)


def _streamed_input_token_details(attributes: Mapping[str, AnyValue]) -> Mapping[str, Any] | None:
    """`input_token_details` when OpenInference took the counts from usage_metadata."""
    raw = attributes.get(OUTPUT_VALUE)
    if not isinstance(raw, str):
        return None
    try:
        outputs = json.loads(raw)
    except ValueError, RecursionError:
        return None
    if not isinstance(outputs, Mapping) or _used_non_streaming_usage(outputs):
        return None
    usage = _first_generation_usage(outputs)
    details = usage.get("input_token_details") if isinstance(usage, Mapping) else None
    return details if isinstance(details, Mapping) else None


def derive_cache_write(attributes: Mapping[str, AnyValue]) -> int | None:
    """The cache-write count OpenInference missed on this span, or None if there is none.

    None for: a span that is not an OpenInference LLM span; one that already has a
    cache_write (never counted twice); one where OpenInference used the non-streaming usage;
    and any output.value that is missing, truncated, masked or not in the expected shape.
    """
    if attributes.get(SPAN_KIND) != SPAN_KIND_LLM or LLM_TOKENS_CACHE_WRITE in attributes:
        return None
    details = _streamed_input_token_details(attributes)
    if details is None:
        return None
    total = 0
    for key in _CACHE_WRITE_DETAILS:
        value = details.get(key, 0)
        # `type() is int` also rejects bools; a negative or odd value means a shape we
        # do not understand, so nothing is added rather than a wrong number.
        if type(value) is not int or value < 0:
            return None
        total += value
    return total or None


def with_cache_write(span: ReadableSpan) -> ReadableSpan:
    """`span`, or a copy of it with the derived cache_write added."""
    attributes = span.attributes or {}
    count = derive_cache_write(attributes)
    if count is None:
        return span
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes={**attributes, LLM_TOKENS_CACHE_WRITE: count},
        events=span.events,
        links=span.links,
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class CacheWriteSpanProcessor(SpanProcessor):
    """Forwards every span to `downstream`, LLM spans with the derived cache_write added.

    One processor in front of all exporters' batch processors, so JSONL and OTLP see the
    same span and the output is parsed once.
    """

    def __init__(self, downstream: Sequence[SpanProcessor]) -> None:
        self._downstream = tuple(downstream)

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        for processor in self._downstream:
            processor.on_start(span, parent_context=parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        try:
            forwarded = with_cache_write(span)
        except Exception:
            # Telemetry must never drop a span or break the request that ended it.
            logger.warning("could not derive cache_write tokens; span exported unchanged")
            forwarded = span
        for processor in self._downstream:
            processor.on_end(forwarded)

    def shutdown(self) -> None:
        for processor in self._downstream:
            processor.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        # One deadline for all of them, as the SDK's SynchronousMultiSpanProcessor does.
        deadline_ns = time.monotonic_ns() + timeout_millis * _NS_PER_MS
        all_flushed = True
        for processor in self._downstream:
            remaining_ms = (deadline_ns - time.monotonic_ns()) // _NS_PER_MS
            if remaining_ms <= 0:
                return False
            if not processor.force_flush(remaining_ms):
                all_flushed = False
        return all_flushed
