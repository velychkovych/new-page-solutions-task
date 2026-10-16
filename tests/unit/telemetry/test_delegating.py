"""The delegating tracer keeps what instrumentations expect of the SDK tracer behind it."""

import json
import random
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from openinference.instrumentation import OITracer, TraceConfig
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.id_generator import IdGenerator

from helpdesk.shared.settings import Settings
from helpdesk.telemetry import setup_telemetry, shutdown_telemetry
from helpdesk.telemetry.delegating import DelegatingTracerProvider


class _FixedIds(IdGenerator):
    def generate_span_id(self) -> int:
        return 0xABC

    def generate_trace_id(self) -> int:
        return 0xDEF


def _span_ids_after_seeding(tracer: object) -> list[int]:
    ids = []
    for _ in range(2):
        # Application code seeding the global `random` must not make span ids repeat.
        random.seed(42)
        span = tracer.start_span("x")  # type: ignore[attr-defined]
        ids.append(span.get_span_context().span_id)
        span.end()
    return ids


def test_openinference_ids_stay_independent_of_the_global_random_seed() -> None:
    delegating = DelegatingTracerProvider()
    sdk = TracerProvider()
    delegating.set_delegate(sdk)
    # OpenInference wraps the tracer it is given and swaps the SDK's RandomIdGenerator for
    # one built on `secrets`; that swap has to survive the delegating layer.
    tracer = OITracer(delegating.get_tracer("t"), TraceConfig())

    first, second = _span_ids_after_seeding(tracer)

    assert first != second
    sdk.shutdown()


def test_a_custom_id_generator_on_the_delegate_is_used_with_and_without_openinference() -> None:
    delegating = DelegatingTracerProvider()
    sdk = TracerProvider(id_generator=_FixedIds())
    delegating.set_delegate(sdk)
    plain = delegating.get_tracer("t")
    wrapped = OITracer(delegating.get_tracer("t"), TraceConfig())

    for tracer in (plain, wrapped):
        span = tracer.start_span("x")
        context = span.get_span_context()
        span.end()
        assert (context.trace_id, context.span_id) == (0xDEF, 0xABC)
    sdk.shutdown()


def test_without_a_delegate_spans_are_non_recording_through_openinference() -> None:
    tracer = OITracer(DelegatingTracerProvider().get_tracer("t"), TraceConfig())

    span = tracer.start_span("x")

    assert not span.is_recording()
    span.end()


async def test_langchain_span_ids_stay_independent_of_the_global_random_seed(
    make_settings: Any,
) -> None:
    settings: Settings = make_settings()
    setup_telemetry("agent", settings)
    model = GenericFakeChatModel(messages=iter([AIMessage(content="a"), AIMessage(content="b")]))

    for _ in range(2):
        random.seed(42)
        await model.ainvoke("hi")
    shutdown_telemetry()

    lines = (settings.traces_dir / "agent.jsonl").read_text(encoding="utf-8").splitlines()
    span_ids = [json.loads(line)["span_id"] for line in lines]
    assert len(span_ids) == 2
    assert span_ids[0] != span_ids[1]
