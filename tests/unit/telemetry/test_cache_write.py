"""cache_write on OpenInference LLM spans, through the real LangChain instrumentation."""

import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from anthropic.types import Message, RawMessageDeltaEvent
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AnyValue

from helpdesk.shared import trace_attributes as ta
from helpdesk.shared.settings import Settings
from helpdesk.telemetry import cache_write as cache_write_module
from helpdesk.telemetry import setup_telemetry
from helpdesk.telemetry.cache_write import CacheWriteSpanProcessor, derive_cache_write

CACHE_WRITE = "llm.token_count.prompt_details.cache_write"
CACHE_READ = "llm.token_count.prompt_details.cache_read"
LLM_KIND = {"openinference.span.kind": "LLM"}


def _llm_records(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [r for r in records if r["attributes"].get("openinference.span.kind") == "LLM"]


def _anthropic() -> ChatAnthropic:
    # Built only for its response-formatting helpers; nothing is ever sent.
    return ChatAnthropic(model_name="test-model", api_key="test-key")  # type: ignore[call-arg,arg-type]


def _delta_usage(
    cache_creation: int, ttl_5m: int | None = None, ttl_1h: int | None = None
) -> dict[str, Any]:
    """Anthropic usage; the TTL breakdown is an extra field the SDK's models keep."""
    usage: dict[str, Any] = {
        "input_tokens": 50,
        "output_tokens": 80,
        "cache_read_input_tokens": 1000,
        "cache_creation_input_tokens": cache_creation,
    }
    if ttl_5m is not None or ttl_1h is not None:
        usage["cache_creation"] = {
            "ephemeral_5m_input_tokens": ttl_5m or 0,
            "ephemeral_1h_input_tokens": ttl_1h or 0,
        }
    return usage


class _StreamingModel(BaseChatModel):
    """Streams as ChatAnthropic does: usage only on the final chunk, and no llm_output.

    The final chunk comes from langchain-anthropic's own event conversion, so its
    usage_metadata has the real shape (cache_creation=0 next to the TTL breakdown).
    """

    delta_usage: dict[str, Any]

    @property
    def _llm_type(self) -> str:
        return "streaming-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(message=AIMessageChunk(content="ok"))
        event = RawMessageDeltaEvent.model_validate(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": self.delta_usage,
            }
        )
        last, _ = _anthropic()._make_message_chunk_from_anthropic_event(
            event, stream_usage=True, coerce_content_to_string=True
        )
        assert last is not None
        yield ChatGenerationChunk(message=last)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in self._stream(messages, stop):
            yield chunk


class _NonStreamingModel(BaseChatModel):
    """Returns a ChatResult exactly as ChatAnthropic formats a non-streamed response."""

    result: ChatResult

    @property
    def _llm_type(self) -> str:
        return "non-streaming-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self.result


def _anthropic_result(usage: dict[str, Any]) -> ChatResult:
    message = Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "test-model",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": usage,
        }
    )
    # langchain-anthropic's own formatting: the real llm_output and usage_metadata.
    return _anthropic()._format_output(message)


async def _llm_span(settings: Settings, model: BaseChatModel, *, stream: bool) -> dict[str, Any]:
    telemetry = setup_telemetry("agent", settings)
    with trace.get_tracer("test").start_as_current_span(ta.SPAN_REQUEST):
        if stream:
            async for _ in model.astream("hi"):
                pass
        else:
            await model.ainvoke("hi")
    assert telemetry.flush()
    (llm,) = _llm_records(settings.traces_dir / "agent.jsonl")
    attributes: dict[str, Any] = llm["attributes"]
    return attributes


@pytest.mark.parametrize(
    ("ttl_5m", "ttl_1h", "expected"),
    [(150, 0, 150), (0, 70, 70), (150, 70, 220)],
)
async def test_streamed_call_gets_cache_write_from_the_ttl_breakdown(
    make_settings: Any, ttl_5m: int, ttl_1h: int, expected: int
) -> None:
    model = _StreamingModel(delta_usage=_delta_usage(expected, ttl_5m, ttl_1h))

    attributes = await _llm_span(make_settings(), model, stream=True)

    assert attributes[CACHE_WRITE] == expected
    assert attributes[CACHE_READ] == 1000


async def test_streamed_call_without_a_ttl_breakdown_keeps_openinferences_count(
    make_settings: Any,
) -> None:
    # Without a breakdown langchain-anthropic leaves the total in cache_creation, which
    # OpenInference records itself.
    model = _StreamingModel(delta_usage=_delta_usage(150))

    attributes = await _llm_span(make_settings(), model, stream=True)

    assert attributes[CACHE_WRITE] == 150
    assert attributes[CACHE_READ] == 1000


async def test_streamed_call_without_cache_writes_gets_no_cache_write(make_settings: Any) -> None:
    model = _StreamingModel(delta_usage=_delta_usage(0, 0, 0))

    attributes = await _llm_span(make_settings(), model, stream=True)

    assert CACHE_WRITE not in attributes
    assert attributes[CACHE_READ] == 1000


async def test_non_streamed_call_is_not_counted_twice(make_settings: Any) -> None:
    # OpenInference reads the raw Anthropic usage (cache_creation_input_tokens=150) here,
    # while usage_metadata repeats the same 150 as ephemeral_5m_input_tokens.
    result = _anthropic_result(_delta_usage(150, 150, 0))
    message = result.generations[0].message
    assert isinstance(message, AIMessage)
    assert message.usage_metadata is not None
    details = message.usage_metadata.get("input_token_details", {})
    assert details.get("ephemeral_5m_input_tokens") == 150

    attributes = await _llm_span(make_settings(), _NonStreamingModel(result=result), stream=False)

    assert attributes[CACHE_WRITE] == 150
    assert attributes[CACHE_READ] == 1000


def _output_value(details: Any) -> str:
    usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    usage["input_token_details"] = details
    message = {"kwargs": {"usage_metadata": usage}}
    return json.dumps({"generations": [[{"message": message}]], "llm_output": None})


GOOD_OUTPUT = _output_value({"cache_creation": 0, "ephemeral_5m_input_tokens": 150})


def test_derives_from_a_well_formed_llm_span() -> None:
    assert derive_cache_write({**LLM_KIND, "output.value": GOOD_OUTPUT}) == 150


@pytest.mark.parametrize(
    "attributes",
    [
        pytest.param({**LLM_KIND}, id="no-output-value"),
        pytest.param({**LLM_KIND, "output.value": GOOD_OUTPUT[:-7]}, id="truncated"),
        pytest.param({**LLM_KIND, "output.value": "__REDACTED__"}, id="masked-by-trace-config"),
        pytest.param({**LLM_KIND, "output.value": "[1, 2]"}, id="not-an-object"),
        pytest.param({**LLM_KIND, "output.value": 42}, id="not-a-string"),
        pytest.param({**LLM_KIND, "output.value": '{"generations": []}'}, id="no-generation"),
        pytest.param({**LLM_KIND, "output.value": '{"generations": [["x"]]}'}, id="odd-generation"),
        pytest.param({**LLM_KIND, "output.value": _output_value(None)}, id="no-details"),
        pytest.param({**LLM_KIND, "output.value": _output_value("x")}, id="details-not-object"),
        pytest.param(
            {**LLM_KIND, "output.value": _output_value({"ephemeral_5m_input_tokens": "150"})},
            id="count-is-a-string",
        ),
        pytest.param(
            {**LLM_KIND, "output.value": _output_value({"ephemeral_5m_input_tokens": True})},
            id="count-is-a-bool",
        ),
        pytest.param(
            {**LLM_KIND, "output.value": _output_value({"ephemeral_5m_input_tokens": -5})},
            id="count-is-negative",
        ),
        pytest.param({**LLM_KIND, "output.value": "[" * 100_000}, id="deeply-nested"),
    ],
)
def test_malformed_or_missing_usage_adds_nothing(attributes: dict[str, AnyValue]) -> None:
    assert derive_cache_write(attributes) is None


def test_a_recorded_cache_write_is_kept() -> None:
    assert derive_cache_write({**LLM_KIND, CACHE_WRITE: 40, "output.value": GOOD_OUTPUT}) is None


def test_non_streaming_usage_in_llm_output_is_left_to_openinference() -> None:
    outputs = json.loads(GOOD_OUTPUT)
    outputs["llm_output"] = {"usage": {"input_tokens": 1, "output_tokens": 1}}
    assert derive_cache_write({**LLM_KIND, "output.value": json.dumps(outputs)}) is None


class _Recorder(SpanProcessor):
    def __init__(self) -> None:
        self.started: list[Span] = []
        self.ended: list[ReadableSpan] = []
        self.flushed_with: list[int] = []
        self.shut_down = False

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        self.started.append(span)

    def on_end(self, span: ReadableSpan) -> None:
        self.ended.append(span)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        self.flushed_with.append(timeout_millis)
        return True

    def shutdown(self) -> None:
        self.shut_down = True


def _provider_with(*downstream: SpanProcessor) -> TracerProvider:
    provider = TracerProvider()
    provider.add_span_processor(CacheWriteSpanProcessor(downstream))
    return provider


def test_processor_adds_cache_write_to_llm_spans_only() -> None:
    first, second = _Recorder(), _Recorder()
    tracer = _provider_with(first, second).get_tracer("test")

    with tracer.start_as_current_span("llm") as span:
        span.set_attributes({**LLM_KIND, "output.value": GOOD_OUTPUT})
    # Same output, but a chain span: OpenInference records no token counts there.
    with tracer.start_as_current_span("chain") as span:
        span.set_attributes({"openinference.span.kind": "CHAIN", "output.value": GOOD_OUTPUT})
    with tracer.start_as_current_span("plain"):
        pass

    for recorder in (first, second):
        assert [s.name for s in recorder.started] == ["llm", "chain", "plain"]
        llm, chain, plain = recorder.ended
        assert (llm.attributes or {})[CACHE_WRITE] == 150
        assert (llm.attributes or {})["output.value"] == GOOD_OUTPUT
        assert CACHE_WRITE not in (chain.attributes or {})
        assert CACHE_WRITE not in (plain.attributes or {})
    # Both exporters get the same copy; untouched spans are passed on as they are.
    assert first.ended[0] is second.ended[0]
    assert first.ended[1] is second.ended[1]


def test_processor_copy_keeps_everything_else() -> None:
    recorder = _Recorder()
    tracer = _provider_with(recorder).get_tracer("test")
    with tracer.start_as_current_span("llm", attributes={**LLM_KIND}) as span:
        span.set_attribute("output.value", GOOD_OUTPUT)
        span.add_event("token", {"n": 1})
        original_context = span.get_span_context()

    (copy,) = recorder.ended
    assert copy.context == original_context
    assert [e.name for e in copy.events] == ["token"]
    assert copy.instrumentation_scope is not None
    assert copy.instrumentation_scope.name == "test"
    assert copy.start_time is not None
    assert copy.end_time is not None


def test_processor_never_drops_a_span_when_derivation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(span: ReadableSpan) -> ReadableSpan:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(cache_write_module, "with_cache_write", broken)
    recorder = _Recorder()
    tracer = _provider_with(recorder).get_tracer("test")

    with tracer.start_as_current_span("llm", attributes={**LLM_KIND}):
        pass

    (span,) = recorder.ended
    assert span.name == "llm"


def test_processor_forwards_flush_and_shutdown() -> None:
    first, second = _Recorder(), _Recorder()
    provider = _provider_with(first, second)

    assert provider.force_flush(5_000)
    provider.shutdown()

    assert len(first.flushed_with) == len(second.flushed_with) == 1
    assert 0 < second.flushed_with[0] <= first.flushed_with[0] <= 5_000
    assert first.shut_down
    assert second.shut_down


def test_setup_installs_the_processor_in_front_of_the_exporters(make_settings: Any) -> None:
    settings: Settings = make_settings()
    telemetry = setup_telemetry("agent", settings)
    exporter = InMemorySpanExporter()
    # A processor added later sees spans without the derived count, which shows the
    # derivation happens on the exporters' path, not on the span itself.
    telemetry.provider.add_span_processor(SimpleSpanProcessor(exporter))

    with trace.get_tracer("test").start_as_current_span("llm", attributes={**LLM_KIND}) as span:
        span.set_attribute("output.value", GOOD_OUTPUT)
    assert telemetry.flush()

    (llm,) = _llm_records(settings.traces_dir / "agent.jsonl")
    assert llm["attributes"][CACHE_WRITE] == 150
    (raw,) = exporter.get_finished_spans()
    assert CACHE_WRITE not in (raw.attributes or {})
