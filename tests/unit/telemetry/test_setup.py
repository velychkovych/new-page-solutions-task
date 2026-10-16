import json
import threading
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
import respx
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from helpdesk.shared import trace_attributes as ta
from helpdesk.shared.settings import Settings
from helpdesk.telemetry import TelemetryConfigError, setup_telemetry, shutdown_telemetry
from helpdesk.telemetry import setup as telemetry_setup
from helpdesk.telemetry.delegating import PROCESS_TRACER_PROVIDER
from helpdesk.telemetry.jsonl_exporter import JsonlSpanExporter
from helpdesk.telemetry.setup import otlp_traces_endpoint


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_jsonl_only_when_no_otlp_endpoint(make_settings: Any) -> None:
    settings: Settings = make_settings()
    telemetry = setup_telemetry("agent", settings)

    (jsonl,) = telemetry.exporters
    assert isinstance(jsonl.inner, JsonlSpanExporter)
    assert jsonl.inner.path == settings.traces_dir / "agent.jsonl"
    assert telemetry.provider.resource.attributes[SERVICE_NAME] == "agent"
    # The global is the delegating provider; this setup's SDK provider sits behind it.
    assert trace.get_tracer_provider() is PROCESS_TRACER_PROVIDER
    assert PROCESS_TRACER_PROVIDER.delegate is telemetry.provider


class _CapturingOtlpExporter(InMemorySpanExporter):
    """Stands in for OTLPSpanExporter, so the test sees what would be sent to Jaeger."""

    created: ClassVar[list[_CapturingOtlpExporter]] = []

    def __init__(self, endpoint: str) -> None:
        super().__init__()
        self.endpoint = endpoint
        _CapturingOtlpExporter.created.append(self)


@pytest.fixture
def fake_otlp(monkeypatch: pytest.MonkeyPatch) -> list[_CapturingOtlpExporter]:
    _CapturingOtlpExporter.created = []
    monkeypatch.setattr(telemetry_setup, "OTLPSpanExporter", _CapturingOtlpExporter)
    return _CapturingOtlpExporter.created


def test_otlp_exporter_receives_redacted_spans_when_endpoint_set(
    make_settings: Any, fake_otlp: list[_CapturingOtlpExporter], secrets: tuple[str, ...]
) -> None:
    settings: Settings = make_settings(otel_exporter_otlp_endpoint="http://jaeger:4318")
    telemetry = setup_telemetry("mcp", settings)
    with trace.get_tracer("test").start_as_current_span("call") as span:
        span.set_attribute("http.request.header.authorization", f"Bearer {secrets[0]}")
        span.set_attribute(ta.TOOL_ARGS, f"token={secrets[0]}")
    assert telemetry.flush()

    (otlp,) = fake_otlp
    assert otlp.endpoint == "http://jaeger:4318/v1/traces"
    (exported,) = otlp.get_finished_spans()
    # Only through the redacting wrapper, never the raw exporter.
    assert exported.attributes == {ta.TOOL_ARGS: "token=[REDACTED]"}
    # The JSONL file gets the same span: OTLP is an addition, not a replacement.
    assert [r["name"] for r in _records(settings.traces_dir / "mcp.jsonl")] == ["call"]


def test_no_otlp_exporter_without_endpoint(
    make_settings: Any, fake_otlp: list[_CapturingOtlpExporter]
) -> None:
    settings: Settings = make_settings()
    telemetry = setup_telemetry("mcp", settings)
    with trace.get_tracer("test").start_as_current_span("call"):
        pass
    assert telemetry.flush()

    assert fake_otlp == []
    assert [r["name"] for r in _records(settings.traces_dir / "mcp.jsonl")] == ["call"]


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("http://jaeger:4318", "http://jaeger:4318/v1/traces"),
        ("http://jaeger:4318/", "http://jaeger:4318/v1/traces"),
        ("http://jaeger:4318/v1/traces", "http://jaeger:4318/v1/traces"),
    ],
)
def test_otlp_endpoint_gets_the_signal_path(base: str, expected: str) -> None:
    assert otlp_traces_endpoint(base) == expected


async def test_libraries_are_instrumented_once_and_follow_each_setup(
    make_settings: Any, tmp_path: Path
) -> None:
    first: Settings = make_settings(traces_dir=tmp_path / "first")
    second: Settings = make_settings(traces_dir=tmp_path / "second")
    setup_telemetry("mcp", first)
    assert HTTPXClientInstrumentor().is_instrumented_by_opentelemetry
    assert LangChainInstrumentor().is_instrumented_by_opentelemetry
    shutdown_telemetry()
    # Instrumentation stays: it is bound to the delegating provider, not to one setup.
    assert HTTPXClientInstrumentor().is_instrumented_by_opentelemetry
    assert LangChainInstrumentor().is_instrumented_by_opentelemetry

    telemetry = setup_telemetry("mcp", second)
    with respx.mock(assert_all_called=True) as mock:
        mock.get("http://mock.test/tickets").respond(200, json={"items": []})
        async with httpx.AsyncClient() as client:
            assert (await client.get("http://mock.test/tickets")).status_code == 200
    assert telemetry.flush()

    assert not (first.traces_dir / "mcp.jsonl").exists()
    (outbound,) = _records(second.traces_dir / "mcp.jsonl")
    assert outbound["name"] == "GET"
    assert outbound["service"] == "mcp"


def test_shutdown_flushes_pending_spans(make_settings: Any) -> None:
    settings: Settings = make_settings()
    setup_telemetry("agent", settings)
    with trace.get_tracer("test").start_as_current_span("last-request"):
        pass

    shutdown_telemetry()

    assert [r["name"] for r in _records(settings.traces_dir / "agent.jsonl")] == ["last-request"]


def test_setup_refuses_when_another_global_provider_exists(make_settings: Any) -> None:
    foreign = TracerProvider()
    trace.set_tracer_provider(foreign)

    # The OTel API cannot replace the global provider, so setup must not pretend.
    with pytest.raises(TelemetryConfigError, match=r"another global tracer provider"):
        setup_telemetry("agent", make_settings())
    # Nothing is half set up: no delegate, no instrumentation, and a retry still refuses.
    assert PROCESS_TRACER_PROVIDER.delegate is None
    assert not HTTPXClientInstrumentor().is_instrumented_by_opentelemetry
    with pytest.raises(TelemetryConfigError, match=r"another global tracer provider"):
        setup_telemetry("agent", make_settings())
    foreign.shutdown()


def test_a_failed_setup_shuts_down_the_provider_it_built(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[telemetry_setup.Telemetry] = []
    build = telemetry_setup.build_telemetry

    def build_and_keep(service_name: str, settings: Settings) -> telemetry_setup.Telemetry:
        built.append(build(service_name, settings))
        return built[-1]

    def fail() -> None:
        raise RuntimeError("instrumentation failed")

    monkeypatch.setattr(telemetry_setup, "build_telemetry", build_and_keep)
    monkeypatch.setattr(telemetry_setup, "_instrument_libraries", fail)
    before = set(threading.enumerate())

    with pytest.raises(RuntimeError, match="instrumentation failed"):
        setup_telemetry("agent", make_settings())

    # Otherwise each failed start leaks the batch processor's worker thread and the
    # JSONL exporter's open file.
    [telemetry] = built
    leaked = [t for t in threading.enumerate() if t not in before and t.is_alive()]
    assert leaked == []
    span = telemetry.provider.get_tracer("t").start_span("after-failure")
    span.end()
    assert not (make_settings().traces_dir / "agent.jsonl").exists()
    assert PROCESS_TRACER_PROVIDER.delegate is None


@pytest.mark.parametrize("name", ["", "Agent", "../etc", "a/b", "x" * 65])
def test_rejects_service_names_that_are_not_safe_file_names(make_settings: Any, name: str) -> None:
    with pytest.raises(TelemetryConfigError, match="invalid service name"):
        setup_telemetry(name, make_settings())
