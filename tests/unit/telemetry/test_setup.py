import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi import FastAPI
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME

from helpdesk.shared.settings import Settings
from helpdesk.telemetry import (
    TelemetryConfigError,
    instrument_app,
    setup_telemetry,
    shutdown_telemetry,
)
from helpdesk.telemetry.jsonl_exporter import JsonlSpanExporter
from helpdesk.telemetry.setup import build_telemetry, otlp_traces_endpoint


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
    assert trace.get_tracer_provider() is telemetry.provider


def test_otlp_exporter_added_when_endpoint_set(make_settings: Any) -> None:
    # Nothing is sent: the test only inspects the configured exporters.
    settings: Settings = make_settings(otel_exporter_otlp_endpoint="http://jaeger:4318")
    telemetry = build_telemetry("mcp", settings)

    jsonl, otlp = telemetry.exporters
    assert isinstance(jsonl.inner, JsonlSpanExporter)
    assert isinstance(otlp.inner, OTLPSpanExporter)
    assert otlp.inner._endpoint == "http://jaeger:4318/v1/traces"
    telemetry.provider.shutdown()


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


def test_setup_is_idempotent_for_the_same_service(make_settings: Any) -> None:
    settings: Settings = make_settings()
    first = setup_telemetry("agent", settings)
    assert setup_telemetry("agent", settings) is first
    with pytest.raises(TelemetryConfigError, match="already set up"):
        setup_telemetry("mcp", settings)


def test_setup_instruments_httpx_and_langchain_and_shutdown_undoes_it(
    make_settings: Any,
) -> None:
    setup_telemetry("agent", make_settings())
    assert HTTPXClientInstrumentor().is_instrumented_by_opentelemetry
    assert LangChainInstrumentor().is_instrumented_by_opentelemetry

    shutdown_telemetry()
    assert not HTTPXClientInstrumentor().is_instrumented_by_opentelemetry
    assert not LangChainInstrumentor().is_instrumented_by_opentelemetry
    shutdown_telemetry()  # a second shutdown is a no-op


def test_shutdown_flushes_pending_spans(make_settings: Any) -> None:
    settings: Settings = make_settings()
    setup_telemetry("agent", settings)
    with trace.get_tracer("test").start_as_current_span("last-request"):
        pass

    shutdown_telemetry()

    assert [r["name"] for r in _records(settings.traces_dir / "agent.jsonl")] == ["last-request"]


def test_setup_refuses_when_another_global_provider_exists(make_settings: Any) -> None:
    settings: Settings = make_settings()
    setup_telemetry("agent", settings)
    shutdown_telemetry()
    # The OTel API cannot replace the global provider, so a second setup must not pretend.
    with pytest.raises(TelemetryConfigError, match="already installed"):
        setup_telemetry("agent", settings)


@pytest.mark.parametrize("name", ["", "Agent", "../etc", "a/b", "x" * 65])
def test_rejects_service_names_that_are_not_safe_file_names(make_settings: Any, name: str) -> None:
    with pytest.raises(TelemetryConfigError, match="invalid service name"):
        setup_telemetry(name, make_settings())


def test_instrument_app_requires_setup() -> None:
    with pytest.raises(TelemetryConfigError, match="setup_telemetry"):
        instrument_app(FastAPI())


async def test_instrumented_app_and_httpx_client_produce_linked_spans(
    make_settings: Any,
) -> None:
    settings: Settings = make_settings()
    telemetry = setup_telemetry("agent", settings)
    app = FastAPI()

    @app.get("/ping")
    async def ping() -> dict[str, int]:
        # An outbound call from inside a request, like the MCP server calling the mock.
        async with httpx.AsyncClient() as upstream:
            response = await upstream.get("http://mock.test/tickets")
        return {"upstream": response.status_code}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    instrument_app(app)
    instrument_app(app)  # idempotent

    with respx.mock(assert_all_called=True) as mock:
        mock.get("http://mock.test/tickets").respond(200, json={"items": []})
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://agent.test"
        ) as client:
            assert (await client.get("/ping")).json() == {"upstream": 200}
            assert (await client.get("/health")).status_code == 200
    assert telemetry.flush()

    records = _records(settings.traces_dir / "agent.jsonl")
    (server,) = [r for r in records if r["name"] == "GET /ping"]
    (outbound,) = [r for r in records if r["name"] == "GET" and r["parent_span_id"]]
    assert outbound["trace_id"] == server["trace_id"]
    assert outbound["parent_span_id"] == server["span_id"]
    assert not [r for r in records if "/health" in r["name"]]
    assert all(r["service"] == "agent" for r in records)
