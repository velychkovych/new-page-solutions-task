import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from helpdesk.shared import trace_attributes as ta
from helpdesk.shared.settings import Settings
from helpdesk.telemetry import (
    TelemetryConfigError,
    instrument_app,
    setup_telemetry,
    shutdown_telemetry,
)
from helpdesk.telemetry import setup as telemetry_setup
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
    assert trace.get_tracer_provider() is telemetry.provider


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
    with pytest.raises(TelemetryConfigError, match=r"already installed.*from the entry point"):
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


async def test_streamed_response_is_one_server_span(make_settings: Any) -> None:
    settings: Settings = make_settings()
    telemetry = setup_telemetry("agent", settings)
    app = FastAPI()

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            for n in range(20):
                yield f"data: {n}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    instrument_app(app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://agent.test"
    ) as client:
        assert (await client.get("/stream")).text.count("data:") == 20
    assert telemetry.flush()

    # Without exclude_spans every SSE event would add an "http send" span.
    assert [r["name"] for r in _records(settings.traces_dir / "agent.jsonl")] == ["GET /stream"]


async def test_plain_starlette_app_joins_the_callers_trace(make_settings: Any) -> None:
    # The MCP server serves FastMCP's Starlette app, not a FastAPI one.
    settings: Settings = make_settings()
    telemetry = setup_telemetry("mcp", settings)

    async def tool(request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/mcp", tool, methods=["POST"])])
    instrument_app(app)
    caller_trace, caller_span = "4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://mcp.test"
    ) as client:
        response = await client.post(
            "/mcp", headers={"traceparent": f"00-{caller_trace}-{caller_span}-01"}
        )
    assert response.text == "ok"
    assert telemetry.flush()

    (server,) = _records(settings.traces_dir / "mcp.jsonl")
    assert server["name"] == "POST /mcp"
    assert server["trace_id"] == caller_trace
    assert server["parent_span_id"] == caller_span
