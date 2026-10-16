import json
import logging
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from opentelemetry import _logs
from opentelemetry._logs import _internal as logs_internal
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
from opentelemetry.util._once import Once
from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from helpdesk.shared.settings import Settings
from helpdesk.telemetry import (
    TelemetryConfigError,
    fastapi_telemetry_config,
    instrument_app,
    setup_telemetry,
    shutdown_telemetry,
)

# A request body value that must never reach a span or a log.
BODY_MARKER = "body-value-7c1d9e4f"


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _client(app: Starlette, base_url: str = "http://agent.test") -> httpx.AsyncClient:
    # raise_app_exceptions=False: a 500 is a response here, as it is behind uvicorn.
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url=base_url)


async def test_instrumented_app_and_httpx_client_produce_linked_spans(
    make_settings: Any,
) -> None:
    settings: Settings = make_settings()
    app = FastAPI(telemetry=fastapi_telemetry_config())

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
    telemetry = setup_telemetry("agent", settings)

    with respx.mock(assert_all_called=True) as mock:
        mock.get("http://mock.test/tickets").respond(200, json={"items": []})
        async with _client(app) as client:
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
    app = FastAPI(telemetry=fastapi_telemetry_config())

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            for n in range(20):
                yield f"data: {n}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    instrument_app(app)
    telemetry = setup_telemetry("agent", settings)
    async with _client(app) as client:
        assert (await client.get("/stream")).text.count("data:") == 20
    assert telemetry.flush()

    # Without exclude_spans every SSE event would add an "http send" span.
    assert [r["name"] for r in _records(settings.traces_dir / "agent.jsonl")] == ["GET /stream"]


async def test_plain_starlette_app_joins_the_callers_trace(make_settings: Any) -> None:
    # The MCP server serves FastMCP's Starlette app, not a FastAPI one.
    settings: Settings = make_settings()

    async def tool(request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/mcp", tool, methods=["POST"])])
    instrument_app(app)
    telemetry = setup_telemetry("mcp", settings)
    caller_trace, caller_span = "4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7"
    async with _client(app, "http://mcp.test") as client:
        response = await client.post(
            "/mcp", headers={"traceparent": f"00-{caller_trace}-{caller_span}-01"}
        )
    assert response.text == "ok"
    assert telemetry.flush()

    (server,) = _records(settings.traces_dir / "mcp.jsonl")
    assert server["name"] == "POST /mcp"
    assert server["trace_id"] == caller_trace
    assert server["parent_span_id"] == caller_span


async def test_app_instrumented_before_setup_is_traced_once_setup_is_active(
    make_settings: Any, tmp_path: Path
) -> None:
    # As in an app factory: instrumented first, telemetry set up later by the lifespan.
    app = FastAPI(telemetry=fastapi_telemetry_config())

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    instrument_app(app)
    first: Settings = make_settings(traces_dir=tmp_path / "first")
    second: Settings = make_settings(traces_dir=tmp_path / "second")

    async with _client(app) as client:
        assert (await client.get("/ping")).status_code == 200  # before any setup
        setup_telemetry("agent", first)
        assert (await client.get("/ping")).status_code == 200
        shutdown_telemetry()
        assert (await client.get("/ping")).status_code == 200  # between setups
        setup_telemetry("agent", second)
        assert (await client.get("/ping")).status_code == 200
        shutdown_telemetry()

    assert [r["name"] for r in _records(first.traces_dir / "agent.jsonl")] == ["GET /ping"]
    assert [r["name"] for r in _records(second.traces_dir / "agent.jsonl")] == ["GET /ping"]


async def test_app_instrumented_during_one_setup_is_traced_into_the_next(
    make_settings: Any, tmp_path: Path
) -> None:
    first: Settings = make_settings(traces_dir=tmp_path / "first")
    second: Settings = make_settings(traces_dir=tmp_path / "second")
    setup_telemetry("agent", first)
    app = FastAPI(telemetry=fastapi_telemetry_config())

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    # Bound to the delegating provider, not to the setup that happens to be active.
    instrument_app(app)
    shutdown_telemetry()
    setup_telemetry("agent", second)
    async with _client(app) as client:
        assert (await client.get("/ping")).status_code == 200
    shutdown_telemetry()

    assert _records(first.traces_dir / "agent.jsonl") == []
    assert [r["name"] for r in _records(second.traces_dir / "agent.jsonl")] == ["GET /ping"]


async def test_instrument_app_after_the_app_started_raises() -> None:
    app = FastAPI(telemetry=fastapi_telemetry_config())

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    async with _client(app) as client:
        assert (await client.get("/ping")).status_code == 200

    with pytest.raises(TelemetryConfigError, match=r"after the app started.*app factory"):
        instrument_app(app)


@pytest.fixture
def otel_logs(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemoryLogRecordExporter]:
    """A real global OTel logger provider, so FastAPI's built-in logs would have a sink."""
    monkeypatch.setattr(logs_internal, "_LOGGER_PROVIDER_SET_ONCE", Once())
    monkeypatch.setattr(logs_internal, "_LOGGER_PROVIDER", None)
    exporter = InMemoryLogRecordExporter()
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    _logs.set_logger_provider(provider)
    yield exporter
    provider.shutdown()


class _Item(BaseModel):
    name: str
    quantity: int


def _items_app(telemetry: Any) -> FastAPI:
    app = FastAPI(telemetry=telemetry) if telemetry is not None else FastAPI()

    @app.post("/items")
    async def create_item(item: _Item) -> dict[str, int]:
        return {"quantity": item.quantity}

    @app.post("/explode")
    async def explode(item: _Item) -> dict[str, str]:
        raise RuntimeError("handler failed")

    return app


async def _post_items(app: FastAPI) -> None:
    async with _client(app) as client:
        valid = {"name": BODY_MARKER, "quantity": 1}
        assert (await client.post("/items", json=valid)).status_code == 200
        invalid = {"name": BODY_MARKER, "quantity": f"not-a-number {BODY_MARKER}"}
        assert (await client.post("/items", json=invalid)).status_code == 422
        assert (await client.post("/explode", json=valid)).status_code == 500


async def test_fastapi_config_with_instrument_app_gives_one_server_span_and_no_body_values(
    make_settings: Any,
    otel_logs: InMemoryLogRecordExporter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings: Settings = make_settings()
    app = _items_app(fastapi_telemetry_config())
    instrument_app(app)
    telemetry = setup_telemetry("agent", settings)

    with caplog.at_level(logging.DEBUG):
        await _post_items(app)
    assert telemetry.flush()

    path = settings.traces_dir / "agent.jsonl"
    records = _records(path)
    # One server span per request: no FastAPI-native server or operation spans.
    assert sorted(r["name"] for r in records) == ["POST /explode", "POST /items", "POST /items"]
    assert BODY_MARKER not in path.read_text(encoding="utf-8")
    assert otel_logs.get_finished_logs() == ()
    assert BODY_MARKER not in caplog.text


async def test_fastapi_config_switches_off_the_built_in_telemetry(
    make_settings: Any, otel_logs: InMemoryLogRecordExporter
) -> None:
    settings: Settings = make_settings()
    telemetry = setup_telemetry("agent", settings)
    path = settings.traces_dir / "agent.jsonl"

    # Control: with the defaults FastAPI traces and logs into the global providers itself.
    await _post_items(_items_app(None))
    assert telemetry.flush()
    assert any(r["name"].startswith("fastapi.") for r in _records(path))
    assert otel_logs.get_finished_logs() != ()
    path.unlink()
    otel_logs.clear()

    await _post_items(_items_app(fastapi_telemetry_config()))
    assert telemetry.flush()

    assert _records(path) == []
    assert otel_logs.get_finished_logs() == ()
