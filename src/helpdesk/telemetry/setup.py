"""Process-wide tracing setup: one tracer provider per service process (CONTRACTS §6).

Call `setup_telemetry(service_name)` once at startup, then `instrument_app(app)` for the
service's ASGI app, and `shutdown_telemetry()` on exit to flush the last batch.
"""

import re
import threading
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

from fastapi import FastAPI
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from starlette.applications import Starlette

from helpdesk.shared.settings import Settings, get_settings
from helpdesk.telemetry.jsonl_exporter import JsonlSpanExporter
from helpdesk.telemetry.redaction import RedactingSpanExporter, Redactor

# The service name becomes a file name under traces/.
_SERVICE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_OTLP_TRACES_PATH = "/v1/traces"
# Short, so `make trace` and the evals find a request's spans soon after the response.
_JSONL_SCHEDULE_DELAY_MS = 200
# Compose health checks hit /health every 5 s; tracing them would bury the real requests.
_EXCLUDED_URLS = "/health$"
# The ASGI middleware adds a span per `receive` and per `send` message, which for an SSE
# stream is one per event. The server span already covers the whole request.
_EXCLUDED_ASGI_SPANS: list[Literal["receive", "send"]] = ["receive", "send"]


class TelemetryConfigError(RuntimeError):
    """Telemetry was set up wrongly: bad service name, twice with different names, or not at all."""


@dataclass(frozen=True, slots=True)
class Telemetry:
    service_name: str
    provider: TracerProvider
    # Redacting wrappers, JSONL first, then OTLP when an endpoint is configured.
    exporters: tuple[RedactingSpanExporter, ...]

    def flush(self, timeout_millis: int = 30_000) -> bool:
        return self.provider.force_flush(timeout_millis)


class _Instrumentor(Protocol):
    # BaseInstrumentor's module is `# type: ignore`d upstream, so mypy cannot see it.
    @property
    def is_instrumented_by_opentelemetry(self) -> bool: ...
    def instrument(self, **kwargs: Any) -> object: ...
    def uninstrument(self, **kwargs: Any) -> object: ...


@dataclass
class _State:
    lock: threading.Lock = field(default_factory=threading.Lock)
    active: Telemetry | None = None
    # Instrumentors are process singletons; remember which ones we switched on.
    instrumented: list[_Instrumentor] = field(default_factory=list)
    apps: list[Starlette] = field(default_factory=list)


_state = _State()


def otlp_traces_endpoint(base_url: str) -> str:
    """OTLPSpanExporter's `endpoint` is the full signal URL; the env var holds the base URL."""
    base = base_url.rstrip("/")
    return base if base.endswith(_OTLP_TRACES_PATH) else base + _OTLP_TRACES_PATH


def build_telemetry(service_name: str, settings: Settings) -> Telemetry:
    """A configured provider that is neither global nor instrumented (also used by tests)."""
    if not _SERVICE_NAME_RE.fullmatch(service_name):
        raise TelemetryConfigError(f"invalid service name {service_name!r}")
    redactor = Redactor.from_settings(settings)
    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))

    jsonl = RedactingSpanExporter(JsonlSpanExporter(settings.traces_dir, service_name), redactor)
    provider.add_span_processor(
        BatchSpanProcessor(jsonl, schedule_delay_millis=_JSONL_SCHEDULE_DELAY_MS)
    )
    exporters = [jsonl]

    if settings.otel_exporter_otlp_endpoint:
        otlp = RedactingSpanExporter(
            OTLPSpanExporter(endpoint=otlp_traces_endpoint(settings.otel_exporter_otlp_endpoint)),
            redactor,
        )
        provider.add_span_processor(BatchSpanProcessor(otlp))
        exporters.append(otlp)

    return Telemetry(service_name=service_name, provider=provider, exporters=tuple(exporters))


def _instrumentors() -> tuple[_Instrumentor, ...]:
    return (HTTPXClientInstrumentor(), LangChainInstrumentor())


def setup_telemetry(service_name: str, settings: Settings | None = None) -> Telemetry:
    """Install the global tracer provider and the httpx and LangChain instrumentation.

    Safe to call again with the same name (returns the active setup). The OTel API lets
    the global provider be set only once per process, so setup after shutdown, or over a
    provider someone else installed, raises instead of silently tracing into nothing.
    Pair it with `shutdown_telemetry()` in the process entry point, not in an app
    lifespan that may start several times in one process (tests).
    """
    with _state.lock:
        if _state.active is not None:
            if _state.active.service_name != service_name:
                raise TelemetryConfigError(
                    f"telemetry is already set up for {_state.active.service_name!r}"
                )
            return _state.active

        telemetry = build_telemetry(service_name, settings or get_settings())
        trace.set_tracer_provider(telemetry.provider)
        if trace.get_tracer_provider() is not telemetry.provider:
            telemetry.provider.shutdown()
            raise TelemetryConfigError(
                "a global tracer provider is already installed: setup_telemetry() works once"
                " per process, so call it from the entry point, not from an app lifespan"
            )

        for instrumentor in _instrumentors():
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument(tracer_provider=telemetry.provider)
                _state.instrumented.append(instrumentor)
        _state.active = telemetry
        return telemetry


def instrument_app(app: Starlette) -> None:
    """Add server spans and `traceparent` extraction to an ASGI app.

    Takes any Starlette app, so FastMCP's `streamable_http_app()` works as well as FastAPI.
    Call after `setup_telemetry`, before serving.
    """
    with _state.lock:
        if _state.active is None:
            raise TelemetryConfigError("call setup_telemetry() before instrument_app()")
        if any(app is known for known in _state.apps):
            return
        # The instrumentor is typed for FastAPI, but it only uses Starlette's API
        # (build_middleware_stack, routes), so a plain Starlette app works the same.
        FastAPIInstrumentor.instrument_app(
            cast(FastAPI, app),
            tracer_provider=_state.active.provider,
            excluded_urls=_EXCLUDED_URLS,
            exclude_spans=_EXCLUDED_ASGI_SPANS,
        )
        _state.apps.append(app)


def shutdown_telemetry() -> None:
    """Flush and close the exporters and undo our instrumentation. Idempotent."""
    with _state.lock:
        telemetry = _state.active
        if telemetry is None:
            return
        for app in _state.apps:
            FastAPIInstrumentor.uninstrument_app(cast(FastAPI, app))
        for instrumentor in _state.instrumented:
            instrumentor.uninstrument()
        _state.apps.clear()
        _state.instrumented.clear()
        _state.active = None
        # Shutdown flushes the batch processors first, so the last request is not lost.
        telemetry.provider.shutdown()
