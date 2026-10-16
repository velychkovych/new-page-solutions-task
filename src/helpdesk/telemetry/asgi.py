"""Server spans for a service's ASGI app; both calls belong in the app factory.

    app = FastAPI(telemetry=fastapi_telemetry_config(), lifespan=lifespan)
    instrument_app(app)

The app's lifespan then wraps its body in `telemetry_lifespan(...)` (see `.setup`).
`instrument_app` may run before any setup: it binds to the delegating global provider, so
the app's requests are traced whenever a setup is active and are no-ops otherwise.
"""

import threading
import weakref
from typing import Literal, cast

from fastapi import FastAPI
from fastapi.telemetry import TelemetryConfig
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from starlette.applications import Starlette

from helpdesk.telemetry.delegating import PROCESS_TRACER_PROVIDER
from helpdesk.telemetry.setup import TelemetryConfigError

# Compose health checks hit /health every 5 s; tracing them would bury the real requests.
_EXCLUDED_URLS = "/health$"
# The ASGI middleware adds a span per `receive` and per `send` message, which for an SSE
# stream is one per event. The server span already covers the whole request.
_EXCLUDED_ASGI_SPANS: list[Literal["receive", "send"]] = ["receive", "send"]

_lock = threading.Lock()
_instrumented: weakref.WeakSet[Starlette] = weakref.WeakSet()


def fastapi_telemetry_config() -> TelemetryConfig:
    """For `FastAPI(telemetry=...)`: switch off FastAPI 0.143's built-in OpenTelemetry.

    With a real global provider it is on by default. `instrument_app` already makes the
    server span, and the built-in parts would duplicate it or bypass our redaction:
    - tracing: a second server span, plus spans per dependency and endpoint
      (`operation_spans`);
    - metrics: the same HTTP server metrics as the ASGI instrumentation;
    - logs: unhandled exception messages and stack traces, which can hold input values,
      sent to the OTel logs pipeline that no redactor covers;
    - auto_configure: OTLP exporters added from env vars, outside the redacting exporter.
    A new dict each call, because FastAPI keeps the mapping it is given.
    """
    return {
        "tracing": False,
        "operation_spans": False,
        "metrics": False,
        "logs": False,
        "auto_configure": False,
    }


def instrument_app(app: Starlette) -> None:
    """Add server spans and `traceparent` extraction to an ASGI app. Idempotent per app.

    Takes any Starlette app, so FastMCP's `streamable_http_app()` works as well as FastAPI.
    Call it in the app factory: Starlette builds the middleware stack once, on the first
    lifespan or request message, and instrumentation added after that would never run.
    """
    with _lock:
        if app in _instrumented:
            return
        if app.middleware_stack is not None:
            raise TelemetryConfigError(
                "instrument_app() was called after the app started, so its requests would"
                " never be traced. Call it in the app factory, before the app is served"
            )
        # The instrumentor is typed for FastAPI, but it only uses Starlette's API
        # (build_middleware_stack, routes), so a plain Starlette app works the same.
        FastAPIInstrumentor.instrument_app(
            cast(FastAPI, app),
            tracer_provider=PROCESS_TRACER_PROVIDER,
            excluded_urls=_EXCLUDED_URLS,
            exclude_spans=_EXCLUDED_ASGI_SPANS,
        )
        _instrumented.add(app)
