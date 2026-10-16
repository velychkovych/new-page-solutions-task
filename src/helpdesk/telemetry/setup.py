"""Tracing setup for one service process (CONTRACTS §6), driven by the app's lifespan.

How a service uses it:
- App factory: `app = FastAPI(telemetry=fastapi_telemetry_config(), ...)`, then
  `instrument_app(app)` before the app is returned (see `helpdesk.telemetry.asgi`).
- Lifespan: `async with telemetry_lifespan("<service>", settings):` around everything else,
  so spans are recorded for the app's whole life and flushed when it stops.

`setup_telemetry` / `shutdown_telemetry` are the same thing without the context manager.
They can run many times in one process (tests start many apps one after the other):
- The OTel global provider is set once per process, so the global is a delegating provider
  (`helpdesk.telemetry.delegating`) installed on the first setup, and httpx and LangChain
  are instrumented once against it.
- Each setup builds a fresh SDK provider (service resource, JSONL and optional OTLP behind
  the redactor, the cache-write processor) and makes it the delegate.
- Setup again with the same service name while it is active returns the active setup and
  counts one more user; only the last shutdown flushes and closes it.
- A different service name while one is active raises: one process is one service.
- After the last shutdown nothing is recorded until the next setup.
"""

import asyncio
import re
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from helpdesk.shared.settings import Settings, get_settings
from helpdesk.telemetry.cache_write import CacheWriteSpanProcessor
from helpdesk.telemetry.delegating import PROCESS_TRACER_PROVIDER
from helpdesk.telemetry.jsonl_exporter import JsonlSpanExporter
from helpdesk.telemetry.redaction import RedactingSpanExporter, Redactor

# The service name becomes a file name under traces/.
_SERVICE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_OTLP_TRACES_PATH = "/v1/traces"
# Short, so `make trace` and the evals find a request's spans soon after the response.
_JSONL_SCHEDULE_DELAY_MS = 200


class TelemetryConfigError(RuntimeError):
    """Telemetry was set up wrongly: a bad service name, a second service in one process,
    another global tracer provider, or an app instrumented after it started."""


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
    # setup_telemetry calls for the active service not yet matched by a shutdown.
    users: int = 0
    # Instrumentors are process singletons; remember which ones we switched on.
    instrumented: list[_Instrumentor] = field(default_factory=list)


_state = _State()


def otlp_traces_endpoint(base_url: str) -> str:
    """OTLPSpanExporter's `endpoint` is the full signal URL; the env var holds the base URL."""
    base = base_url.rstrip("/")
    return base if base.endswith(_OTLP_TRACES_PATH) else base + _OTLP_TRACES_PATH


def _check_service_name(service_name: str) -> None:
    if not _SERVICE_NAME_RE.fullmatch(service_name):
        raise TelemetryConfigError(
            f"invalid service name {service_name!r}: it names the trace file, so use 1-64"
            " characters from a-z, 0-9, '_' and '-', starting with a letter or digit"
        )


def build_telemetry(service_name: str, settings: Settings) -> Telemetry:
    """A configured SDK provider that is not made active (setup_telemetry and tests use it)."""
    _check_service_name(service_name)
    redactor = Redactor.from_settings(settings)
    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))

    jsonl = RedactingSpanExporter(JsonlSpanExporter(settings.traces_dir, service_name), redactor)
    exporters = [jsonl]
    processors: list[SpanProcessor] = [
        BatchSpanProcessor(jsonl, schedule_delay_millis=_JSONL_SCHEDULE_DELAY_MS)
    ]
    if settings.otel_exporter_otlp_endpoint:
        otlp = RedactingSpanExporter(
            OTLPSpanExporter(endpoint=otlp_traces_endpoint(settings.otel_exporter_otlp_endpoint)),
            redactor,
        )
        exporters.append(otlp)
        processors.append(BatchSpanProcessor(otlp))
    provider.add_span_processor(CacheWriteSpanProcessor(processors))

    return Telemetry(service_name=service_name, provider=provider, exporters=tuple(exporters))


def _install_global_provider() -> None:
    current = trace.get_tracer_provider()
    if current is PROCESS_TRACER_PROVIDER:
        return
    if isinstance(current, trace.ProxyTracerProvider):
        trace.set_tracer_provider(PROCESS_TRACER_PROVIDER)
        if trace.get_tracer_provider() is PROCESS_TRACER_PROVIDER:
            return
        current = trace.get_tracer_provider()
    raise TelemetryConfigError(
        f"another global tracer provider ({type(current).__qualname__}) is already installed,"
        " and the OTel API cannot replace it, so spans would bypass the JSONL files and"
        " redaction. Remove whatever installs it (for example OTEL_PYTHON_TRACER_PROVIDER or"
        " FASTAPI_OTEL_AUTO_CONFIGURE) and let setup_telemetry() install the provider"
    )


def _instrument_libraries() -> None:
    """httpx and LangChain, once per process, against the delegating provider."""
    for instrumentor in (HTTPXClientInstrumentor(), LangChainInstrumentor()):
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument(tracer_provider=PROCESS_TRACER_PROVIDER)
            _state.instrumented.append(instrumentor)


def setup_telemetry(service_name: str, settings: Settings | None = None) -> Telemetry:
    """Make `service_name`'s tracing active and return it.

    With the same name while it is active: returns the active setup (its settings are kept)
    and counts one more user, so pair every call with one `shutdown_telemetry()`.
    """
    _check_service_name(service_name)
    with _state.lock:
        active = _state.active
        if active is not None:
            if active.service_name != service_name:
                raise TelemetryConfigError(
                    f"telemetry is already set up for service {active.service_name!r}, so it"
                    f" cannot also be set up for {service_name!r}: one process runs one"
                    " service. Shut the active one down first (end its telemetry_lifespan)"
                )
            _state.users += 1
            return active

        telemetry = build_telemetry(service_name, settings or get_settings())
        try:
            _install_global_provider()
            _instrument_libraries()
        except BaseException:
            telemetry.provider.shutdown()
            raise
        PROCESS_TRACER_PROVIDER.set_delegate(telemetry.provider)
        _state.active = telemetry
        _state.users = 1
        return telemetry


def _release(entered: Telemetry | None = None) -> Telemetry | None:
    """Count one shutdown; the setup to close when that was its last user.

    With `entered`, only that setup is counted down: a lifespan whose setup was already
    closed by an unbalanced `shutdown_telemetry()` must not close a later one.
    """
    with _state.lock:
        active = _state.active
        if active is None or (entered is not None and active is not entered):
            return None
        _state.users -= 1
        if _state.users > 0:
            return None
        # Spans that start from now on are non-recording; ones already started on this
        # provider still end into it and are flushed by its shutdown.
        PROCESS_TRACER_PROVIDER.set_delegate(None)
        _state.active = None
        return active


def shutdown_telemetry() -> None:
    """Count one shutdown; the last one flushes and closes the exporters. Extra calls are no-ops.

    It blocks while the batch processors flush (an OTLP export can take seconds), so async
    code should use `telemetry_lifespan`, which flushes in a worker thread.
    """
    if (released := _release()) is not None:
        # Shutdown flushes the batch processors first, so the last request is not lost.
        released.provider.shutdown()


@asynccontextmanager
async def telemetry_lifespan(
    service_name: str, settings: Settings | None = None
) -> AsyncIterator[Telemetry]:
    """Set up on entry; shut down on exit, also when the body raises or is cancelled."""
    telemetry = setup_telemetry(service_name, settings)
    try:
        yield telemetry
    finally:
        # The bookkeeping happens here, synchronously, so a setup that follows straight
        # after (even if this task is cancelled mid-flush) starts a fresh provider.
        if (released := _release(telemetry)) is not None:
            await asyncio.to_thread(released.provider.shutdown)


def _reset_for_tests() -> None:
    """Back to a fresh process: close any setup whatever its users, undo instrumentation.

    Only for test isolation; a service never needs it.
    """
    with _state.lock:
        active = _state.active
        _state.active = None
        _state.users = 0
        PROCESS_TRACER_PROVIDER.set_delegate(None)
        for instrumentor in _state.instrumented:
            instrumentor.uninstrument()
        _state.instrumented.clear()
    if active is not None:
        active.provider.shutdown()
