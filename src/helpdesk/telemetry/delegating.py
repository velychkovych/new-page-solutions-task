"""A process-global tracer provider whose real (SDK) provider can be swapped.

The OTel API lets the global provider be set once per process, and every tracer handed out
binds to a provider for good:
- a `ProxyTracer` from `trace.get_tracer()` before setup binds to the first global provider
  it sees (opentelemetry-api 1.45.1, `ProxyTracer._tracer`);
- instrumentations (httpx, OpenInference, the ASGI middleware) keep the tracer they got
  when they were installed.

So the global provider is this delegating one, installed once, and each `setup_telemetry`
only swaps the SDK provider behind it. Every tracer it hands out looks the active delegate
up when a span starts, so a span always goes to whichever setup is active at that moment.
With no delegate the spans are non-recording, exactly as with no OTel SDK at all.
"""

from collections.abc import Iterator, Sequence
from typing import override

from opentelemetry.context import Context
from opentelemetry.trace import Link, NoOpTracer, Span, SpanKind, Tracer, TracerProvider

# The decorator the API's own ProxyTracer and NoOpTracer use, so `start_as_current_span`
# also works as a decorator on async functions; it has no public alias in 1.45.1.
from opentelemetry.util._decorator import _agnosticcontextmanager
from opentelemetry.util.types import Attributes

_NO_OP_TRACER = NoOpTracer()


class DelegatingTracerProvider(TracerProvider):
    def __init__(self) -> None:
        self._delegate: TracerProvider | None = None

    @property
    def delegate(self) -> TracerProvider | None:
        return self._delegate

    def set_delegate(self, delegate: TracerProvider | None) -> None:
        # A plain assignment: a span starting concurrently sees the old or the new one.
        self._delegate = delegate

    @override
    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: str | None = None,
        schema_url: str | None = None,
        attributes: Attributes = None,
    ) -> Tracer:
        return DelegatingTracer(
            self, instrumenting_module_name, instrumenting_library_version, schema_url, attributes
        )


class DelegatingTracer(Tracer):
    def __init__(
        self,
        provider: DelegatingTracerProvider,
        instrumenting_module_name: str,
        instrumenting_library_version: str | None,
        schema_url: str | None,
        attributes: Attributes,
    ) -> None:
        self._provider = provider
        self._name = instrumenting_module_name
        self._version = instrumenting_library_version
        self._schema_url = schema_url
        self._attributes = attributes
        # The delegate's tracer, cached until the delegate changes. One tuple, so a
        # concurrent reader never pairs one delegate with another delegate's tracer.
        self._bound: tuple[TracerProvider, Tracer] | None = None

    def _current(self) -> Tracer:
        delegate = self._provider.delegate
        if delegate is None:
            return _NO_OP_TRACER
        bound = self._bound
        if bound is not None and bound[0] is delegate:
            return bound[1]
        tracer = delegate.get_tracer(self._name, self._version, self._schema_url, self._attributes)
        self._bound = (delegate, tracer)
        return tracer

    @override
    def start_span(
        self,
        name: str,
        context: Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Attributes = None,
        links: Sequence[Link] | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> Span:
        # OpenInference's OITracer calls this unbound with its own proxy as `self`; the
        # proxy forwards `_current` here, so that works too.
        return self._current().start_span(
            name,
            context,
            kind,
            attributes,
            links,
            start_time,
            record_exception,
            set_status_on_exception,
        )

    @override
    @_agnosticcontextmanager
    def start_as_current_span(
        self,
        name: str,
        context: Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Attributes = None,
        links: Sequence[Link] | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> Iterator[Span]:
        # Resolved on entry, not when the decorator is applied, so a function decorated
        # at import time still traces into the setup active when it runs.
        with self._current().start_as_current_span(
            name,
            context,
            kind,
            attributes,
            links,
            start_time,
            record_exception,
            set_status_on_exception,
            end_on_exit,
        ) as span:
            yield span


# The one instance the process installs as the OTel global provider.
PROCESS_TRACER_PROVIDER = DelegatingTracerProvider()
