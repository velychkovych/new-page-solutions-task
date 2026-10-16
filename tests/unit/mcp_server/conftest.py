"""Fixtures for the MCP server tests. respx stubs the mock; nothing touches the network.

Test modules may import the helper classes from `tests.unit.mcp_server.conftest`: pytest
registers conftest under that name (importlib import mode) before collecting them.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

import httpx
import pytest
import respx
from mcp.client.session import ClientSession
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer

from helpdesk.mcp_server.mock_client import MockApiClient, create_mock_http_client
from helpdesk.mcp_server.redaction import SecretRedactor
from helpdesk.mcp_server.retry import RetryPolicy
from helpdesk.mcp_server.server import build_server
from helpdesk.shared.settings import Settings

MOCK_URL = "http://mock.test"
TOKEN = "tok-3f9a1c7e5b2d4a6f8e0c"


class FakeClock:
    """Monotonic clock that only moves when the code under test sleeps or a stub says so."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class RecordingUniform:
    """Fake RNG: records each (low, high) and returns `fraction` of the way up."""

    def __init__(self, fraction: float = 1.0) -> None:
        self.fraction = fraction
        self.calls: list[tuple[float, float]] = []

    def __call__(self, low: float, high: float) -> float:
        self.calls.append((low, high))
        return low + (high - low) * self.fraction


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "mock_url": MOCK_URL,
        "mock_api_token": TOKEN,
        "mcp_host": "127.0.0.1",
        "mcp_port": 8001,
        "retry_max_attempts": 3,
        "retry_backoff_base_s": 0.5,
        "retry_after_cap_s": 5.0,
        "tool_attempt_timeout_s": 5.0,
        "tool_deadline_s": 15.0,
        "trace_result_max_bytes": 2048,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.fixture
def token() -> str:
    return TOKEN


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def uniform() -> RecordingUniform:
    return RecordingUniform()


@pytest.fixture
def span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer(span_exporter: InMemorySpanExporter) -> Tracer:
    # A private provider, so these tests never touch the global tracer provider.
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider.get_tracer("tests.mcp_server")


@pytest.fixture
def mock_api() -> respx.Router:
    return respx.Router(base_url=MOCK_URL, assert_all_called=False)


type ServerFactory = Callable[..., FastMCP[Any]]


@pytest.fixture
def make_server(
    mock_api: respx.Router,
    clock: FakeClock,
    uniform: RecordingUniform,
    tracer: Tracer,
) -> ServerFactory:
    def factory(settings: Settings | None = None, *, global_tracer: bool = False) -> FastMCP[Any]:
        """`global_tracer=True` uses the process tracer, as the real server does, so spans
        go to whichever telemetry setup is active instead of the in-memory exporter."""
        settings = settings or make_settings()
        http = create_mock_http_client(
            settings, TOKEN, transport=httpx.MockTransport(mock_api.async_handler)
        )
        client = MockApiClient(
            http,
            RetryPolicy.from_settings(settings),
            clock=clock,
            sleep=clock.sleep,
            uniform=uniform,
        )
        return build_server(
            settings,
            client,
            redactor=SecretRedactor([TOKEN]),
            tracer=None if global_tracer else tracer,
        )

    return factory


type SessionFactory = Callable[..., AbstractAsyncContextManager[ClientSession]]


@pytest.fixture
def mcp_session(make_server: ServerFactory) -> SessionFactory:
    """An MCP client session over the real FastMCP request path (in-memory transport)."""

    @asynccontextmanager
    async def connect(settings: Settings | None = None) -> AsyncIterator[ClientSession]:
        async with create_connected_server_and_client_session(make_server(settings)) as session:
            yield session

    return connect
