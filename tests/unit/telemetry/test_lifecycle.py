"""Setup and shutdown from app lifespans: repeated, per service, reference counted."""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.trace import INVALID_TRACE_ID, ProxyTracer

from helpdesk.shared.settings import Settings
from helpdesk.telemetry import (
    TelemetryConfigError,
    setup_telemetry,
    shutdown_telemetry,
    telemetry_lifespan,
)
from helpdesk.telemetry.delegating import PROCESS_TRACER_PROVIDER

# Obtained at import, before any setup, like `tracer = trace.get_tracer(__name__)` in a
# service module.
_MODULE_TRACER = trace.get_tracer("tests.telemetry.module_level")


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _emit(name: str) -> None:
    with trace.get_tracer("test").start_as_current_span(name):
        pass


def test_setup_shutdown_setup_again_in_one_process(make_settings: Any, tmp_path: Path) -> None:
    first: Settings = make_settings(traces_dir=tmp_path / "first")
    second: Settings = make_settings(traces_dir=tmp_path / "second")

    one = setup_telemetry("agent", first)
    _emit("first-run")
    shutdown_telemetry()
    two = setup_telemetry("agent", second)
    _emit("second-run")
    shutdown_telemetry()

    assert two is not one
    assert two.provider is not one.provider
    assert [r["name"] for r in _records(first.traces_dir / "agent.jsonl")] == ["first-run"]
    assert [r["name"] for r in _records(second.traces_dir / "agent.jsonl")] == ["second-run"]


def test_service_a_then_b_writes_to_b_file_as_b(make_settings: Any) -> None:
    settings: Settings = make_settings()

    setup_telemetry("agent", settings)
    _emit("agent-span")
    shutdown_telemetry()
    mcp = setup_telemetry("mcp", settings)
    _emit("mcp-span")
    shutdown_telemetry()

    agent_records = _records(settings.traces_dir / "agent.jsonl")
    mcp_records = _records(settings.traces_dir / "mcp.jsonl")
    assert [(r["name"], r["service"]) for r in agent_records] == [("agent-span", "agent")]
    assert [(r["name"], r["service"]) for r in mcp_records] == [("mcp-span", "mcp")]
    assert mcp.provider.resource.attributes["service.name"] == "mcp"


def test_same_service_setup_is_reference_counted(make_settings: Any) -> None:
    settings: Settings = make_settings()
    first = setup_telemetry("agent", settings)
    assert setup_telemetry("agent", settings) is first

    shutdown_telemetry()
    # One user is left, so the setup is still active and still recording.
    assert PROCESS_TRACER_PROVIDER.delegate is first.provider
    _emit("still-recorded")

    shutdown_telemetry()
    assert PROCESS_TRACER_PROVIDER.delegate is None
    shutdown_telemetry()  # extra shutdowns are no-ops
    assert [r["name"] for r in _records(settings.traces_dir / "agent.jsonl")] == ["still-recorded"]


def test_a_different_service_while_one_is_active_raises(make_settings: Any) -> None:
    settings: Settings = make_settings()
    agent = setup_telemetry("agent", settings)

    with pytest.raises(TelemetryConfigError, match=r"already set up for service 'agent'.*'mcp'"):
        setup_telemetry("mcp", settings)

    # The refused call counted no user: one shutdown still ends the agent's setup.
    assert PROCESS_TRACER_PROVIDER.delegate is agent.provider
    shutdown_telemetry()
    assert PROCESS_TRACER_PROVIDER.delegate is None


def test_tracer_obtained_before_any_setup_records_into_the_active_setup(
    make_settings: Any,
) -> None:
    settings: Settings = make_settings()
    early = trace.get_tracer("tests.telemetry.early")
    assert isinstance(early, ProxyTracer)

    setup_telemetry("agent", settings)
    with early.start_as_current_span("early-tracer"):
        pass
    with _MODULE_TRACER.start_as_current_span("module-tracer-agent"):
        pass
    shutdown_telemetry()
    # A ProxyTracer binds to the first global provider for good; that provider is the
    # delegating one, so the same tracer follows the next setup too.
    setup_telemetry("mcp", settings)
    with _MODULE_TRACER.start_as_current_span("module-tracer-mcp"):
        pass
    shutdown_telemetry()

    assert [r["name"] for r in _records(settings.traces_dir / "agent.jsonl")] == [
        "early-tracer",
        "module-tracer-agent",
    ]
    assert [r["name"] for r in _records(settings.traces_dir / "mcp.jsonl")] == ["module-tracer-mcp"]


def test_nothing_is_recorded_or_written_after_shutdown(make_settings: Any) -> None:
    settings: Settings = make_settings()
    setup_telemetry("agent", settings)
    shutdown_telemetry()

    with trace.get_tracer("test").start_as_current_span("after-shutdown") as span:
        assert not span.is_recording()
        assert span.get_span_context().trace_id == INVALID_TRACE_ID
    with _MODULE_TRACER.start_as_current_span("after-shutdown-module") as span:
        assert not span.is_recording()

    assert not settings.traces_dir.exists()


async def test_lifespan_sets_up_on_entry_and_shuts_down_on_exit(make_settings: Any) -> None:
    settings: Settings = make_settings()

    async with telemetry_lifespan("agent", settings) as telemetry:
        assert PROCESS_TRACER_PROVIDER.delegate is telemetry.provider
        # A nested lifespan for the same service shares the setup.
        async with telemetry_lifespan("agent", settings) as nested:
            assert nested is telemetry
        assert PROCESS_TRACER_PROVIDER.delegate is telemetry.provider
        _emit("served")

    assert PROCESS_TRACER_PROVIDER.delegate is None
    # Flushed by the exit itself, with no explicit flush.
    assert [r["name"] for r in _records(settings.traces_dir / "agent.jsonl")] == ["served"]


async def test_lifespan_shuts_down_when_the_body_raises(make_settings: Any) -> None:
    settings: Settings = make_settings()

    async def failing_startup() -> None:
        async with telemetry_lifespan("agent", settings):
            _emit("before-error")
            raise RuntimeError("startup failed")

    with pytest.raises(RuntimeError, match="startup failed"):
        await failing_startup()

    assert PROCESS_TRACER_PROVIDER.delegate is None
    assert [r["name"] for r in _records(settings.traces_dir / "agent.jsonl")] == ["before-error"]
    # The setup is really gone: another service can start in the same process.
    async with telemetry_lifespan("mcp", settings) as mcp:
        assert mcp.service_name == "mcp"


async def test_lifespan_releases_the_setup_when_cancelled(make_settings: Any) -> None:
    settings: Settings = make_settings()
    entered = asyncio.Event()

    async def serve() -> None:
        async with telemetry_lifespan("agent", settings):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(serve())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert PROCESS_TRACER_PROVIDER.delegate is None
    async with telemetry_lifespan("mcp", settings) as mcp:
        assert mcp.service_name == "mcp"


async def test_lifespan_exit_only_releases_the_setup_it_entered(make_settings: Any) -> None:
    settings: Settings = make_settings()

    async with telemetry_lifespan("agent", settings):
        # Unbalanced: closes the lifespan's setup from inside it.
        shutdown_telemetry()
        mcp = setup_telemetry("mcp", settings)

    # The lifespan's exit must not count against a setup it did not make.
    assert PROCESS_TRACER_PROVIDER.delegate is mcp.provider
    _emit("mcp-span")
    shutdown_telemetry()
    assert PROCESS_TRACER_PROVIDER.delegate is None
    [record] = _records(settings.traces_dir / "mcp.jsonl")
    assert (record["name"], record["service"]) == ("mcp-span", "mcp")


async def test_lifespan_exit_still_releases_a_shared_same_service_setup(
    make_settings: Any,
) -> None:
    settings: Settings = make_settings()

    outer = setup_telemetry("agent", settings)
    async with telemetry_lifespan("agent", settings) as inner:
        assert inner is outer
    # The lifespan gave back its own count only; the outer user still holds the setup.
    assert PROCESS_TRACER_PROVIDER.delegate is outer.provider
    shutdown_telemetry()
    assert PROCESS_TRACER_PROVIDER.delegate is None
