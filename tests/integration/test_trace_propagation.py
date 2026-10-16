"""The three entry points as processes on loopback: one /chat request is one trace across
agent, MCP server and mock (CONTRACTS §6).

`python -m helpdesk.{mock,mcp_server,agent}` run as the compose services do, with the fake
model scripted to call search_tickets. No docker, no model API: the only network is
127.0.0.1. Each service writes <tmp>/traces/<service>.jsonl, flushed when it stops.
"""

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from helpdesk.shared import trace_attributes as attr

API_TOKEN = "integration-api-token-71c4"
ADMIN_TOKEN = "integration-admin-token-a93e"
STARTUP_TIMEOUT_S = 30.0
SCRIPT = [
    {"tool_calls": [{"name": "search_tickets", "args": {"query": "vpn"}, "id": "call_search"}]},
    {"content": "INC-0003 already tracks the VPN drops."},
]

type Span = dict[str, Any]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class Service:
    name: str
    url: str
    process: subprocess.Popen[str]
    output: str = ""

    def stop(self) -> None:
        """SIGTERM: uvicorn shuts the app down, and its lifespan flushes the spans."""
        if self.process.poll() is None:
            self.process.terminate()
        self.output, _ = self.process.communicate(timeout=20)


@dataclass
class Stack:
    tmp: Path
    services: dict[str, Service]

    @property
    def traces(self) -> Path:
        return self.tmp / "traces"

    def env(self, **extra: str) -> dict[str, str]:
        # Only what the services need: nothing from the developer's env or .env leaks in,
        # so no OTLP endpoint and no real model key.
        keep = {"PATH", "HOME", "SYSTEMROOT", "TMPDIR"}
        base = {k: v for k, v in os.environ.items() if k in keep}
        return {**base, "TRACES_DIR": str(self.traces), "LOG_LEVEL": "INFO", **extra}

    def start(self, name: str, module: str, port: int, **env: str) -> Service:
        process = subprocess.Popen(
            [sys.executable, "-m", module],
            cwd=self.tmp,  # no .env here
            env=self.env(**env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        service = Service(name, f"http://127.0.0.1:{port}", process)
        self.services[name] = service
        return service

    def wait_healthy(self, service: Service) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while True:
            if service.process.poll() is not None:
                service.stop()
                pytest.fail(f"{service.name} exited early:\n{service.output}")
            try:
                if httpx.get(f"{service.url}/health", timeout=0.5).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                pytest.fail(f"{service.name} did not become healthy")
            time.sleep(0.1)

    def stop_all(self) -> None:
        for service in reversed(list(self.services.values())):
            service.stop()

    def spans(self, service: str) -> list[Span]:
        path = self.traces / f"{service}.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def stack(tmp_path: Path) -> Iterator[Stack]:
    stack = Stack(tmp_path, {})
    script = tmp_path / "script.json"
    script.write_text(json.dumps(SCRIPT), encoding="utf-8")
    mock_port, mcp_port, agent_port = _free_port(), _free_port(), _free_port()
    try:
        mock = stack.start(
            "mock",
            "helpdesk.mock",
            mock_port,
            MOCK_HOST="127.0.0.1",
            MOCK_PORT=str(mock_port),
            MOCK_DB_PATH=str(tmp_path / "data" / "mock.sqlite"),
            MOCK_API_TOKEN=API_TOKEN,
            MOCK_ADMIN_TOKEN=ADMIN_TOKEN,
        )
        mcp = stack.start(
            "mcp",
            "helpdesk.mcp_server",
            mcp_port,
            MCP_HOST="127.0.0.1",
            MCP_PORT=str(mcp_port),
            MOCK_URL=mock.url,
            MOCK_API_TOKEN=API_TOKEN,
        )
        stack.wait_healthy(mock)
        stack.wait_healthy(mcp)
        # Like compose: the agent starts once the MCP server is healthy, since its
        # lifespan lists the tools.
        agent = stack.start(
            "agent",
            "helpdesk.agent",
            agent_port,
            AGENT_HOST="127.0.0.1",
            AGENT_PORT=str(agent_port),
            MCP_URL=f"{mcp.url}/mcp",
            MODEL_PROVIDER="fake",
            FAKE_MODEL_SCRIPT=str(script),
            CHECKPOINT_DB_PATH=str(tmp_path / "data" / "checkpoints.sqlite"),
            STORE_DB_PATH=str(tmp_path / "data" / "store.sqlite"),
        )
        stack.wait_healthy(agent)
        yield stack
    finally:
        stack.stop_all()


def _in_trace(spans: list[Span], trace_id: str) -> list[Span]:
    return [span for span in spans if span["trace_id"] == trace_id]


def test_one_chat_request_is_one_trace_across_agent_mcp_and_mock(stack: Stack) -> None:
    agent = stack.services["agent"]
    response = httpx.post(
        f"{agent.url}/chat",
        json={"user_id": "alice", "session_id": "s1", "message": "My VPN keeps dropping"},
        timeout=30,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok", body
    (call,) = body["tool_calls"]
    assert call["name"] == "search_tickets"
    assert call.get("error") is None
    # A real result from the seeded mock, through the MCP server.
    assert "INC-0003" in call["result_summary"]
    trace_id = body["trace_id"]
    assert len(trace_id) == 32
    assert trace_id != "0" * 32

    stack.stop_all()
    agent_spans = _in_trace(stack.spans("agent"), trace_id)
    mcp_spans = _in_trace(stack.spans("mcp"), trace_id)
    mock_spans = _in_trace(stack.spans("mock"), trace_id)

    assert attr.SPAN_REQUEST in {span["name"] for span in agent_spans}
    (tool,) = [span for span in mcp_spans if span["name"] == attr.SPAN_TOOL]
    assert tool["attributes"][attr.TOOL_NAME] == "search_tickets"
    (search,) = [span for span in mock_spans if span["name"] == "GET /tickets"]

    # Each hop's server span is a child of the caller's own span, not merely the same id:
    # agent (httpx client) → MCP "POST /mcp" that ran the tool → mcp.tool → httpx client
    # → mock "GET /tickets".
    agent_ids = {span["span_id"] for span in agent_spans}
    mcp_ids = {span["span_id"] for span in mcp_spans}
    (carrier,) = [span for span in mcp_spans if span["span_id"] == tool["parent_span_id"]]
    assert carrier["name"] == "POST /mcp"
    assert carrier["parent_span_id"] in agent_ids
    assert search["parent_span_id"] in mcp_ids

    for service in ("agent", "mcp", "mock"):
        raw = (stack.traces / f"{service}.jsonl").read_text(encoding="utf-8")
        assert API_TOKEN not in raw
        assert ADMIN_TOKEN not in raw
    for service in stack.services.values():
        assert API_TOKEN not in service.output
        assert ADMIN_TOKEN not in service.output


def test_agent_does_not_start_when_the_mcp_server_is_unreachable(tmp_path: Path) -> None:
    stack = Stack(tmp_path, {})
    script = tmp_path / "script.json"
    script.write_text(json.dumps(SCRIPT), encoding="utf-8")
    port = _free_port()
    agent = stack.start(
        "agent",
        "helpdesk.agent",
        port,
        AGENT_HOST="127.0.0.1",
        AGENT_PORT=str(port),
        # Nothing listens here: the port was free a moment ago and nothing took it.
        MCP_URL=f"http://127.0.0.1:{_free_port()}/mcp",
        MODEL_PROVIDER="fake",
        FAKE_MODEL_SCRIPT=str(script),
        CHECKPOINT_DB_PATH=str(tmp_path / "data" / "checkpoints.sqlite"),
        STORE_DB_PATH=str(tmp_path / "data" / "store.sqlite"),
    )
    try:
        agent.process.wait(timeout=STARTUP_TIMEOUT_S)
    finally:
        agent.stop()
    # uvicorn's startup-failure status: the lifespan's list_tools raised.
    assert agent.process.returncode == 3, agent.output
    assert "ToolTransportError" in agent.output
