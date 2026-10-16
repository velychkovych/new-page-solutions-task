"""The real entry points as processes: `python -m helpdesk.mock` under uvicorn on loopback,
and `python -m helpdesk.mock.seed --reset` run against it from a second process
(what `make seed` and the eval runner do)."""

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

API_TOKEN = "integration-api-token-5d1e"
ADMIN_TOKEN = "integration-admin-token-0b7c"
AUTH = {"Authorization": f"Bearer {API_TOKEN}"}
ADMIN = {**AUTH, "X-Admin-Token": ADMIN_TOKEN}
STARTUP_TIMEOUT_S = 20.0
CALLER_TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
CALLER_SPAN_ID = "b7ad6b7169203331"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    # Only what the mock needs; nothing from the developer's environment or .env leaks in,
    # so no OTLP endpoint is set and spans go only to the tmp traces dir.
    base = {k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "SYSTEMROOT", "TMPDIR"}}
    return {
        **base,
        "MOCK_DB_PATH": str(tmp_path / "data" / "mock.sqlite"),
        "TRACES_DIR": str(tmp_path / "process-traces"),
        **extra,
    }


@dataclass
class Server:
    url: str
    process: subprocess.Popen[str]
    env: dict[str, str]
    cwd: Path


@pytest.fixture
def server(tmp_path: Path) -> Iterator[Server]:
    port = _free_port()
    env = _env(
        tmp_path,
        MOCK_API_TOKEN=API_TOKEN,
        MOCK_ADMIN_TOKEN=ADMIN_TOKEN,
        MOCK_HOST="127.0.0.1",
        MOCK_PORT=str(port),
        LOG_LEVEL="DEBUG",
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "helpdesk.mock"],
        cwd=tmp_path,  # an empty dir, so no .env is picked up
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    try:
        while True:
            if process.poll() is not None:
                pytest.fail(f"mock exited early: {process.stdout.read() if process.stdout else ''}")
            try:
                if httpx.get(f"{url}/health", timeout=0.5).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                pytest.fail("mock did not become healthy")
            time.sleep(0.1)
        yield Server(url=url, process=process, env=env, cwd=tmp_path)
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_refuses_to_start_without_tokens(tmp_path: Path) -> None:
    for env in (
        _env(tmp_path),
        _env(tmp_path, MOCK_API_TOKEN=API_TOKEN),
        _env(tmp_path, MOCK_ADMIN_TOKEN=ADMIN_TOKEN, MOCK_API_TOKEN=""),
    ):
        result = subprocess.run(
            [sys.executable, "-m", "helpdesk.mock"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode != 0
        assert "is not set" in result.stderr
        assert API_TOKEN not in result.stderr
        assert ADMIN_TOKEN not in result.stderr
    assert not (tmp_path / "data" / "mock.sqlite").exists()
    assert not (tmp_path / "process-traces").exists()


def test_serves_seeded_data_and_resets_from_another_process(server: Server) -> None:
    with httpx.Client(base_url=server.url, timeout=5) as client:
        assert client.get("/tickets").status_code == 401
        traceparent = {"traceparent": f"00-{CALLER_TRACE_ID}-{CALLER_SPAN_ID}-01"}
        assert client.get("/tickets/INC-0003", headers={**AUTH, **traceparent}).status_code == 200
        first = client.get("/tickets", params={"q": "network"}, headers=AUTH).json()
        assert len(first["items"]) == 5
        assert first["next_cursor"] is not None

        body = {
            "title": "Docking station",
            "description": "USB-C dock does not charge.",
            "category": "hardware",
            "priority": "low",
            "reporter_id": "carol",
        }
        headers = {**AUTH, "Idempotency-Key": "proc-key-1"}
        created = client.post("/tickets", json=body, headers=headers)
        assert created.status_code == 201
        replay = client.post("/tickets", json=body, headers=headers)
        assert replay.headers["Idempotent-Replay"] == "true"
        assert client.get("/admin/writes", headers=ADMIN).json() == {
            "total": 1,
            "by_key": {"proc-key-1": 1},
        }

        reset = subprocess.run(
            [sys.executable, "-m", "helpdesk.mock.seed", "--reset"],
            cwd=server.cwd,
            env=server.env,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        assert "reset and seeded 15 tickets" in reset.stdout

        # The running server sees the reset immediately: nothing is cached in memory.
        assert client.get(f"/tickets/{created.json()['id']}", headers=AUTH).status_code == 404
        assert client.get("/admin/writes", headers=ADMIN).json() == {"total": 0, "by_key": {}}
        fresh = client.post("/tickets", json=body, headers=headers)
        assert fresh.status_code == 201
        assert "Idempotent-Replay" not in fresh.headers

    server.process.terminate()
    output, _ = server.process.communicate(timeout=10)
    assert "GET /tickets" in output  # access logs are on, and still carry no token
    assert API_TOKEN not in output
    assert ADMIN_TOKEN not in output

    # Telemetry was set up when the app started, and its shutdown flushed the spans.
    traces = Path(server.env["TRACES_DIR"]) / "mock.jsonl"
    raw = traces.read_text(encoding="utf-8")
    assert API_TOKEN not in raw
    assert ADMIN_TOKEN not in raw
    records = [json.loads(line) for line in raw.splitlines()]
    assert {r["service"] for r in records} == {"mock"}
    assert not [r for r in records if "/health" in r["name"]]
    (joined,) = [r for r in records if r["trace_id"] == CALLER_TRACE_ID]
    assert joined["name"] == "GET /tickets/{ticket_id}"
    assert joined["parent_span_id"] == CALLER_SPAN_ID
    assert {"POST /tickets", "GET /admin/writes"} <= {r["name"] for r in records}
    assert not (server.cwd / "traces").exists()
