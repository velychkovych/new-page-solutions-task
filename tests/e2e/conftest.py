"""Helpers for the e2e tests: the compose stack with `agent-e2e` (`make test-e2e`).

The agent under test is `agent-e2e` on 127.0.0.1:8081: the real image, the real MCP
server and mock, and the scripted fake model from tests/e2e/fixtures. Every reply step is
chosen from the conversation (helpdesk.agent.fake_model), so one script serves every test:

    search_tickets("printer") -> get_ticket(INC-0011) -> create_ticket (proposal) -> "".

The empty last step makes the reply after an approved write the agent's own deterministic
text about the write's outcome, so a test can tell a saved write from a failed one.

The mock publishes no port. Its admin API and its data are read from inside the mock
container (`docker compose exec mock`), which holds both tokens; no token reaches the
host, and none is ever printed.
"""

import json
import re
import subprocess
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
import pytest
from pydantic import BaseModel

from helpdesk.shared.api import (
    ChatResponse,
    DoneData,
    ErrorData,
    PendingActionView,
    SseEvent,
    StatusData,
    TokenData,
    ToolEndData,
    ToolStartData,
)

ROOT = Path(__file__).resolve().parents[2]
AGENT_URL = "http://127.0.0.1:8081"
E2E_SERVICE = "agent-e2e"
USER_ID = "alice"
SCRIPT = json.loads((ROOT / "tests/e2e/fixtures/fake_model_script.json").read_text("utf-8"))
PROPOSED_ARGS: dict[str, Any] = SCRIPT[2]["tool_calls"][0]["args"]
# Matches only the ticket the script creates: the seed has a floor 2 printer, not floor 3.
CREATED_QUERY = "printer floor 3"
TRACES = ROOT / "traces"
TRACE_FILES = {
    "agent": TRACES / "e2e" / "agent.jsonl",
    "mcp": TRACES / "mcp.jsonl",
    "mock": TRACES / "mock.jsonl",
}
CHAOS_OFF: dict[str, Any] = {
    "enabled": False,
    "rate_limit_rate": 0,
    "error_rate": 0,
    "slow_rate": 0,
    "slow_ms": 0,
}
HEALTH_TIMEOUT_S = 60.0
TRACE_FLUSH_TIMEOUT_S = 10.0

# Runs inside the mock container. Prints only the status and the response body.
_MOCK_REQUEST = """
import json, os, sys, urllib.error, urllib.request
method, path, body = sys.argv[1], sys.argv[2], sys.argv[3]
headers = {"Authorization": "Bearer " + os.environ["MOCK_API_TOKEN"]}
if path.startswith("/admin/"):
    headers["X-Admin-Token"] = os.environ["MOCK_ADMIN_TOKEN"]
if body:
    headers["Content-Type"] = "application/json"
request = urllib.request.Request(
    "http://127.0.0.1:8000" + path, data=body.encode() or None, method=method, headers=headers
)
try:
    with urllib.request.urlopen(request, timeout=10) as response:
        status, raw = response.status, response.read()
except urllib.error.HTTPError as error:
    status, raw = error.code, error.read()
print(json.dumps({"status": status, "body": json.loads(raw or b"null")}))
"""


def compose(*args: str, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", "--profile", "e2e", *args],
        cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False,
    )  # fmt: skip


def _checked(result: subprocess.CompletedProcess[str]) -> str:
    assert result.returncode == 0, f"{result.args[:6]} failed: {result.stderr[-2000:]}"
    return result.stdout


# --- the mock, from inside its container -------------------------------------------------


class Mock:
    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        payload = json.dumps(body) if body is not None else ""
        out = _checked(
            compose("exec", "-T", "mock", "python", "-c", _MOCK_REQUEST, method, path, payload)
        )
        return json.loads(out)

    def ok(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        response = self.request(method, path, body)
        assert response["status"] == 200, response
        return response["body"]

    def reset(self) -> None:
        _checked(compose("exec", "-T", "mock", "python", "-m", "helpdesk.mock.seed", "--reset"))

    def chaos(self, **config: Any) -> None:
        self.ok("POST", "/admin/chaos", {**CHAOS_OFF, **config})

    def chaos_off(self) -> None:
        self.ok("POST", "/admin/chaos", CHAOS_OFF)

    def writes(self) -> dict[str, Any]:
        report: dict[str, Any] = self.ok("GET", "/admin/writes")
        return report

    def search(self, query: str) -> list[dict[str, Any]]:
        result = self.ok("GET", "/tickets?" + urlencode({"q": query, "limit": 20}))
        assert result["next_cursor"] is None, result
        items: list[dict[str, Any]] = result["items"]
        return items

    def ticket(self, ticket_id: str) -> dict[str, Any]:
        ticket: dict[str, Any] = self.ok("GET", f"/tickets/{ticket_id}")
        return ticket


def assert_exactly_one_write(mock: Mock) -> None:
    writes = mock.writes()
    assert writes["total"] == 1, writes
    assert list(writes["by_key"].values()) == [1], writes


def created_ticket(mock: Mock) -> dict[str, Any]:
    """The one ticket the script creates, read from the mock itself."""
    (summary,) = mock.search(CREATED_QUERY)
    ticket = mock.ticket(summary["id"])
    assert ticket["title"] == PROPOSED_ARGS["title"]
    assert ticket["description"] == PROPOSED_ARGS["description"]
    # Injected by execute_write from the request, never chosen by the model.
    assert ticket["reporter_id"] == USER_ID
    return ticket


# --- the agent -------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    name: SseEvent
    data: Any


DATA_MODELS: dict[SseEvent, type[BaseModel]] = {
    SseEvent.TOKEN: TokenData,
    SseEvent.TOOL_START: ToolStartData,
    SseEvent.TOOL_END: ToolEndData,
    SseEvent.STATUS: StatusData,
    SseEvent.APPROVAL_REQUIRED: PendingActionView,
    SseEvent.ERROR: ErrorData,
    SseEvent.DONE: DoneData,
}
FRAME = re.compile(r"event: (?P<name>[a-z_]+)\ndata: (?P<data>[^\n]*)\n\n")
COMMENT = re.compile(r"(?::[^\n]*\n)+\n")


def parse_sse(body: str) -> list[Event]:
    """Strict: every frame is `event:`/`data:` with compact JSON valid for its model."""
    events: list[Event] = []
    rest = body
    while rest:
        if comment := COMMENT.match(rest):
            rest = rest[comment.end() :]
            continue
        frame = FRAME.match(rest)
        assert frame is not None, f"not an SSE frame: {rest[:120]!r}"
        name = SseEvent(frame["name"])
        data = DATA_MODELS[name].model_validate_json(frame["data"])
        assert frame["data"] == data.model_dump_json()
        events.append(Event(name, data))
        rest = rest[frame.end() :]
    return events


def assert_contract_order(events: list[Event]) -> None:
    """The CONTRACTS §4 ordering rules."""
    names = [e.name for e in events]
    assert names.count(SseEvent.DONE) == 1, names
    assert names[-1] is SseEvent.DONE, names
    assert names.count(SseEvent.ERROR) <= 1, names
    assert names.count(SseEvent.APPROVAL_REQUIRED) <= 1, names
    if events[-1].data.status == "awaiting_approval":
        assert names[-2] is SseEvent.APPROVAL_REQUIRED, names
    else:
        assert SseEvent.APPROVAL_REQUIRED not in names, names
    open_calls: set[str] = set()
    for event in events:
        if event.name is SseEvent.TOOL_START:
            assert event.data.call_id not in open_calls
            open_calls.add(event.data.call_id)
        elif event.name is SseEvent.TOOL_END:
            assert event.data.call_id in open_calls, "tool_end without its tool_start"
            open_calls.remove(event.data.call_id)
    assert not open_calls, open_calls


def named(events: list[Event], name: SseEvent) -> list[Any]:
    return [e.data for e in events if e.name is name]


def tokens(events: list[Event]) -> str:
    return "".join(e.data.text for e in events if e.name is SseEvent.TOKEN)


class Agent:
    def __init__(self, client: httpx.Client, session_id: str) -> None:
        self.client = client
        self.session_id = session_id

    def _body(self, message: str | None, action_id: str | None, approve: bool) -> dict[str, Any]:
        body: dict[str, Any] = {"user_id": USER_ID, "session_id": self.session_id}
        if action_id is not None:
            body["decision"] = {"action_id": action_id, "approve": approve}
        else:
            body["message"] = message
        return body

    def chat(
        self, message: str | None = None, *, action_id: str | None = None, approve: bool = True
    ) -> ChatResponse:
        response = self.client.post("/chat", json=self._body(message, action_id, approve))
        assert response.status_code == 200, response.text
        return ChatResponse.model_validate_json(response.text)

    def stream(
        self, message: str | None = None, *, action_id: str | None = None, approve: bool = True
    ) -> list[Event]:
        response = self.client.post("/chat/stream", json=self._body(message, action_id, approve))
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(response.text)
        assert_contract_order(events)
        return events

    def propose(self) -> PendingActionView:
        """Run the script's proposal turn and check it wrote nothing."""
        response = self.chat("The printer on floor 3 keeps jamming on double-sided jobs.")
        assert response.status.value == "awaiting_approval", response
        assert [c.name for c in response.tool_calls] == ["search_tickets", "get_ticket"]
        assert all(c.error is None for c in response.tool_calls), response.tool_calls
        pending = response.pending_action
        assert pending is not None
        assert pending.tool == "create_ticket"
        assert pending.args == PROPOSED_ARGS
        return pending


def restart_e2e_agent() -> None:
    _checked(compose("restart", E2E_SERVICE))
    wait_until_healthy()


def wait_until_healthy(url: str = AGENT_URL, timeout_s: float = HEALTH_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                return
        except httpx.TransportError:
            pass
        assert time.monotonic() < deadline, f"{url} not healthy after {timeout_s} s"
        time.sleep(0.5)


# --- traces on the host (./traces is bind-mounted into every service) --------------------


def spans_in_trace(service: str, trace_id: str) -> list[dict[str, Any]]:
    path = TRACE_FILES[service]
    if not path.exists():
        return []
    spans = []
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            if not line.endswith("\n"):
                # The exporter is still appending this line; the next poll reads it whole.
                break
            if trace_id in line:
                span = json.loads(line)
                if span["trace_id"] == trace_id:
                    spans.append(span)
    return spans


def wait_for_spans(
    service: str,
    trace_id: str,
    name: str,
    timeout_s: float = TRACE_FLUSH_TIMEOUT_S,
    *,
    count: int = 1,
) -> list[dict[str, Any]]:
    """The trace's spans of this service once `count` spans named `name` are there.

    Spans are exported in timed batches, so spans that end close together can land in
    different batches: wait for the last one a test needs, not the first.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        spans = spans_in_trace(service, trace_id)
        if sum(span["name"] == name for span in spans) >= count:
            return spans
        assert time.monotonic() < deadline, f"no {count} {name!r} spans of {trace_id} in {service}"
        time.sleep(0.25)


# --- fixtures --------------------------------------------------------------------------


@pytest.fixture
def mock() -> Iterator[Mock]:
    """The seeded mock with chaos off; chaos is turned off again even when a test fails."""
    stack_mock = Mock()
    stack_mock.chaos_off()
    stack_mock.reset()
    try:
        yield stack_mock
    finally:
        stack_mock.chaos_off()


@pytest.fixture
def agent() -> Iterator[Agent]:
    wait_until_healthy()
    with httpx.Client(base_url=AGENT_URL, timeout=60) as client:
        # A fresh session per test: the e2e agent's checkpoints outlive test runs.
        yield Agent(client, f"e2e-{uuid.uuid4().hex[:12]}")
