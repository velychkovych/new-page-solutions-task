"""Behaviour of the page's own JavaScript, run in Node through page_harness.mjs.

The harness executes the <script> blocks of web/index.html against a DOM stand-in and a
scripted fetch, so these tests cover the shipped code: the SSE parser, the /chat
fallback rule, never re-sending a request whose stream started, the done/in-flight
rules, the Jaeger link and the session rebuild. Node is not a Python dependency, so the
module is skipped (with a reason under `-ra`) where `node` is not on PATH.
"""

import base64
import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
HARNESS = HERE / "page_harness.mjs"
WEB = HERE.parents[2] / "web"
INDEX = WEB / "index.html"
FIXTURES = WEB / "fixtures"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

TRACE = "4bf92f3577b34da6a3ce929d0e0e4736"
STORAGE_KEY = "helpdesk.identity"
SESSION_PENDING: dict[str, Any] = json.loads((FIXTURES / "session-pending.json").read_bytes())
ACTION_ID: str = SESSION_PENDING["pending_action"]["action_id"]
EMPTY_SESSION = {"json": {"messages": []}}

type Report = dict[str, Any]


def frame(name: str, data: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def token(text: str) -> str:
    return frame("token", {"text": text})


def done(status: str = "ok") -> str:
    return frame("done", {"trace_id": TRACE, "status": status})


def run_page(scenario: dict[str, Any]) -> Report:
    assert NODE is not None
    proc = subprocess.run(
        [NODE, str(HARNESS), str(INDEX)],
        input=json.dumps(scenario),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    report: Report = json.loads(proc.stdout)
    assert report["errors"] == []
    return report


def scenario(
    steps: list[dict[str, Any]],
    routes: dict[str, list[dict[str, Any]]],
    *,
    session: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "storage": {STORAGE_KEY: json.dumps({"userId": "alice", "sessionId": "s1"})},
        "routes": {"GET /sessions/alice/s1": [session or EMPTY_SESSION], **routes},
        "steps": [{"op": "idle"}, *steps],
        **extra,
    }


def fetches(report: Report) -> list[str]:
    return [entry["key"] for entry in report["log"] if entry["event"] == "fetch"]


def bodies(report: Report, key: str) -> list[dict[str, Any]]:
    return [e["body"] for e in report["log"] if e["event"] == "fetch" and e["key"] == key]


def chat_ok(reply: str) -> dict[str, Any]:
    return {
        "json": {"reply": reply, "status": "ok", "tool_calls": [], "trace_id": TRACE},
    }


SEND = [{"op": "send", "text": "hello"}, {"op": "idle"}, {"op": "snapshot"}]
APPROVE = [{"op": "click", "id": "approve"}, {"op": "idle"}, {"op": "snapshot"}]


# --- the page's SSE parser --------------------------------------------------------


def oracle(raw: bytes) -> list[list[str]]:
    """Fixture frames are exactly `event: <name>\\ndata: <json>` (test_sse_fixtures)."""
    events = []
    for block in raw.decode().split("\n\n"):
        if block and not block.startswith(":"):
            name, data = block.split("\n")
            events.append([name.removeprefix("event: "), data.removeprefix("data: ")])
    return events


def chunkings(raw: bytes) -> list[list[bytes]]:
    rng = random.Random(7)  # noqa: S311 - reproducible chunk boundaries
    out = [[raw], [raw[i : i + 1] for i in range(len(raw))]]
    for size in (2, 3, 7, 64):
        out.append([raw[i : i + size] for i in range(0, len(raw), size)])
    for _ in range(5):
        cuts = sorted(rng.sample(range(1, len(raw)), k=min(25, len(raw) - 1)))
        out.append([raw[a:b] for a, b in zip([0, *cuts], [*cuts, len(raw)], strict=True)])
    return out


def page_parse(cases: list[list[bytes]]) -> list[list[list[str]]]:
    encoded = [[base64.b64encode(c).decode() for c in chunks] for chunks in cases]
    parsed: list[list[list[str]]] = run_page({"parse": encoded})["parsed"]
    return parsed


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.sse")), ids=lambda p: p.name)
def test_page_parser_reads_every_fixture_under_any_chunking(path: Path) -> None:
    raw = path.read_bytes()
    expected = oracle(raw)
    assert expected
    cases = chunkings(raw)
    assert page_parse(cases) == [expected] * len(cases)


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        pytest.param(
            [b': ping\n\nevent: token\n: inner comment\ndata: {"text":"a"}\n\nevent: lonely\n\n'],
            [["token", '{"text":"a"}']],
            id="comments-and-dataless-frames",
        ),
        pytest.param(
            [b'event: token\r\ndata: {"text":"a"}\r\n\r\nevent: token\rdata: {"text":"b"}\r\r'],
            [["token", '{"text":"a"}'], ["token", '{"text":"b"}']],
            id="crlf-and-cr",
        ),
        pytest.param(
            [b"event: token\r", b'\ndata: {"text":"c"}\r', b"\n\r", b"\n"],
            [["token", '{"text":"c"}']],
            id="crlf-split-across-chunks",
        ),
        pytest.param(
            [b"event: x\ndata:  two\ndata:one\n\n"],
            [["x", " two\none"]],
            id="multi-data-lines",
        ),
        pytest.param(
            [b'data: {"a":1}\n\nevent: token\ndata: {"text":"cut"}\n'],
            [["message", '{"a":1}']],
            id="default-name-and-unterminated-frame",
        ),
        pytest.param(
            [bytes([b]) for b in 'event: token\ndata: {"text":"Привіт 🔋"}\n\n'.encode()],
            [["token", '{"text":"Привіт 🔋"}']],
            id="multibyte-split",
        ),
    ],
)
def test_page_parser_edge_cases(chunks: list[bytes], expected: list[list[str]]) -> None:
    assert page_parse([chunks]) == [expected]


# --- fallback to /chat, and never re-sending ------------------------------------------


@pytest.mark.parametrize(
    "stream",
    [
        pytest.param({"status": 404}, id="404"),
        pytest.param({"status": 405}, id="405"),
        pytest.param({"status": 501}, id="501"),
        pytest.param({"networkError": True}, id="network-error"),
        pytest.param({"status": 200, "json": {"reply": "x"}}, id="not-an-event-stream"),
    ],
)
def test_falls_back_to_chat_when_streaming_is_unavailable(stream: dict[str, Any]) -> None:
    report = run_page(
        scenario(SEND, {"POST /chat/stream": [stream], "POST /chat": [chat_ok("Hi there")]})
    )
    assert fetches(report) == ["GET /sessions/alice/s1", "POST /chat/stream", "POST /chat"]
    assert bodies(report, "POST /chat") == bodies(report, "POST /chat/stream")
    [snap] = report["snapshots"]
    assert "Hi there" in snap["messages"][-1]["text"]
    assert snap["status"] == "Done."
    assert not snap["busy"]


STARTED_OR_FAILED = [
    pytest.param({"status": 500}, id="500"),
    pytest.param({"status": 502}, id="502"),
    pytest.param({"status": 503}, id="503"),
    pytest.param({"chunks": [token("par")]}, id="closed-before-done"),
    pytest.param({"chunks": [token("par")], "reset": True}, id="reset-mid-stream"),
]


@pytest.mark.parametrize("stream", STARTED_OR_FAILED)
def test_a_started_or_failed_message_is_never_resent(stream: dict[str, Any]) -> None:
    report = run_page(
        scenario(SEND, {"POST /chat/stream": [stream], "POST /chat": [chat_ok("again")]})
    )
    # The session view is reloaded instead, because the server may have acted.
    assert fetches(report) == [
        "GET /sessions/alice/s1",
        "POST /chat/stream",
        "GET /sessions/alice/s1",
    ]


@pytest.mark.parametrize("stream", STARTED_OR_FAILED)
def test_a_started_or_failed_decision_is_never_resent(stream: dict[str, Any]) -> None:
    report = run_page(
        scenario(
            APPROVE,
            {"POST /chat/stream": [stream], "POST /chat": [chat_ok("again")]},
            session={"json": SESSION_PENDING},
        )
    )
    assert fetches(report) == [
        "GET /sessions/alice/s1",
        "POST /chat/stream",
        "GET /sessions/alice/s1",
    ]
    [decision] = bodies(report, "POST /chat/stream")
    assert decision == {
        "user_id": "alice",
        "session_id": "s1",
        "decision": {"action_id": ACTION_ID, "approve": True},
    }


# --- done and in-flight rules ---------------------------------------------------------


@pytest.mark.parametrize(
    "chunks",
    [
        pytest.param(
            [
                ": ping\n\n" + 'event: token\n: inner\ndata: {"text":"Hel"}\n\n',
                token("lo") + done() + token(" IGNORED") + frame("error", {"message": "late"}),
            ],
            id="same-chunk",
        ),
        pytest.param(
            [token("Hel"), token("lo") + done(), token(" IGNORED"), done("error")],
            id="later-chunk",
        ),
    ],
)
def test_comments_are_ignored_and_nothing_after_done_counts(chunks: list[str]) -> None:
    report = run_page(scenario(SEND, {"POST /chat/stream": [{"chunks": chunks}]}))
    [snap] = report["snapshots"]
    assistant = snap["messages"][-1]
    assert assistant["class"] == "msg msg-assistant"
    assert assistant["text"].startswith("assistantHello")
    assert "IGNORED" not in assistant["text"]
    assert snap["status"] == "Done."
    assert snap["statusKind"] == "info"


def test_turn_ends_at_done_even_if_the_server_keeps_the_stream_open() -> None:
    lingering = {"chunks": [token("Hi"), done()], "pings": 20, "pingMs": 200}
    report = run_page(
        scenario(
            [
                {"op": "send", "text": "hello"},
                {"op": "idle", "timeoutMs": 1000},
                {"op": "snapshot"},
            ],
            {"POST /chat/stream": [lingering]},
        )
    )
    # The 20 pings would hold the connection for 4 s; the turn must not wait for them.
    assert report["idle"] == [True, True]
    [snap] = report["snapshots"]
    assert snap["status"] == "Done."
    assert not any(snap["disabled"][k] for k in ("send", "new-session", "user-id", "session-id"))
    assert {"event": "cancel", "key": "POST /chat/stream"} in report["log"]


def test_controls_are_disabled_while_a_request_is_in_flight() -> None:
    report = run_page(
        scenario(
            [{"op": "send", "text": "hello"}, {"op": "sleep", "ms": 50}, {"op": "snapshot"}],
            {"POST /chat/stream": [{"chunks": [token("Hi")], "hold": True}]},
            session={"json": SESSION_PENDING},
        )
    )
    [snap] = report["snapshots"]
    assert snap["busy"]
    assert all(snap["disabled"].values()), snap["disabled"]


def test_a_double_click_sends_one_decision() -> None:
    reply = {"chunks": [token("Done."), done()]}
    report = run_page(
        scenario(
            [{"op": "click", "id": "approve"}, *APPROVE],
            {"POST /chat/stream": [reply]},
            session={"json": SESSION_PENDING},
        )
    )
    assert fetches(report).count("POST /chat/stream") == 1


# --- Jaeger link ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("meta", "expected"),
    [
        pytest.param("", None, id="unset"),
        pytest.param("http://jaeger.test:16686", "http://jaeger.test:16686/trace/", id="plain"),
        pytest.param("http://jaeger.test:16686/", "http://jaeger.test:16686/trace/", id="slash"),
        pytest.param(
            "https://ops.test/jaeger//", "https://ops.test/jaeger/trace/", id="path-prefix"
        ),
        pytest.param(
            "http://jaeger.test:16686/?x=1#h", "http://jaeger.test:16686/trace/", id="query-hash"
        ),
        pytest.param("javascript:alert(1)", None, id="javascript"),
        pytest.param("ftp://jaeger.test/", None, id="ftp"),
    ],
)
def test_trace_link_uses_the_configured_jaeger_base(meta: str, expected: str | None) -> None:
    report = run_page(
        scenario(SEND, {"POST /chat/stream": [{"chunks": [token("Hi"), done()]}]}, jaegerMeta=meta)
    )
    [snap] = report["snapshots"]
    assert TRACE in snap["trace"]
    links = set(snap["traceLinks"])
    assert links == (set() if expected is None else {expected + TRACE})


def test_trace_link_ignores_a_jaeger_query_parameter() -> None:
    # Otherwise any crafted link to the page could point "open in Jaeger" anywhere.
    report = run_page(
        scenario(
            SEND,
            {"POST /chat/stream": [{"chunks": [token("Hi"), done()]}]},
            search="?jaeger=http%3A%2F%2Fevil.test%2F",
        )
    )
    [snap] = report["snapshots"]
    assert snap["traceLinks"] == []


# --- session rebuild --------------------------------------------------------------------


def test_session_view_rebuilds_messages_and_buttons() -> None:
    report = run_page(scenario([{"op": "snapshot"}], {}, session={"json": SESSION_PENDING}))
    [snap] = report["snapshots"]
    assert [m["class"] for m in snap["messages"]] == [
        "msg msg-user",
        "msg msg-tool",
        "msg msg-tool",
        "msg msg-assistant",
    ]
    # Untrusted text is shown as text.
    assert "<img src=x onerror=alert(1)>" in snap["messages"][1]["text"]
    assert not snap["pendingHidden"]
    assert not snap["disabled"]["approve"]
    assert not snap["disabled"]["decline"]


@pytest.mark.parametrize(
    "failure",
    [pytest.param({"status": 500}, id="500"), pytest.param({"networkError": True}, id="network")],
)
def test_failed_session_switch_clears_the_previous_session(failure: dict[str, Any]) -> None:
    report = run_page(
        scenario(
            [
                {"op": "set", "id": "session-id", "value": "s2"},
                {"op": "idle"},
                {"op": "snapshot"},
                *APPROVE,
            ],
            {"GET /sessions/alice/s2": [failure]},
            session={"json": SESSION_PENDING},
        )
    )
    snap = report["snapshots"][0]
    assert snap["statusKind"] == "error"
    assert snap["messages"] == []
    assert snap["pendingHidden"]
    assert snap["disabled"]["approve"]
    assert snap["disabled"]["decline"]
    assert "POST /chat/stream" not in fetches(report)


def test_successful_session_switch_replaces_the_previous_session() -> None:
    report = run_page(
        scenario(
            [{"op": "set", "id": "session-id", "value": "s2"}, {"op": "idle"}, {"op": "snapshot"}],
            {"GET /sessions/alice/s2": [EMPTY_SESSION]},
            session={"json": SESSION_PENDING},
        )
    )
    [snap] = report["snapshots"]
    assert snap["messages"] == []
    assert snap["pendingHidden"]
    assert snap["disabled"]["approve"]


def test_a_decision_is_never_sent_under_a_different_identity() -> None:
    # The session id was edited but not yet committed (no change event): the pending
    # action belongs to s1, so it must not be sent for s2. The page reloads instead.
    report = run_page(
        scenario(
            [{"op": "type", "id": "session-id", "value": "s2"}, *APPROVE],
            {"GET /sessions/alice/s2": [EMPTY_SESSION]},
            session={"json": SESSION_PENDING},
        )
    )
    assert "POST /chat/stream" not in fetches(report)
    assert fetches(report)[-1] == "GET /sessions/alice/s2"
    [snap] = report["snapshots"]
    assert snap["pendingHidden"]
