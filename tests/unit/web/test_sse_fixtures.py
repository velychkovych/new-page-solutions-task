"""The web fixtures are valid §4 streams, and a correct incremental parser reads them.

The parser below mirrors the one in web/index.html (same algorithm, same edge cases),
so it is the oracle for the fixtures. These tests need no node or browser; the page's
own parser is run against the same fixtures in test_page_behaviour.py.
"""

import codecs
import json
import random
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest
from pydantic import BaseModel

from helpdesk.shared.api import (
    ApprovalRequiredData,
    ChatResponse,
    ChatStatus,
    DoneData,
    ErrorData,
    SessionView,
    SseEvent,
    StatusData,
    TokenData,
    ToolEndData,
    ToolStartData,
)
from helpdesk.shared.errors import parse_tool_error
from helpdesk.shared.tools import WRITE_TOOLS, ToolName

FIXTURES = Path(__file__).resolve().parents[3] / "web" / "fixtures"
SSE_FIXTURES = sorted(FIXTURES.glob("*.sse"))
SESSION_FIXTURES = sorted(FIXTURES.glob("session-*.json"))
CHAT_FIXTURES = sorted(p for p in FIXTURES.glob("*.json") if p not in SESSION_FIXTURES)

DATA_MODELS: dict[SseEvent, type[BaseModel]] = {
    SseEvent.TOKEN: TokenData,
    SseEvent.TOOL_START: ToolStartData,
    SseEvent.TOOL_END: ToolEndData,
    SseEvent.STATUS: StatusData,
    SseEvent.APPROVAL_REQUIRED: ApprovalRequiredData,
    SseEvent.ERROR: ErrorData,
    SseEvent.DONE: DoneData,
}

type RawEvent = tuple[str, str]
type Event = tuple[SseEvent, BaseModel]


# --- reference parser ------------------------------------------------------------


class SseParser:
    """Incremental SSE parser; feed decoded text in chunks of any size."""

    def __init__(self, on_event: Callable[[str, str], None]) -> None:
        self._on_event = on_event
        self._buffer = ""
        self._event_name = ""
        self._data_lines: list[str] = []

    def push(self, text: str) -> None:
        buffer = self._buffer + text
        start = 0
        i = 0
        while i < len(buffer):
            ch = buffer[i]
            if ch in "\r\n":
                # A trailing "\r" may be the first half of "\r\n": wait for more input.
                if ch == "\r" and i + 1 == len(buffer):
                    break
                self._line(buffer[start:i])
                if ch == "\r" and buffer[i + 1] == "\n":
                    i += 1
                start = i + 1
            i += 1
        self._buffer = buffer[start:]

    def end(self) -> None:
        if self._buffer.endswith("\r"):
            self._line(self._buffer[:-1])
        # An unterminated frame is discarded, as the SSE spec says.
        self._buffer = ""
        self._event_name = ""
        self._data_lines = []

    def _line(self, line: str) -> None:
        if line == "":
            self._dispatch()
            return
        if line.startswith(":"):
            return
        field, sep, value = line.partition(":")
        if sep and value.startswith(" "):
            value = value[1:]
        if field == "event":
            self._event_name = value
        elif field == "data":
            self._data_lines.append(value)

    def _dispatch(self) -> None:
        name = self._event_name or "message"
        self._event_name = ""
        if not self._data_lines:
            return
        data = "\n".join(self._data_lines)
        self._data_lines = []
        self._on_event(name, data)


def parse_chunks(chunks: Iterable[bytes]) -> list[RawEvent]:
    """Decode bytes incrementally, as TextDecoder(stream=true) does in the page."""
    events: list[RawEvent] = []
    parser = SseParser(lambda name, data: events.append((name, data)))
    decoder = codecs.getincrementaldecoder("utf-8")()
    for chunk in chunks:
        parser.push(decoder.decode(chunk))
    parser.push(decoder.decode(b"", final=True))
    parser.end()
    return events


def parse_bytes(raw: bytes) -> list[RawEvent]:
    return parse_chunks([raw])


def validate(raw_events: list[RawEvent]) -> list[Event]:
    typed: list[Event] = []
    for name, data in raw_events:
        event = SseEvent(name)
        typed.append((event, DATA_MODELS[event].model_validate_json(data)))
    return typed


def load(path: Path) -> list[Event]:
    return validate(parse_bytes(path.read_bytes()))


def fixture(name: str) -> list[Event]:
    return load(FIXTURES / f"{name}.sse")


# --- §4 ordering rules -------------------------------------------------------------


def _tool_violations(events: list[Event]) -> list[str]:
    problems: list[str] = []
    started: set[str] = set()
    ended: set[str] = set()
    for _, data in events:
        if isinstance(data, ToolStartData):
            if data.call_id in started:
                problems.append(f"tool_start {data.call_id} repeated")
            started.add(data.call_id)
        elif isinstance(data, ToolEndData):
            if data.call_id not in started or data.call_id in ended:
                problems.append(f"tool_end {data.call_id} has no open tool_start")
            ended.add(data.call_id)
    return problems


def _approval_violations(events: list[Event]) -> list[str]:
    problems: list[str] = []
    names = [name for name, _ in events]
    approvals = [i for i, name in enumerate(names) if name is SseEvent.APPROVAL_REQUIRED]
    tool_positions = [
        i for i, name in enumerate(names) if name in {SseEvent.TOOL_START, SseEvent.TOOL_END}
    ]
    if len(approvals) > 1:
        problems.append("approval_required more than once")
    if approvals and tool_positions and approvals[0] < tool_positions[-1]:
        problems.append("approval_required before the last tool event")
    done = events[-1][1] if events else None
    if not isinstance(done, DoneData):
        return problems
    if done.status is ChatStatus.AWAITING_APPROVAL:
        if len(approvals) != 1 or approvals[0] != len(events) - 2:
            problems.append(
                "awaiting_approval needs exactly one approval_required just before done"
            )
    elif approvals:
        # pending_action exists iff status is awaiting_approval (§3.1).
        problems.append("approval_required in a stream that does not await approval")
    return problems


def ordering_violations(events: list[Event]) -> list[str]:
    names = [name for name, _ in events]
    problems: list[str] = []
    if names.count(SseEvent.DONE) != 1 or names[-1:] != [SseEvent.DONE]:
        problems.append("done must appear exactly once, last")
    if names.count(SseEvent.ERROR) > 1:
        problems.append("error more than once")
    return problems + _tool_violations(events) + _approval_violations(events)


# --- tests: the parser -----------------------------------------------------------


def test_parser_ignores_comments_and_dataless_frames() -> None:
    raw = b': ping\n\nevent: token\n: inner comment\ndata: {"text":"a"}\n\nevent: lonely\n\n'
    assert parse_bytes(raw) == [("token", '{"text":"a"}')]


def test_parser_handles_crlf_and_cr_line_endings() -> None:
    raw = b'event: token\r\ndata: {"text":"a"}\r\n\r\nevent: token\rdata: {"text":"b"}\r\r'
    assert parse_bytes(raw) == [("token", '{"text":"a"}'), ("token", '{"text":"b"}')]
    # "\r" then "\n" in separate chunks is one line break, not two.
    assert parse_chunks([b"event: token\r", b'\ndata: {"text":"c"}\r', b"\n\r", b"\n"]) == [
        ("token", '{"text":"c"}')
    ]


def test_parser_joins_multiple_data_lines_and_strips_one_space() -> None:
    assert parse_bytes(b"event: x\ndata:  two\ndata:one\n\n") == [("x", " two\none")]


def test_parser_defaults_event_name_and_discards_unterminated_frame() -> None:
    assert parse_bytes(b'data: {"a":1}\n\nevent: token\ndata: {"text":"cut"}\n') == [
        ("message", '{"a":1}')
    ]


def test_parser_decodes_multibyte_characters_split_across_chunks() -> None:
    raw = 'event: token\ndata: {"text":"Привіт 🔋"}\n\n'.encode()
    one_byte_chunks = [raw[i : i + 1] for i in range(len(raw))]
    assert parse_chunks(one_byte_chunks) == [("token", '{"text":"Привіт 🔋"}')]


# --- tests: the fixtures ---------------------------------------------------------


REQUIRED = {"propose", "approve", "decline", "reask", "error", "ping-split"}


def test_required_fixtures_exist() -> None:
    assert {p.stem for p in SSE_FIXTURES} >= REQUIRED
    assert CHAT_FIXTURES, "a /chat fallback fixture is needed"
    assert SESSION_FIXTURES, "a session view fixture is needed"
    # Fixture mode answers /chat/stream with 404 for a name without .sse, which is how
    # the page's real fallback path is exercised.
    assert all(not (FIXTURES / f"{p.stem}.sse").exists() for p in CHAT_FIXTURES)


@pytest.mark.parametrize("path", SSE_FIXTURES, ids=lambda p: p.name)
def test_fixture_events_validate_against_shared_models(path: Path) -> None:
    raw = path.read_bytes()
    raw.decode("utf-8")  # strict UTF-8
    events = load(path)
    assert events
    # Exactly the contract wire format: no frame was silently dropped by the parser.
    frames = [f for f in raw.decode().split("\n\n") if f and not f.startswith(":")]
    assert len(frames) == len(events)
    for frame in frames:
        lines = frame.split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")
        data = lines[1].removeprefix("data: ")
        assert json.dumps(json.loads(data), separators=(",", ":"), ensure_ascii=False) == data


@pytest.mark.parametrize("path", SSE_FIXTURES, ids=lambda p: p.name)
def test_fixture_terminates_its_last_frame(path: Path) -> None:
    # The end-of-file-fixer hook trims a trailing "\n\n" to "\n", which would make the
    # final `done` frame unterminated (and dropped). Fixtures end with a comment line.
    raw = path.read_bytes()
    last_frame_end = raw.rindex(b"\n\n") + 2
    assert all(line.startswith(b":") for line in raw[last_frame_end:].splitlines())


@pytest.mark.parametrize("path", SSE_FIXTURES, ids=lambda p: p.name)
def test_fixture_follows_ordering_rules(path: Path) -> None:
    assert ordering_violations(load(path)) == []


@pytest.mark.parametrize("path", SSE_FIXTURES, ids=lambda p: p.name)
def test_fixture_parses_identically_for_any_chunking(path: Path) -> None:
    raw = path.read_bytes()
    expected = parse_bytes(raw)
    for size in range(1, 40):
        chunks = [raw[i : i + size] for i in range(0, len(raw), size)]
        assert parse_chunks(chunks) == expected, f"chunk size {size}"
    rng = random.Random(4)  # noqa: S311 - reproducible chunk boundaries
    for _ in range(50):
        cuts = sorted(rng.sample(range(1, len(raw)), k=min(25, len(raw) - 1)))
        chunks = [raw[a:b] for a, b in zip([0, *cuts], [*cuts, len(raw)], strict=True)]
        assert parse_chunks(chunks) == expected
    # The page's own replay sizes (FIXTURE_CHUNK_SIZES in index.html).
    sizes = [1, 7, 2, 13, 3, 31, 5, 64]
    chunks, offset, index = [], 0, 0
    while offset < len(raw):
        size = sizes[index % len(sizes)]
        chunks.append(raw[offset : offset + size])
        offset, index = offset + size, index + 1
    assert parse_chunks(chunks) == expected


def done_of(events: list[Event]) -> DoneData:
    done = events[-1][1]
    assert isinstance(done, DoneData)
    return done


def of_type[M: BaseModel](events: list[Event], model: type[M]) -> list[M]:
    return [data for _, data in events if isinstance(data, model)]


def approval_of(events: list[Event]) -> ApprovalRequiredData:
    [approval] = [d for n, d in events if n is SseEvent.APPROVAL_REQUIRED]
    assert isinstance(approval, ApprovalRequiredData)
    return approval


def test_propose_fixture_reads_then_awaits_approval() -> None:
    events = fixture("propose")
    tools = [ToolName(d.name) for d in of_type(events, ToolStartData)]
    assert tools == [ToolName.SEARCH_TICKETS, ToolName.GET_TICKET]
    assert done_of(events).status is ChatStatus.AWAITING_APPROVAL
    approval = approval_of(events)
    assert approval.tool in WRITE_TOOLS
    assert "idempotency_key" not in approval.args
    assert "author_id" not in approval.args
    # A prompt-injection string reaches the page, which must render it as text.
    assert any("<img" in d.result_summary for d in of_type(events, ToolEndData))


def test_approve_fixture_executes_the_proposed_write() -> None:
    proposed = approval_of(fixture("propose"))
    events = fixture("approve")
    [write] = of_type(events, ToolStartData)
    assert write.name == proposed.tool
    assert write.args == proposed.args
    assert all(d.error is None for d in of_type(events, ToolEndData))
    assert done_of(events).status is ChatStatus.OK


def test_decline_fixture_writes_nothing() -> None:
    events = fixture("decline")
    assert of_type(events, ToolStartData) == []
    assert done_of(events).status is ChatStatus.OK


def test_reask_fixture_repeats_the_same_pending_action() -> None:
    events = fixture("reask")
    assert of_type(events, ToolStartData) == []
    assert approval_of(events) == approval_of(fixture("propose"))
    assert done_of(events).status is ChatStatus.AWAITING_APPROVAL


def test_error_fixture_has_one_error_and_contract_tool_error_text() -> None:
    events = fixture("error")
    assert len(of_type(events, ErrorData)) == 1
    assert done_of(events).status is ChatStatus.ERROR
    errors = [d.error for d in of_type(events, ToolEndData) if d.error]
    assert errors
    assert all(parse_tool_error(e) is not None for e in errors)
    assert of_type(events, StatusData)


def test_ping_split_fixture_has_pings_and_multibyte_text() -> None:
    raw = (FIXTURES / "ping-split.sse").read_bytes()
    assert raw.startswith(b": ping\n\n")
    assert raw.count(b": ping\n\n") >= 3
    assert any(len(ch.encode()) == 4 for ch in raw.decode())  # a 4-byte UTF-8 character
    tokens = "".join(d.text for d in of_type(fixture("ping-split"), TokenData))
    # SSE-looking text inside a JSON string is data, not a frame.
    assert "event: done\ndata: {}" in tokens


@pytest.mark.parametrize("path", CHAT_FIXTURES, ids=lambda p: p.name)
def test_chat_response_fixture_validates(path: Path) -> None:
    response = ChatResponse.model_validate_json(path.read_bytes())
    assert (response.pending_action is not None) == (
        response.status is ChatStatus.AWAITING_APPROVAL
    )


@pytest.mark.parametrize("path", SESSION_FIXTURES, ids=lambda p: p.name)
def test_session_fixture_validates(path: Path) -> None:
    view = SessionView.model_validate_json(path.read_bytes())
    assert view.messages


# --- the ordering checker catches what it should --------------------------------


def _start(call_id: str) -> RawEvent:
    return ("tool_start", json.dumps({"call_id": call_id, "name": "get_ticket", "args": {}}))


def _end(call_id: str) -> RawEvent:
    return ("tool_end", json.dumps({"call_id": call_id, "result_summary": "", "latency_ms": 1}))


def _done(status: str) -> RawEvent:
    return ("done", json.dumps({"trace_id": "0" * 32, "status": status}))


_APPROVAL: RawEvent = (
    "approval_required",
    json.dumps({"action_id": "a1", "tool": "add_comment", "args": {}, "summary": "s"}),
)
_ERROR: RawEvent = ("error", json.dumps({"message": "boom"}))
_TOKEN: RawEvent = ("token", json.dumps({"text": "t"}))


@pytest.mark.parametrize(
    ("raw", "problem"),
    [
        ([_end("c1"), _done("ok")], "no open tool_start"),
        ([_start("c1"), _end("c1"), _end("c1"), _done("ok")], "no open tool_start"),
        ([_start("c1"), _end("c1")], "done must appear"),
        ([_done("ok"), _TOKEN], "done must appear"),
        ([_done("ok"), _done("ok")], "done must appear"),
        ([_APPROVAL, _APPROVAL, _done("awaiting_approval")], "more than once"),
        ([_APPROVAL, _start("c1"), _end("c1"), _done("awaiting_approval")], "before the last"),
        ([_done("awaiting_approval")], "exactly one approval_required"),
        ([_APPROVAL, _TOKEN, _done("awaiting_approval")], "just before done"),
        ([_APPROVAL, _done("ok")], "does not await approval"),
        ([_ERROR, _ERROR, _done("error")], "error more than once"),
    ],
)
def test_ordering_checker_rejects_bad_streams(raw: list[RawEvent], problem: str) -> None:
    assert any(problem in p for p in ordering_violations(validate(raw)))


def test_ordering_checker_accepts_a_good_stream() -> None:
    raw = [_start("c1"), _TOKEN, _end("c1"), _TOKEN, _APPROVAL, _done("awaiting_approval")]
    assert ordering_violations(validate(raw)) == []
