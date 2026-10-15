import io
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from helpdesk.shared import trace_attributes as ta
from helpdesk.shared.settings import Settings
from helpdesk.telemetry.records import SpanEventRecord, SpanRecord
from helpdesk.telemetry.show_trace import ExitCode, main, normalise_trace_id

ROOT = Path(__file__).resolve().parents[3]
TRACE = "4bf92f3577b34da6a3ce929d0e0e4736"
OTHER_TRACE = "0af7651916cd43dd8448eb211c80319c"
T0 = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)


def _span(
    span_id: str,
    parent: str | None,
    name: str,
    service: str,
    *,
    offset_ms: int = 0,
    trace_id: str = TRACE,
    attributes: dict[str, JsonValue] | None = None,
    events: list[SpanEventRecord] | None = None,
) -> SpanRecord:
    start = T0 + timedelta(milliseconds=offset_ms)
    return SpanRecord(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent,
        name=name,
        service=service,
        start_time=start,
        end_time=start + timedelta(milliseconds=10),
        duration_ms=10.0,
        status="ok",
        attributes=attributes or {},
        events=events or [],
    )


def _write(path: Path, *spans: SpanRecord, extra: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.writelines(span.model_dump_json() + "\n" for span in spans)
        file.write(extra)


def _run(settings: Settings, *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), settings=settings, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (TRACE, TRACE),
        (TRACE.upper(), TRACE),
        (f"0x{TRACE}", TRACE),
        (f"  {TRACE}\n", TRACE),
        ("0" * 32, None),
        (TRACE[:-1], None),
        (TRACE + "0", None),
        ("z" * 32, None),
    ],
)
def test_normalise_trace_id(raw: str, expected: str | None) -> None:
    assert normalise_trace_id(raw) == expected


def test_invalid_trace_id_exits_with_its_own_code(make_settings: Any) -> None:
    code, out, err = _run(make_settings(), "not-a-trace")
    assert code == ExitCode.INVALID_TRACE_ID == 2
    assert out == ""
    assert "invalid trace id 'not-a-trace'" in err


def test_unknown_trace_id_exits_not_found(make_settings: Any) -> None:
    settings: Settings = make_settings()
    _write(
        settings.traces_dir / "agent.jsonl",
        _span("a" * 16, None, "x", "agent", trace_id=OTHER_TRACE),
    )
    code, out, err = _run(settings, TRACE)
    assert code == ExitCode.NOT_FOUND == 1
    assert out == ""
    assert f"trace {TRACE} not found in 1 *.jsonl file(s)" in err


def test_missing_traces_dir_is_reported(make_settings: Any) -> None:
    code, _, err = _run(make_settings(), TRACE)
    assert code == ExitCode.NOT_FOUND
    assert "directory does not exist" in err


def test_merges_services_and_shows_orphans(make_settings: Any) -> None:
    settings: Settings = make_settings()
    root, tool, orphan = "1" * 16, "2" * 16, "3" * 16
    _write(
        settings.traces_dir / "agent.jsonl",
        _span(root, None, ta.SPAN_REQUEST, "agent"),
        _span("9" * 16, None, "other", "agent", trace_id=OTHER_TRACE),
        # A line of this trace that a service was still writing.
        extra=f'{{"trace_id": "{TRACE}", "span_\n',
    )
    _write(
        settings.traces_dir / "mcp.jsonl",
        _span(tool, root, ta.SPAN_TOOL, "mcp", offset_ms=5),
        # Its parent was never exported (e.g. the mock runs without telemetry).
        _span(orphan, "f" * 16, "POST /tickets", "mock", offset_ms=7),
    )

    code, out, err = _run(settings, TRACE)

    assert code == ExitCode.OK
    lines = out.splitlines()
    assert lines[0] == f"trace {TRACE}: 3 spans, services: agent, mcp, mock"
    assert lines[1].startswith(f"{ta.SPAN_REQUEST} [agent] 10.0 ms ok")
    assert lines[2].startswith(f"  {ta.SPAN_TOOL} [mcp] 10.0 ms ok")
    assert lines[3].startswith("POST /tickets [mock] 10.0 ms ok")
    assert f"(orphan: parent {'f' * 16} is in no trace file)" in lines[3]
    assert "other" not in out
    assert "skipped 1 unreadable line" in err


def test_line_cut_inside_a_multibyte_character_is_skipped(make_settings: Any) -> None:
    settings: Settings = make_settings()
    path = settings.traces_dir / "agent.jsonl"
    root, later = "1" * 16, "2" * 16
    _write(path, _span(root, None, ta.SPAN_REQUEST, "agent", attributes={"text": "café"}))
    # A service died mid-write inside "é" (two UTF-8 bytes); after its restart the next
    # record is appended right behind the broken bytes, so they end up mid-file.
    cut = _span("3" * 16, root, "lost", "agent").model_dump_json().replace("lost", "é")
    with path.open("ab") as file:
        file.write(cut.encode("utf-8")[: cut.encode("utf-8").index("é".encode()) + 1])
    _write(path, _span(later, root, "after-restart", "agent", offset_ms=5))
    _write(path, _span("4" * 16, None, "another", "agent", offset_ms=9))

    code, out, err = _run(settings, TRACE)

    assert code == ExitCode.OK
    assert out.splitlines()[0] == f"trace {TRACE}: 2 spans, services: agent"
    assert "another" in out
    assert "skipped 1 unreadable line" in err


def test_renders_tokens_tool_details_and_retry_events(make_settings: Any) -> None:
    settings: Settings = make_settings()
    root, llm, tool = "1" * 16, "2" * 16, "3" * 16
    _write(
        settings.traces_dir / "agent.jsonl",
        _span(root, None, ta.SPAN_REQUEST, "agent", attributes={ta.REQUEST_STATUS: "ok"}),
        _span(
            llm,
            root,
            "ChatAnthropic",
            "agent",
            offset_ms=1,
            attributes={
                "llm.model_name": "claude-sonnet-5-5",
                "llm.token_count.prompt": 1200,
                "llm.token_count.completion": 80,
                "llm.token_count.prompt_details.cache_read": 1000,
            },
        ),
    )
    _write(
        settings.traces_dir / "mcp.jsonl",
        _span(
            tool,
            root,
            ta.SPAN_TOOL,
            "mcp",
            offset_ms=2,
            attributes={
                ta.TOOL_NAME: "search_tickets",
                ta.TOOL_ATTEMPTS: 3,
                ta.TOOL_ERROR_CODE: "UPSTREAM_UNAVAILABLE",
            },
            events=[
                SpanEventRecord(
                    name=ta.EVENT_RETRY,
                    timestamp=T0,
                    attributes={
                        ta.RETRY_ATTEMPT: 1,
                        ta.RETRY_HTTP_STATUS: 503,
                        ta.RETRY_WAIT_MS: 250,
                    },
                ),
                SpanEventRecord(
                    name="exception",
                    timestamp=T0,
                    attributes={"exception.type": "ReadTimeout", "exception.message": "timed out"},
                ),
            ],
        ),
    )

    code, out, _ = _run(settings, TRACE)

    assert code == ExitCode.OK
    assert "request_status=ok" in out
    assert (
        "  ChatAnthropic [agent] 10.0 ms ok  model=claude-sonnet-5-5"
        "  tokens in=1200 out=80 cache_read=1000 cache_write=0"
    ) in out
    assert (
        "  mcp.tool [mcp] 10.0 ms ok  tool=search_tickets  attempts=3"
        "  error_code=UPSTREAM_UNAVAILABLE"
    ) in out
    assert "    · retry.attempt attempt=1 http.status_code=503 wait_ms=250" in out
    assert "    · exception ReadTimeout: timed out" in out


def test_script_wrapper_runs_from_the_command_line(make_settings: Any, tmp_path: Path) -> None:
    settings: Settings = make_settings()
    _write(settings.traces_dir / "agent.jsonl", _span("1" * 16, None, ta.SPAN_REQUEST, "agent"))
    env = {**os.environ, "TRACES_DIR": str(settings.traces_dir)}

    found = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "show_trace.py"), TRACE],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,  # no .env here, so only the env above configures Settings
        check=False,
    )
    missing = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "show_trace.py"), OTHER_TRACE],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=False,
    )

    assert found.returncode == 0, found.stderr
    assert f"{ta.SPAN_REQUEST} [agent]" in found.stdout
    assert missing.returncode == 1
    assert "not found" in missing.stderr
