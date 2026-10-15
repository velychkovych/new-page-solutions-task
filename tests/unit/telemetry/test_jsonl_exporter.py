import json
import re
import threading
from datetime import datetime
from pathlib import Path

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

from helpdesk.shared import trace_attributes as ta
from helpdesk.telemetry.jsonl_exporter import JsonlSpanExporter
from helpdesk.telemetry.records import SpanRecord

CONTRACT_FIELDS = {
    "trace_id",
    "span_id",
    "parent_span_id",
    "name",
    "service",
    "start_time",
    "end_time",
    "duration_ms",
    "status",
    "attributes",
    "events",
}
HEX32 = re.compile(r"^[0-9a-f]{32}$")
HEX16 = re.compile(r"^[0-9a-f]{16}$")


def _finished_spans() -> list[ReadableSpan]:
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span(ta.SPAN_REQUEST) as root:
        root.set_attribute(ta.USER_ID, "alice")
        with tracer.start_as_current_span(ta.SPAN_TOOL) as tool:
            tool.set_attribute(ta.TOOL_ATTEMPTS, 2)
            tool.add_event(ta.EVENT_RETRY, {ta.RETRY_ATTEMPT: 1, ta.RETRY_WAIT_MS: 250})
            tool.set_status(Status(StatusCode.ERROR, "boom"))
    return list(memory.get_finished_spans())


def test_writes_one_contract_line_per_span(tmp_path: Path) -> None:
    traces_dir = tmp_path / "does" / "not" / "exist"
    exporter = JsonlSpanExporter(traces_dir, "agent")

    assert exporter.export(_finished_spans()) is SpanExportResult.SUCCESS

    lines = (traces_dir / "agent.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    tool, root = (json.loads(line) for line in lines)  # children end first
    for record in (tool, root):
        assert set(record) == CONTRACT_FIELDS
        assert HEX32.fullmatch(record["trace_id"])
        assert HEX16.fullmatch(record["span_id"])
        assert record["service"] == "agent"
        start = datetime.fromisoformat(record["start_time"])
        end = datetime.fromisoformat(record["end_time"])
        assert start.tzinfo is not None
        assert start <= end
        assert record["duration_ms"] >= 0
    assert root["parent_span_id"] is None
    assert tool["parent_span_id"] == root["span_id"]
    assert tool["trace_id"] == root["trace_id"]
    assert root["status"] == "ok"
    assert tool["status"] == "error"
    assert root["attributes"] == {ta.USER_ID: "alice"}
    assert tool["attributes"] == {ta.TOOL_ATTEMPTS: 2}
    (event,) = tool["events"]
    assert event["name"] == ta.EVENT_RETRY
    assert event["attributes"] == {ta.RETRY_ATTEMPT: 1, ta.RETRY_WAIT_MS: 250}
    assert datetime.fromisoformat(event["timestamp"]).tzinfo is not None
    # Every line round-trips through the reader's model.
    for line in lines:
        SpanRecord.model_validate_json(line)


def test_appends_across_batches(tmp_path: Path) -> None:
    exporter = JsonlSpanExporter(tmp_path, "mcp")
    exporter.export(_finished_spans())
    exporter.export(_finished_spans())
    assert len((tmp_path / "mcp.jsonl").read_text(encoding="utf-8").splitlines()) == 4


def test_concurrent_exports_never_interleave_lines(tmp_path: Path) -> None:
    exporter = JsonlSpanExporter(tmp_path, "agent")
    batch = _finished_spans() * 50
    barrier = threading.Barrier(8)

    def export() -> None:
        barrier.wait()
        for _ in range(5):
            exporter.export(batch)

    threads = [threading.Thread(target=export) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lines = (tmp_path / "agent.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 8 * 5 * len(batch)
    for line in lines:
        SpanRecord.model_validate_json(line)


def test_export_after_shutdown_fails_without_writing(tmp_path: Path) -> None:
    exporter = JsonlSpanExporter(tmp_path, "agent")
    exporter.shutdown()
    assert exporter.export(_finished_spans()) is SpanExportResult.FAILURE
    assert not (tmp_path / "agent.jsonl").exists()


def test_unwritable_path_reports_failure(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    exporter = JsonlSpanExporter(blocker, "agent")
    assert exporter.export(_finished_spans()) is SpanExportResult.FAILURE
