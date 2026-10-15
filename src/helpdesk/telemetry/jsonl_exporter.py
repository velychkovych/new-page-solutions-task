"""Writes finished spans to `<traces_dir>/<service>.jsonl`, one span per line (CONTRACTS §6)."""

import logging
import threading
from collections.abc import Sequence
from pathlib import Path

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from helpdesk.telemetry.records import span_to_record

logger = logging.getLogger(__name__)


class JsonlSpanExporter(SpanExporter):
    """Appends JSON lines to one file per service.

    The SDK calls `export` from the batch processor's worker thread, and `force_flush` or
    `shutdown` can call it from another thread at the same time, so writes are serialised
    with a lock. Blocking file I/O is fine here: it never runs on the event loop.
    """

    def __init__(self, traces_dir: Path, service_name: str) -> None:
        self.path = traces_dir / f"{service_name}.jsonl"
        self.service_name = service_name
        self._lock = threading.Lock()
        self._shut_down = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        records = [
            record
            for span in spans
            if (record := span_to_record(span, self.service_name)) is not None
        ]
        if not records:
            return SpanExportResult.SUCCESS
        # One write per batch, so a reader never sees half of a batch's lines interleaved.
        payload = "".join(record.model_dump_json() + "\n" for record in records)
        with self._lock:
            if self._shut_down:
                return SpanExportResult.FAILURE
            try:
                # Created on every batch, so deleting traces/ while a service runs is harmless.
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as file:
                    file.write(payload)
                    file.flush()
            except OSError as exc:
                # The path and errno only; span content could be sensitive.
                logger.warning("could not write spans to %s: %s", self.path, exc.strerror)
                return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        with self._lock:
            self._shut_down = True

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        # Every export already closes the file, so there is nothing buffered here.
        return True
