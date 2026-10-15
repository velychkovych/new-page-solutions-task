"""Track E: OpenTelemetry setup, JSONL exporter and redaction."""

from helpdesk.telemetry.setup import (
    Telemetry,
    TelemetryConfigError,
    instrument_app,
    setup_telemetry,
    shutdown_telemetry,
)

__all__ = [
    "Telemetry",
    "TelemetryConfigError",
    "instrument_app",
    "setup_telemetry",
    "shutdown_telemetry",
]
