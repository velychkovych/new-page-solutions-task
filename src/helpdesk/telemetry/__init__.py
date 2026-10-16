"""Track E: OpenTelemetry setup, JSONL exporter and redaction.

A service wires it in two places:
- app factory: `FastAPI(telemetry=fastapi_telemetry_config(), ...)`, then `instrument_app(app)`;
- lifespan: `async with telemetry_lifespan("<service>", settings):` around the app's life.
"""

from helpdesk.telemetry.asgi import fastapi_telemetry_config, instrument_app
from helpdesk.telemetry.setup import (
    Telemetry,
    TelemetryConfigError,
    setup_telemetry,
    shutdown_telemetry,
    telemetry_lifespan,
)

__all__ = [
    "Telemetry",
    "TelemetryConfigError",
    "fastapi_telemetry_config",
    "instrument_app",
    "setup_telemetry",
    "shutdown_telemetry",
    "telemetry_lifespan",
]
