from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.util._once import Once
from pydantic import SecretStr

from helpdesk.shared.settings import Settings
from helpdesk.telemetry.setup import _reset_for_tests

MOCK_API_TOKEN = "mock-api-token-4f8a2c9e1b7d"
MOCK_ADMIN_TOKEN = "mock-admin-token-9c1e7a3f5b2d"
ANTHROPIC_API_KEY = "sk-ant-api03-test-key-0d9f8e7c"
OPENAI_API_KEY = "sk-openai-test-key-6b5a4c3d"
SECRETS = (MOCK_API_TOKEN, MOCK_ADMIN_TOKEN, ANTHROPIC_API_KEY, OPENAI_API_KEY)

type MakeSettings = Callable[..., Settings]


@pytest.fixture(autouse=True)
def fresh_otel(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Start each test as a fresh process would: no global provider, nothing instrumented.

    The OTel API allows `set_tracer_provider` once per process (a `Once` guard plus a
    module global, see opentelemetry/trace/__init__.py); monkeypatch puts both back, so a
    test can also see the delegating provider being installed, or install a foreign one.
    """
    monkeypatch.setattr(trace, "_TRACER_PROVIDER_SET_ONCE", Once())
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", None)
    # Keep a developer's shell from turning on OTLP export or FastAPI's own telemetry.
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("FASTAPI_OTEL_AUTO_CONFIGURE", raising=False)
    yield
    # Closes any setup whatever its user count, and undoes the process-singleton
    # instrumentors, so other tests (and other tracks' tests) start clean.
    _reset_for_tests()


@pytest.fixture
def make_settings(tmp_path: Path) -> MakeSettings:
    def make(**overrides: Any) -> Settings:
        values: dict[str, Any] = {
            "traces_dir": tmp_path / "traces",
            "mock_api_token": SecretStr(MOCK_API_TOKEN),
            "mock_admin_token": SecretStr(MOCK_ADMIN_TOKEN),
            "anthropic_api_key": SecretStr(ANTHROPIC_API_KEY),
            "openai_api_key": SecretStr(OPENAI_API_KEY),
            "otel_exporter_otlp_endpoint": None,
        }
        values.update(overrides)
        return Settings(_env_file=None, **values)

    return make


@pytest.fixture
def secrets() -> tuple[str, ...]:
    """The configured secret values `make_settings` puts into Settings."""
    return SECRETS
