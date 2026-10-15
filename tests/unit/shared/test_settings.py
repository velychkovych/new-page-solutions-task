from pathlib import Path

import pytest
from pydantic import SecretStr

from helpdesk.shared.settings import MissingSecretError, ModelProvider, Settings, require_secret

ROOT = Path(__file__).resolve().parents[3]


def test_defaults_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    settings = Settings(_env_file=None)
    assert settings.model_provider is ModelProvider.ANTHROPIC
    assert settings.model_name == "claude-sonnet-5-5"
    assert settings.retry_max_attempts == 3
    assert settings.tool_deadline_s == 15.0
    assert settings.recursion_limit == 25


def test_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PROVIDER", "fake")
    monkeypatch.setenv("CHAOS_ERROR_RATE", "0.25")
    settings = Settings(_env_file=None)
    assert settings.model_provider is ModelProvider.FAKE
    assert settings.chaos_error_rate == 0.25


def test_rejects_rate_outside_unit_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAOS_ERROR_RATE", "1.5")
    with pytest.raises(ValueError, match="less than or equal to 1"):
        Settings(_env_file=None)


def test_secrets_are_masked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_API_TOKEN", "super-secret-value")
    settings = Settings(_env_file=None)
    assert "super-secret-value" not in repr(settings)
    assert "super-secret-value" not in str(settings.model_dump())


def test_require_secret() -> None:
    assert require_secret(SecretStr("t0k"), "MOCK_API_TOKEN") == "t0k"
    for missing in (None, SecretStr("")):
        with pytest.raises(MissingSecretError, match="MOCK_API_TOKEN is not set"):
            require_secret(missing, "MOCK_API_TOKEN")


def _clear_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


def test_empty_values_mean_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Compose passes `${VAR:-}` as an empty string; .env.example has `KEY=` lines.
    _clear_settings_env(monkeypatch)
    for name in ("CHAOS_SEED", "MODEL_TEMPERATURE", "FAKE_MODEL_SCRIPT", "AWS_REGION"):
        monkeypatch.setenv(name, "")
    settings = Settings(_env_file=None)
    assert settings.chaos_seed == 0
    assert settings.model_temperature is None
    assert settings.fake_model_script is None
    assert settings.aws_region is None


def test_loads_env_example(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_settings_env(monkeypatch)
    settings = Settings(_env_file=ROOT / ".env.example")
    assert settings.mock_api_token is None
    assert settings.model_temperature is None
    assert settings.otel_exporter_otlp_endpoint is None


def test_chaos_failure_rates_must_fit_one_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_settings_env(monkeypatch)
    monkeypatch.setenv("CHAOS_RATE_LIMIT_RATE", "0.6")
    monkeypatch.setenv("CHAOS_ERROR_RATE", "0.5")
    with pytest.raises(ValueError, match="must be <= 1"):
        Settings(_env_file=None)
