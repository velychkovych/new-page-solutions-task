"""Every environment variable the three services read, in one place.

Services must read config only through `Settings`. Defaults are for running a
service locally with `uv run`; docker-compose overrides hosts, URLs and paths.
Secrets default to None so a missing token fails loudly at startup instead of
silently using a well-known dev value.
"""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Rate = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveSeconds = Annotated[float, Field(gt=0.0)]


class ModelProvider(StrEnum):
    """Config values. The model factory maps them to `init_chat_model` providers (CONTRACTS §7)."""

    ANTHROPIC = "anthropic"
    # -> init_chat_model provider "anthropic_bedrock" (ChatAnthropicBedrock), not legacy "bedrock".
    BEDROCK = "bedrock"
    OPENAI = "openai"
    FAKE = "fake"


class MissingSecretError(RuntimeError):
    """A service started without a secret it cannot work without."""

    def __init__(self, env_var: str) -> None:
        super().__init__(f"{env_var} is not set. Run `make setup` or add it to .env.")
        self.env_var = env_var


def require_secret(value: SecretStr | None, env_var: str) -> str:
    """Unwrap a secret at the single point of use; never log the result."""
    if value is None or not value.get_secret_value():
        raise MissingSecretError(env_var)
    return value.get_secret_value()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # `KEY=` in .env and `${KEY:-}` in compose mean "use the default", not "".
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
        protected_namespaces=(),
    )

    log_level: str = "INFO"

    # --- mock ticketing API -------------------------------------------------
    mock_host: str = "127.0.0.1"
    mock_port: int = 8000
    mock_db_path: Path = Path("data/mock.sqlite")
    mock_api_token: SecretStr | None = None
    mock_admin_token: SecretStr | None = None

    # Chaos defaults apply at mock startup; POST /admin/chaos overrides them at runtime.
    chaos_enabled: bool = False
    chaos_rate_limit_rate: Rate = 0.0
    chaos_error_rate: Rate = 0.0
    chaos_slow_rate: Rate = 0.0
    chaos_slow_ms: Annotated[int, Field(ge=0, le=60_000)] = 2_000
    # Fixed default so chaos runs are reproducible even when nobody passes a seed.
    chaos_seed: int = 0
    chaos_fail_after_commit_rate: Rate = 0.0
    chaos_retry_after_s: Annotated[int, Field(ge=0, le=60)] = 1

    # --- MCP server ----------------------------------------------------------
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8001
    mock_url: str = "http://127.0.0.1:8000"
    retry_max_attempts: Annotated[int, Field(ge=1, le=10)] = 3
    retry_backoff_base_s: PositiveSeconds = 0.5
    retry_after_cap_s: PositiveSeconds = 5.0
    tool_attempt_timeout_s: PositiveSeconds = 5.0
    tool_deadline_s: PositiveSeconds = 15.0

    # --- agent ---------------------------------------------------------------
    agent_host: str = "127.0.0.1"
    agent_port: int = 8080
    mcp_url: str = "http://127.0.0.1:8001/mcp"
    model_provider: ModelProvider = ModelProvider.ANTHROPIC
    model_name: str = "claude-sonnet-5-5"
    model_max_tokens: Annotated[int, Field(ge=1)] = 1_024
    model_timeout_s: PositiveSeconds = 60.0
    # SDK retries for the *model* API (429/529). Tool retries live only in the MCP server.
    model_max_retries: Annotated[int, Field(ge=0, le=10)] = 2
    # The eval judge (LLM-as-judge). Built by the same factory and reuses the MODEL_MAX_TOKENS,
    # MODEL_TIMEOUT_S and MODEL_MAX_RETRIES limits; only the provider and name differ.
    judge_model_provider: ModelProvider = ModelProvider.ANTHROPIC
    judge_model_name: str = "claude-haiku-5-5"
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    aws_region: str | None = None
    # JSON file of scripted AI messages, used when model_provider is "fake" (e2e tests).
    fake_model_script: Path | None = None
    checkpoint_db_path: Path = Path("data/checkpoints.sqlite")
    store_db_path: Path = Path("data/store.sqlite")
    recursion_limit: Annotated[int, Field(ge=1)] = 25
    turn_deadline_s: PositiveSeconds = 90.0
    history_token_budget: Annotated[int, Field(ge=1)] = 8_000
    fact_ttl_days: Annotated[int, Field(ge=1)] = 90
    slow_tool_notice_s: PositiveSeconds = 3.0
    web_dir: Path = Path("web")

    # --- telemetry -----------------------------------------------------------
    traces_dir: Path = Path("traces")
    otel_exporter_otlp_endpoint: str | None = None
    jaeger_ui_url: str | None = None
    trace_result_max_bytes: Annotated[int, Field(ge=1)] = 2_048

    @model_validator(mode="after")
    def _chaos_failure_rates_fit_one_draw(self) -> Self:
        if self.chaos_rate_limit_rate + self.chaos_error_rate > 1.0:
            raise ValueError("CHAOS_RATE_LIMIT_RATE + CHAOS_ERROR_RATE must be <= 1")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
