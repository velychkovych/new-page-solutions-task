"""`python -m helpdesk.agent`: builds the model and the MCP provider, then serves the app."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI

from helpdesk.agent import __main__ as entry
from helpdesk.agent.fake_model import ScriptedChatModel
from helpdesk.agent.mcp_provider import McpToolProvider
from helpdesk.shared.settings import MissingSecretError, Settings

SCRIPT = '[{"content": "Hello from the fake model."}]'


class _MustNotServe:
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("the server must not be started")


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


async def test_serve_refuses_without_a_model_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uvicorn, "Server", _MustNotServe)
    with pytest.raises(MissingSecretError, match="ANTHROPIC_API_KEY is not set"):
        await entry.serve(_settings(model_provider="anthropic", anthropic_api_key=None))


def test_main_exits_with_a_message_when_the_fake_model_has_no_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(uvicorn, "Server", _MustNotServe)
    with pytest.raises(SystemExit) as caught:
        entry.main(_settings(model_provider="fake", fake_model_script=None))
    assert "FAKE_MODEL_SCRIPT" in str(caught.value.code)


async def test_serve_builds_the_app_and_binds_the_configured_address(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = tmp_path / "script.json"
    script.write_text(SCRIPT, encoding="utf-8")
    configs: list[uvicorn.Config] = []
    built: list[dict[str, Any]] = []
    real_create_app = entry.create_app

    def recording_create_app(**kwargs: Any) -> FastAPI:
        built.append(kwargs)
        return real_create_app(**kwargs)

    class _RecordingServer:
        def __init__(self, config: uvicorn.Config) -> None:
            configs.append(config)

        async def serve(self) -> None:
            return None

    monkeypatch.setattr(uvicorn, "Server", _RecordingServer)
    monkeypatch.setattr(entry, "create_app", recording_create_app)
    settings = _settings(
        model_provider="fake",
        fake_model_script=script,
        agent_host="0.0.0.0",  # noqa: S104 - compose binding
        agent_port=8123,
        mcp_url="http://mcp:8001/mcp",
    )
    await entry.serve(settings)

    (config,) = configs
    assert (config.host, config.port) == ("0.0.0.0", 8123)  # noqa: S104
    assert config.lifespan == "on"
    # Bounded, so a connected stream cannot hold shutdown before the lifespan's drain.
    assert config.timeout_graceful_shutdown == entry.GRACEFUL_SHUTDOWN_S
    assert isinstance(config.app, FastAPI)
    (kwargs,) = built
    assert kwargs["settings"] is settings
    assert isinstance(kwargs["tool_provider"], McpToolProvider)
    assert isinstance(kwargs["model"], ScriptedChatModel)


def test_module_entry_point_exits_without_a_model_key(tmp_path: Path) -> None:
    # tmp_path as cwd, so no developer .env is picked up.
    drop = {"ANTHROPIC_API_KEY", "MODEL_PROVIDER", "FAKE_MODEL_SCRIPT"}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    completed = subprocess.run(
        [sys.executable, "-m", "helpdesk.agent"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 1
    assert "ANTHROPIC_API_KEY is not set" in completed.stderr
