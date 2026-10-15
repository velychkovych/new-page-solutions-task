"""`python -m helpdesk.mcp_server` refuses to start without MOCK_API_TOKEN."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import uvicorn
from tests.unit.mcp_server.conftest import make_settings

from helpdesk.mcp_server import __main__ as entry
from helpdesk.shared.settings import MissingSecretError


class _MustNotServe:
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("the server must not be started without a token")


@pytest.mark.parametrize("token", [None, ""])
async def test_serve_refuses_without_token(
    token: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(uvicorn, "Server", _MustNotServe)
    with pytest.raises(MissingSecretError, match="MOCK_API_TOKEN is not set"):
        await entry.serve(make_settings(mock_api_token=token))


def test_main_exits_with_message_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uvicorn, "Server", _MustNotServe)
    with pytest.raises(SystemExit) as caught:
        entry.main(make_settings(mock_api_token=None))
    assert "MOCK_API_TOKEN is not set" in str(caught.value.code)


async def test_serve_binds_configured_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    configs: list[uvicorn.Config] = []

    class _RecordingServer:
        def __init__(self, config: uvicorn.Config) -> None:
            configs.append(config)

        async def serve(self) -> None:
            return None

    monkeypatch.setattr(uvicorn, "Server", _RecordingServer)
    await entry.serve(make_settings(mcp_host="0.0.0.0", mcp_port=8123))  # noqa: S104
    (config,) = configs
    assert (config.host, config.port) == ("0.0.0.0", 8123)  # noqa: S104


def test_module_entry_point_exits_without_token(tmp_path: Path) -> None:
    # tmp_path as cwd, so no developer .env is picked up.
    env = {k: v for k, v in os.environ.items() if k != "MOCK_API_TOKEN"}
    completed = subprocess.run(
        [sys.executable, "-m", "helpdesk.mcp_server"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 1
    assert "MOCK_API_TOKEN is not set" in completed.stderr
