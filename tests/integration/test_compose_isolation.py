"""Static checks on the resolved compose file: network isolation and secret scoping.

Needs only the docker CLI (no daemon). The e2e twin in tests/e2e checks the same
thing against running containers.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from helpdesk.shared.settings import Settings

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not installed")


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    # --env-file /dev/null: resolve with dummy values, never with the developer's real .env.
    env = {**os.environ, "MOCK_API_TOKEN": "dummy-api", "MOCK_ADMIN_TOKEN": "dummy-admin"}
    out = subprocess.run(
        [
            "docker", "compose", "--env-file", os.devnull, "--profile", "observability",
            "config", "--format", "json",
        ],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True,
    )  # fmt: skip
    return json.loads(out.stdout)  # type: ignore[no-any-return]


def _networks(compose: dict[str, Any], service: str) -> set[str]:
    return set(compose["services"][service]["networks"])


def _env(compose: dict[str, Any], service: str) -> dict[str, str]:
    return compose["services"][service].get("environment") or {}


def test_agent_and_mock_share_no_network(compose: dict[str, Any]) -> None:
    assert _networks(compose, "agent").isdisjoint(_networks(compose, "mock"))


def test_mcp_bridges_agent_and_mock(compose: dict[str, Any]) -> None:
    mcp = _networks(compose, "mcp")
    assert _networks(compose, "agent") <= mcp
    assert _networks(compose, "mock") <= mcp


def test_jaeger_does_not_change_agent_mock_isolation(compose: dict[str, Any]) -> None:
    # Joining both networks must not put the agent and the mock on a common network.
    assert "jaeger" in compose["services"]
    assert _networks(compose, "agent").isdisjoint(_networks(compose, "mock"))


def test_agent_has_no_mock_credentials_or_url(compose: dict[str, Any]) -> None:
    env = _env(compose, "agent")
    assert not {"MOCK_API_TOKEN", "MOCK_ADMIN_TOKEN", "MOCK_URL"} & set(env)
    assert env["MCP_URL"].startswith("http://mcp:")


def test_secrets_reach_only_their_service(compose: dict[str, Any]) -> None:
    assert "MOCK_ADMIN_TOKEN" not in _env(compose, "mcp")
    for service in ("mock", "mcp"):
        assert not {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"} & set(_env(compose, service))


def test_no_service_loads_a_whole_env_file(compose: dict[str, Any]) -> None:
    for name, service in compose["services"].items():
        assert "env_file" not in service, name


def test_published_ports_are_loopback_only(compose: dict[str, Any]) -> None:
    for name, service in compose["services"].items():
        for port in service.get("ports", []):
            assert port.get("host_ip") == "127.0.0.1", name


def test_mock_publishes_no_port(compose: dict[str, Any]) -> None:
    # A published port is reachable from the agent container via host.docker.internal.
    assert not compose["services"]["mock"].get("ports")


def test_core_services_have_healthchecks(compose: dict[str, Any]) -> None:
    for name in ("mock", "mcp", "agent"):
        assert compose["services"][name]["healthcheck"]["test"], name


@pytest.mark.parametrize("service", ["mock", "mcp", "agent"])
def test_settings_load_from_each_service_environment(
    compose: dict[str, Any], service: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    for key, value in _env(compose, service).items():
        monkeypatch.setenv(key, value or "")
    Settings(_env_file=None)
