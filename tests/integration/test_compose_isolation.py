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
# Every agent: the browser one, the fake-model one the e2e tests use (profile e2e) and the
# one the live evals use (profile eval).
AGENTS = ("agent", "agent-e2e", "agent-eval")

pytestmark = pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not installed")


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    # --env-file /dev/null: resolve with dummy values, never with the developer's real .env.
    env = {**os.environ, "MOCK_API_TOKEN": "dummy-api", "MOCK_ADMIN_TOKEN": "dummy-admin"}
    out = subprocess.run(
        [
            "docker", "compose", "--env-file", os.devnull, "--profile", "observability",
            "--profile", "e2e", "--profile", "eval", "config", "--format", "json",
        ],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True,
    )  # fmt: skip
    return json.loads(out.stdout)  # type: ignore[no-any-return]


def _networks(compose: dict[str, Any], service: str) -> set[str]:
    return set(compose["services"][service]["networks"])


def _env(compose: dict[str, Any], service: str) -> dict[str, str]:
    return compose["services"][service].get("environment") or {}


@pytest.mark.parametrize("agent", AGENTS)
def test_agent_and_mock_share_no_network(compose: dict[str, Any], agent: str) -> None:
    assert _networks(compose, agent).isdisjoint(_networks(compose, "mock"))


@pytest.mark.parametrize("agent", AGENTS)
def test_mcp_bridges_agent_and_mock(compose: dict[str, Any], agent: str) -> None:
    mcp = _networks(compose, "mcp")
    assert _networks(compose, agent) <= mcp
    assert _networks(compose, "mock") <= mcp


@pytest.mark.parametrize("agent", AGENTS)
def test_jaeger_does_not_change_agent_mock_isolation(compose: dict[str, Any], agent: str) -> None:
    # Joining both networks must not put the agent and the mock on a common network.
    assert "jaeger" in compose["services"]
    assert _networks(compose, agent).isdisjoint(_networks(compose, "mock"))


@pytest.mark.parametrize("agent", AGENTS)
def test_agent_has_no_mock_credentials_or_url(compose: dict[str, Any], agent: str) -> None:
    env = _env(compose, agent)
    assert not {"MOCK_API_TOKEN", "MOCK_ADMIN_TOKEN", "MOCK_URL"} & set(env)
    assert env["MCP_URL"].startswith("http://mcp:")


def test_secrets_reach_only_their_service(compose: dict[str, Any]) -> None:
    assert "MOCK_ADMIN_TOKEN" not in _env(compose, "mcp")
    for service in ("mock", "mcp"):
        assert not {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"} & set(_env(compose, service))


def test_e2e_agent_gets_no_model_key(compose: dict[str, Any]) -> None:
    env = _env(compose, "agent-e2e")
    assert env["MODEL_PROVIDER"] == "fake"
    assert not {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"} & set(env)


def test_e2e_agent_keeps_its_own_state(compose: dict[str, Any]) -> None:
    # Its own checkpoints and its own span file: the two agents never share either.
    e2e, agent = compose["services"]["agent-e2e"], compose["services"]["agent"]
    volume_sources = {
        name: {v["source"] for v in service["volumes"] if v["type"] == "volume"}
        for name, service in (("agent", agent), ("agent-e2e", e2e))
    }
    assert volume_sources["agent"].isdisjoint(volume_sources["agent-e2e"])
    assert _env(compose, "agent-e2e")["TRACES_DIR"] != _env(compose, "agent")["TRACES_DIR"]
    script = [v for v in e2e["volumes"] if v["target"] == "/app/e2e"]
    assert script
    assert script[0]["read_only"] is True
    assert _env(compose, "agent-e2e")["FAKE_MODEL_SCRIPT"].startswith("/app/e2e/")


def test_e2e_agent_is_only_in_the_e2e_profile(compose: dict[str, Any]) -> None:
    assert compose["services"]["agent-e2e"]["profiles"] == ["e2e"]


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


@pytest.mark.parametrize("service", ["mock", "mcp", *AGENTS])
def test_spans_reach_the_hosts_traces_directory(compose: dict[str, Any], service: str) -> None:
    # `make trace` and the e2e trace check read every service's JSONL from ./traces.
    mounts = {
        (v["source"], v["target"])
        for v in compose["services"][service].get("volumes", [])
        if v["type"] == "bind"
    }
    assert (str(ROOT / "traces"), "/app/traces") in mounts
    assert _env(compose, service)["TRACES_DIR"].startswith("/app/traces")


def test_core_services_have_healthchecks(compose: dict[str, Any]) -> None:
    for name in ("mock", "mcp", *AGENTS):
        assert compose["services"][name]["healthcheck"]["test"], name


@pytest.mark.parametrize("service", ["mock", "mcp", *AGENTS])
def test_settings_load_from_each_service_environment(
    compose: dict[str, Any], service: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    for key, value in _env(compose, service).items():
        monkeypatch.setenv(key, value or "")
    Settings(_env_file=None)
