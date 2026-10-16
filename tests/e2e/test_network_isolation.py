"""Against the running stack (`make up`): no agent container can reach the mock.

Every agent is checked: `agent`, its fake-model twin `agent-e2e` and the eval agent
`agent-eval` (`make test-e2e` starts both).
"""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.e2e

AGENTS = pytest.mark.parametrize("agent", ["agent", "agent-e2e", "agent-eval"])


def _python_in(service: str, code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", "--profile", "e2e", "--profile", "eval", "exec", "-T", service,
         "python", "-c", code],
        cwd=ROOT, capture_output=True, text=True, timeout=30, check=False,
    )  # fmt: skip


RESOLVE = "import socket; socket.getaddrinfo('{host}', None)"


def _resolves(service: str, host: str) -> bool:
    return _python_in(service, RESOLVE.format(host=host)).returncode == 0


@AGENTS
def test_agent_cannot_resolve_mock(agent: str) -> None:
    result = _python_in(agent, RESOLVE.format(host="mock"))
    assert result.returncode != 0
    assert "gaierror" in result.stderr


@AGENTS
def test_agent_can_resolve_mcp(agent: str) -> None:
    # Positive control: without it, a broken exec would make the test above pass vacuously.
    assert _resolves(agent, "mcp")


def test_mcp_can_resolve_mock() -> None:
    assert _resolves("mcp", "mock")


@AGENTS
def test_agent_cannot_reach_mock_through_the_host(agent: str) -> None:
    # Docker Desktop routes host.docker.internal to the host's published ports.
    url = "http://host.docker.internal:8000/health"
    result = _python_in(agent, f"import urllib.request; urllib.request.urlopen('{url}', timeout=3)")
    assert result.returncode != 0


@AGENTS
def test_agent_env_has_no_mock_token(agent: str) -> None:
    result = _python_in(agent, "import os; print('MOCK_API_TOKEN' in os.environ)")
    assert result.returncode == 0
    assert result.stdout.strip() == "False"
