"""Static checks on `agent-eval`, the compose service the live evals run against.

Needs only the docker CLI (no daemon). Network isolation and secret scoping for it are in
tests/integration/test_compose_isolation.py with the other agents.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from evals.live_stack import AGENT_URL, PROFILE, SERVICE, TRACE_DIRS

ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not installed")


def _config(**env: str) -> dict[str, Any]:
    # --env-file /dev/null: resolve with dummy values, never with the developer's real .env.
    # The developer's MODEL_NAME must not decide the default this test sees.
    host = {key: value for key, value in os.environ.items() if key != "MODEL_NAME"}
    full_env = {
        **host,
        "MOCK_API_TOKEN": "dummy-api",
        "MOCK_ADMIN_TOKEN": "dummy-admin",
        "ANTHROPIC_API_KEY": "dummy-model-key",
        **env,
    }
    out = subprocess.run(
        ["docker", "compose", "--env-file", os.devnull, "--profile", "e2e", "--profile", PROFILE,
         "config", "--format", "json"],
        cwd=ROOT, env=full_env, capture_output=True, text=True, check=True,
    )  # fmt: skip
    config: dict[str, Any] = json.loads(out.stdout)
    return config


@pytest.fixture(scope="module")
def services() -> dict[str, Any]:
    services: dict[str, Any] = _config()["services"]
    return services


def _volumes(service: dict[str, Any]) -> set[str]:
    return {v["source"] for v in service["volumes"] if v["type"] == "volume"}


def test_it_is_only_in_the_eval_profile(services: dict[str, Any]) -> None:
    assert services[SERVICE]["profiles"] == [PROFILE]


def test_it_runs_the_agent_like_the_browser_agent(services: dict[str, Any]) -> None:
    eval_agent, agent = services[SERVICE], services["agent"]
    assert eval_agent["image"] == agent["image"]
    assert eval_agent["command"] == agent["command"]
    assert eval_agent["stop_grace_period"] == agent["stop_grace_period"]
    assert eval_agent["networks"].keys() == agent["networks"].keys()
    same = ("MODEL_PROVIDER", "MODEL_NAME", "MODEL_MAX_TOKENS", "MODEL_TIMEOUT_S", "MCP_URL")
    for key in same:
        assert eval_agent["environment"][key] == agent["environment"][key], key
    assert eval_agent["environment"]["ANTHROPIC_API_KEY"] == "dummy-model-key"


def test_it_keeps_its_own_state_port_and_spans(services: dict[str, Any]) -> None:
    eval_agent = services[SERVICE]
    for other in ("agent", "agent-e2e"):
        assert _volumes(eval_agent).isdisjoint(_volumes(services[other])), other
        other_traces = services[other]["environment"]["TRACES_DIR"]
        assert eval_agent["environment"]["TRACES_DIR"] != other_traces, other
    assert eval_agent["environment"]["TRACES_DIR"] == "/app/traces/eval"
    # The runner reads its spans from ./traces/eval and the mcp/mock spans from ./traces.
    assert TRACE_DIRS == (ROOT / "traces", ROOT / "traces" / "eval")
    (port,) = eval_agent["ports"]
    assert (port["host_ip"], port["published"], port["target"]) == ("127.0.0.1", "8082", 8080)
    assert AGENT_URL == "http://127.0.0.1:8082"


def test_model_name_comes_from_the_environment() -> None:
    # `--model` works by setting MODEL_NAME for `docker compose up`.
    services = _config(MODEL_NAME="claude-haiku-5-5")["services"]
    assert services[SERVICE]["environment"]["MODEL_NAME"] == "claude-haiku-5-5"
