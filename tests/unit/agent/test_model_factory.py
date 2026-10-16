"""Model factory per CONTRACTS §7, and the prompt-caching markers (PLAN §1)."""

import json

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatAnthropicBedrock
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from helpdesk.agent.fake_model import ScriptedChatModel, parse_script
from helpdesk.agent.memory import REMEMBER_FACT_DEFINITION
from helpdesk.agent.model import ModelConfigError, build_chat_model
from helpdesk.agent.prompt import system_message
from helpdesk.agent.prompt_caching import (
    CACHE_CONTROL,
    bind_tool_definitions,
    cacheable_system_message,
)
from helpdesk.agent.tool_provider import ToolDefinition
from helpdesk.shared.settings import MissingSecretError, Settings

DUMMY = {
    "anthropic_api_key": SecretStr("sk-ant-dummy"),
    "openai_api_key": SecretStr("sk-dummy"),
    "aws_region": "us-east-1",
}


@pytest.fixture
def make_settings(tmp_path):
    def make(provider: str, **extra) -> Settings:
        return Settings(
            _env_file=None,
            model_provider=provider,
            model_max_tokens=321,
            model_timeout_s=12.5,
            model_max_retries=4,
            traces_dir=tmp_path / "traces",
            otel_exporter_otlp_endpoint=None,
            **{**DUMMY, **extra},
        )

    return make


@pytest.mark.parametrize(
    ("provider", "cls"),
    [
        ("anthropic", ChatAnthropic),
        ("bedrock", ChatAnthropicBedrock),
        ("openai", ChatOpenAI),
    ],
)
def test_factory_builds_the_right_class_without_calling_it(provider, cls, make_settings):
    settings = make_settings(provider, model_name="some-model-from-settings")
    model = build_chat_model(settings)
    assert type(model) is cls
    assert model.max_retries == 4
    assert model.max_tokens == 321
    # The name comes only from Settings.
    name = model.model_name if isinstance(model, ChatOpenAI) else model.model
    assert name == "some-model-from-settings"


def test_factory_never_sets_temperature(make_settings):
    for provider in ("anthropic", "bedrock", "openai"):
        model = build_chat_model(make_settings(provider))
        assert model.temperature is None, provider
        assert "temperature" not in model.model_fields_set, provider


def test_factory_passes_timeout_and_region(make_settings):
    anthropic = build_chat_model(make_settings("anthropic"))
    assert anthropic.default_request_timeout == 12.5
    assert anthropic.anthropic_api_key.get_secret_value() == "sk-ant-dummy"
    bedrock = build_chat_model(make_settings("bedrock"))
    assert bedrock.region_name == "us-east-1"
    openai = build_chat_model(make_settings("openai"))
    assert openai.request_timeout == 12.5


def test_factory_requires_the_api_key(make_settings):
    with pytest.raises(MissingSecretError, match="ANTHROPIC_API_KEY"):
        build_chat_model(make_settings("anthropic", anthropic_api_key=None))


def test_fake_provider_reads_the_script(tmp_path, make_settings):
    script = tmp_path / "script.json"
    script.write_text(
        json.dumps(
            [
                {"tool_calls": [{"name": "search_tickets", "args": {"query": "vpn"}}]},
                {"content": "Found it."},
            ]
        )
    )
    model = build_chat_model(make_settings("fake", fake_model_script=script))
    assert isinstance(model, ScriptedChatModel)
    assert model.bind_tools([]) is model


def test_fake_provider_needs_a_script(make_settings):
    with pytest.raises(ModelConfigError, match="FAKE_MODEL_SCRIPT"):
        build_chat_model(make_settings("fake"))


async def test_scripted_model_keeps_tool_calls_when_streamed():
    script = [
        {"tool_calls": [{"name": "get_ticket", "args": {"ticket_id": "INC-0007"}}]},
        {"content": "In progress."},
    ]
    model = ScriptedChatModel.from_script(parse_script(json.dumps(script)))
    chunks = [chunk async for chunk in model.astream([HumanMessage("status?")])]
    assert len(chunks) == 1
    assert chunks[0].tool_calls[0]["name"] == "get_ticket"
    assert chunks[0].tool_calls[0]["args"] == {"ticket_id": "INC-0007"}
    second = await model.ainvoke([HumanMessage("and?")])
    assert second.content == "In progress."
    again = await model.ainvoke([HumanMessage("cycle")])
    # A fresh object every time, so add_messages never replaces an earlier reply.
    assert again is not chunks[0]
    assert again.id is None or again.id != chunks[0].id


DEFINITIONS = [
    ToolDefinition(
        name="search_tickets",
        description="Search.",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    ),
    REMEMBER_FACT_DEFINITION,
]


@pytest.mark.parametrize("provider", ["anthropic", "bedrock"])
def test_prompt_caching_markers_reach_the_anthropic_payload(provider, make_settings):
    model = build_chat_model(make_settings(provider))
    bound = bind_tool_definitions(model, DEFINITIONS)
    system = cacheable_system_message(system_message([]), model)
    payload = model._get_request_payload([system, HumanMessage("hi")], **bound.kwargs)

    assert payload["system"][-1]["cache_control"] == CACHE_CONTROL
    tools = payload["tools"]
    assert [t["name"] for t in tools] == ["search_tickets", "remember_fact"]
    assert tools[-1]["cache_control"] == CACHE_CONTROL
    assert "cache_control" not in tools[0]
    if provider == "anthropic":
        # The direct API takes the top-level breakpoint as-is.
        assert payload["cache_control"] == CACHE_CONTROL
    else:
        # Bedrock gets it expanded onto the last message block instead.
        assert "cache_control" not in payload
        assert payload["messages"][-1]["content"][-1]["cache_control"] == CACHE_CONTROL
    assert "temperature" not in payload or payload["temperature"] is None


def test_no_caching_markers_for_other_providers(make_settings):
    model = build_chat_model(make_settings("openai"))
    system = system_message([])
    assert cacheable_system_message(system, model) is system
    bound = bind_tool_definitions(model, DEFINITIONS)
    assert "cache_control" not in bound.kwargs
    assert all("cache_control" not in tool for tool in bound.kwargs["tools"])
