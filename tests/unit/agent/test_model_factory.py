"""Model factory per CONTRACTS §7, and the prompt-caching markers (PLAN §1)."""

import json

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatAnthropicBedrock
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from helpdesk.agent.fake_model import ScriptedChatModel, parse_script, script_position
from helpdesk.agent.memory import REMEMBER_FACT_DEFINITION
from helpdesk.agent.model import ModelConfigError, build_chat_model, build_judge_model
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


def _model_name(model) -> str:
    return model.model_name if isinstance(model, ChatOpenAI) else model.model


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
    assert _model_name(model) == "some-model-from-settings"


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


@pytest.mark.parametrize(
    ("provider", "cls"),
    [
        ("anthropic", ChatAnthropic),
        ("bedrock", ChatAnthropicBedrock),
        ("openai", ChatOpenAI),
    ],
)
def test_judge_factory_builds_the_right_class_without_calling_it(provider, cls, make_settings):
    # The agent's provider is deliberately different: the judge reads only JUDGE_MODEL_*.
    settings = make_settings(
        "fake",
        model_name="agent-model",
        judge_model_provider=provider,
        judge_model_name="judge-model-from-settings",
    )
    judge = build_judge_model(settings)
    assert type(judge) is cls
    assert _model_name(judge) == "judge-model-from-settings"
    assert judge.max_tokens == 321
    assert judge.max_retries == 4
    assert judge.temperature is None
    assert "temperature" not in judge.model_fields_set


def test_judge_factory_passes_timeout_and_region(make_settings):
    anthropic = build_judge_model(make_settings("fake", judge_model_provider="anthropic"))
    assert anthropic.default_request_timeout == 12.5
    bedrock = build_judge_model(make_settings("fake", judge_model_provider="bedrock"))
    assert bedrock.region_name == "us-east-1"
    openai = build_judge_model(make_settings("fake", judge_model_provider="openai"))
    assert openai.request_timeout == 12.5


def test_judge_factory_defaults_to_haiku_and_leaves_the_agent_model_alone(make_settings):
    settings = make_settings("anthropic", model_name="claude-sonnet-5-5")
    judge = build_judge_model(settings)
    agent = build_chat_model(settings)
    assert judge.model == "claude-haiku-5-5"
    assert agent.model == "claude-sonnet-5-5"


def test_judge_factory_requires_the_api_key(make_settings):
    settings = make_settings("fake", judge_model_provider="anthropic", anthropic_api_key=None)
    with pytest.raises(MissingSecretError, match="ANTHROPIC_API_KEY"):
        build_judge_model(settings)


def test_judge_factory_never_builds_the_agents_scripted_model(tmp_path, make_settings):
    script = tmp_path / "script.json"
    script.write_text(json.dumps([{"content": "Agent reply."}]))
    settings = make_settings("fake", fake_model_script=script, judge_model_provider="fake")
    with pytest.raises(ModelConfigError, match="JUDGE_MODEL_PROVIDER=fake"):
        build_judge_model(settings)


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


SCRIPT = [
    {"tool_calls": [{"name": "search_tickets", "args": {"query": "printer"}, "id": "s"}]},
    {"tool_calls": [{"name": "create_ticket", "args": {"title": "t"}, "id": "w"}]},
    {"content": "Done."},
]


def _scripted() -> ScriptedChatModel:
    return ScriptedChatModel.from_script(parse_script(json.dumps(SCRIPT)))


def _step(model: ScriptedChatModel, step: int) -> AIMessage:
    expected = model.responses[step]
    assert isinstance(expected, AIMessage)
    return expected


async def test_scripted_model_replies_by_position_in_the_conversation():
    model = _scripted()
    history: list[BaseMessage] = [HumanMessage("printer jams")]
    replies = []
    for _ in range(3):
        reply = await model.ainvoke(history)
        replies.append(reply)
        history.append(reply)
        for call in reply.tool_calls:
            history.append(ToolMessage("{}", tool_call_id=call["id"]))
    assert [r.tool_calls for r in replies] == [_step(model, i).tool_calls for i in range(3)]
    assert replies[2].content == "Done."


async def test_every_user_message_starts_the_script_again():
    model = _scripted()
    first = await model.ainvoke([HumanMessage("one")])
    history = [HumanMessage("one"), first, ToolMessage("{}", tool_call_id="s")]
    history += [AIMessage("Done."), HumanMessage("two")]
    again = await model.ainvoke(history)
    assert again.tool_calls == first.tool_calls


async def test_scripted_model_has_no_memory_between_instances():
    # An approval resumes the graph without a user message. A new process (a restarted
    # agent) must give the same next step as the one that made the proposal.
    history: list[BaseMessage] = [HumanMessage("printer jams")]
    history += [_step(_scripted(), 0), ToolMessage("{}", tool_call_id="s")]
    history += [_step(_scripted(), 1), ToolMessage("{}", tool_call_id="w")]
    before = _scripted()
    await before.ainvoke([HumanMessage("unrelated")])
    after = await _scripted().ainvoke(history)
    assert after.content == "Done."
    assert (await before.ainvoke(history)).content == "Done."


async def test_scripted_model_wraps_round_past_the_end():
    model = _scripted()
    history: list[BaseMessage] = [HumanMessage("x"), *(AIMessage(f"r{i}") for i in range(3))]
    wrapped = await model.ainvoke(history)
    assert wrapped.tool_calls == _step(model, 0).tool_calls


async def test_scripted_model_returns_a_fresh_message_each_time():
    model = _scripted()
    first = await model.ainvoke([HumanMessage("a")])
    second = await model.ainvoke([HumanMessage("a")])
    # A fresh object every time, so add_messages never replaces an earlier reply.
    assert second is not first
    assert second.id is None or second.id != first.id
    assert model.responses[0].id is None


def test_script_position_counts_replies_since_the_last_user_message():
    assert script_position([]) == 0
    assert script_position([SystemMessage("sys"), HumanMessage("a")]) == 0
    history = [HumanMessage("a"), AIMessage("1"), HumanMessage("b"), AIMessage("2")]
    assert script_position([*history, ToolMessage("{}", tool_call_id="x")]) == 1
    # No user message at all: every reply counts.
    assert script_position([AIMessage("1"), AIMessage("2")]) == 2


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
