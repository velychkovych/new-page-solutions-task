"""The model factory: MODEL_PROVIDER → chat model, JUDGE_MODEL_PROVIDER → eval judge (CONTRACTS §7).

This is the only code path that builds a model. No temperature is passed for any
provider: the installed langchain-anthropic rejects non-default values for
claude-sonnet-5-5. Model names only ever come from Settings.
"""

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from helpdesk.agent.fake_model import ScriptedChatModel
from helpdesk.shared.settings import ModelProvider, Settings, require_secret

# MODEL_PROVIDER → init_chat_model provider. Bedrock uses ChatAnthropicBedrock (a
# ChatAnthropic subclass, so caching and error mapping are shared), not legacy "bedrock".
INIT_CHAT_MODEL_PROVIDER: dict[ModelProvider, str] = {
    ModelProvider.ANTHROPIC: "anthropic",
    ModelProvider.BEDROCK: "anthropic_bedrock",
    ModelProvider.OPENAI: "openai",
}


class ModelConfigError(ValueError):
    """The model settings cannot produce a model."""


def _provider_kwargs(settings: Settings, provider: ModelProvider) -> dict[str, Any]:
    match provider:
        case ModelProvider.ANTHROPIC:
            return {"api_key": require_secret(settings.anthropic_api_key, "ANTHROPIC_API_KEY")}
        case ModelProvider.BEDROCK:
            # Credentials come from the AWS default chain; the region from Settings.
            return {"region_name": settings.aws_region}
        case ModelProvider.OPENAI:
            return {"api_key": require_secret(settings.openai_api_key, "OPENAI_API_KEY")}
        case ModelProvider.FAKE:
            return {}


def _init_model(settings: Settings, provider: ModelProvider, name: str) -> BaseChatModel:
    # The agent and the judge share the limits: one place sets max_tokens, the request
    # timeout and the SDK retries, so the judge cannot grow a second retry policy.
    model: BaseChatModel = init_chat_model(
        name,
        model_provider=INIT_CHAT_MODEL_PROVIDER[provider],
        max_tokens=settings.model_max_tokens,
        timeout=settings.model_timeout_s,
        max_retries=settings.model_max_retries,
        **_provider_kwargs(settings, provider),
    )
    return model


def build_chat_model(settings: Settings) -> BaseChatModel:
    if settings.model_provider is ModelProvider.FAKE:
        if settings.fake_model_script is None:
            raise ModelConfigError("MODEL_PROVIDER=fake needs FAKE_MODEL_SCRIPT")
        return ScriptedChatModel.from_script_file(settings.fake_model_script)
    return _init_model(settings, settings.model_provider, settings.model_name)


def build_judge_model(settings: Settings) -> BaseChatModel:
    """The eval judge (LLM-as-judge), from JUDGE_MODEL_PROVIDER and JUDGE_MODEL_NAME."""
    if settings.judge_model_provider is ModelProvider.FAKE:
        # FAKE_MODEL_SCRIPT scripts the agent's replies; reusing it would make the judge
        # grade with agent turns. A fake judge is scripted by its caller, not built here.
        raise ModelConfigError(
            "JUDGE_MODEL_PROVIDER=fake has no built-in judge: pass a scripted judge model "
            "to the evals instead"
        )
    return _init_model(settings, settings.judge_model_provider, settings.judge_model_name)
