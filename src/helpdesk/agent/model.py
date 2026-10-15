"""The model factory: MODEL_PROVIDER → chat model (CONTRACTS §7).

This is the only code path that builds a model. No temperature is passed for any
provider: the installed langchain-anthropic rejects non-default values for
claude-sonnet-5-5. The model name only ever comes from Settings.
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


def _provider_kwargs(settings: Settings) -> dict[str, Any]:
    match settings.model_provider:
        case ModelProvider.ANTHROPIC:
            return {"api_key": require_secret(settings.anthropic_api_key, "ANTHROPIC_API_KEY")}
        case ModelProvider.BEDROCK:
            # Credentials come from the AWS default chain; the region from Settings.
            return {"region_name": settings.aws_region}
        case ModelProvider.OPENAI:
            return {"api_key": require_secret(settings.openai_api_key, "OPENAI_API_KEY")}
        case ModelProvider.FAKE:
            return {}


def build_chat_model(settings: Settings) -> BaseChatModel:
    if settings.model_provider is ModelProvider.FAKE:
        if settings.fake_model_script is None:
            raise ModelConfigError("MODEL_PROVIDER=fake needs FAKE_MODEL_SCRIPT")
        return ScriptedChatModel.from_script_file(settings.fake_model_script)
    model: BaseChatModel = init_chat_model(
        settings.model_name,
        model_provider=INIT_CHAT_MODEL_PROVIDER[settings.model_provider],
        max_tokens=settings.model_max_tokens,
        timeout=settings.model_timeout_s,
        max_retries=settings.model_max_retries,
        **_provider_kwargs(settings),
    )
    return model
