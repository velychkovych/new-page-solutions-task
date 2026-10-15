"""Tool binding and Claude prompt caching.

The installed langchain-anthropic only offers AnthropicPromptCachingMiddleware, which
works inside `create_agent`. This graph is a custom StateGraph, so the middleware's
three steps are replicated here: cache_control on the last system block, on the last
tool definition, and as a top-level request kwarg. ChatAnthropic sends the kwarg as-is
to the Claude API; ChatAnthropicBedrock expands it into a block-level breakpoint.
"""

from collections.abc import Sequence
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import Runnable

from helpdesk.agent.tool_provider import ToolDefinition

# Same value the middleware uses by default.
CACHE_CONTROL: dict[str, str] = {"type": "ephemeral", "ttl": "5m"}


def is_anthropic_family(model: BaseChatModel) -> bool:
    # ChatAnthropicBedrock subclasses ChatAnthropic, so Bedrock is included.
    return isinstance(model, ChatAnthropic)


def _anthropic_tools(definitions: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = [
        {"name": d.name, "description": d.description, "input_schema": d.input_schema}
        for d in definitions
    ]
    if tools:
        # One breakpoint after the last tool caches the whole contiguous tool block.
        tools[-1] = {**tools[-1], "cache_control": dict(CACHE_CONTROL)}
    return tools


def _openai_tools(definitions: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": d.name,
                "description": d.description,
                "parameters": d.input_schema,
            },
        }
        for d in definitions
    ]


def bind_tool_definitions(
    model: BaseChatModel, definitions: Sequence[ToolDefinition]
) -> Runnable[LanguageModelInput, AIMessage]:
    if is_anthropic_family(model):
        return model.bind_tools(_anthropic_tools(definitions), cache_control=dict(CACHE_CONTROL))
    return model.bind_tools(_openai_tools(definitions))


def cacheable_system_message(system: SystemMessage, model: BaseChatModel) -> SystemMessage:
    """The system prompt as a text block marked cacheable, for Anthropic-family models."""
    if not is_anthropic_family(model) or not isinstance(system.content, str):
        return system
    block = {"type": "text", "text": system.content, "cache_control": dict(CACHE_CONTROL)}
    return SystemMessage(content=[block], id=system.id)
