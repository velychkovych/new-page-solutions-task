"""The `fake` model provider: scripted AI messages from FAKE_MODEL_SCRIPT (e2e tests).

Built on FakeMessagesListChatModel because it does not stream, so LangGraph's
"messages" stream mode falls back to a single invoke and tool calls survive.
(GenericFakeChatModel drops tool calls when streamed.)

The reply is chosen from the conversation, not from a counter in process memory, so an
agent restarted between a proposal and its approval continues the script where the
checkpointed conversation left off (tests/e2e/test_restart.py).
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, override

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter


class ScriptedToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    args: dict[str, JsonValue] = Field(default_factory=dict)
    id: str | None = None


class ScriptedMessage(BaseModel):
    """One scripted model reply: text, tool calls, or both."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str = ""
    tool_calls: list[ScriptedToolCall] = Field(default_factory=list)


_SCRIPT = TypeAdapter(list[ScriptedMessage])


def to_ai_messages(script: Sequence[ScriptedMessage]) -> list[BaseMessage]:
    return [
        AIMessage(
            content=step.content,
            tool_calls=[
                {"name": call.name, "args": call.args, "id": call.id or f"call_{i}_{j}"}
                for j, call in enumerate(step.tool_calls)
            ],
        )
        for i, step in enumerate(script)
    ]


def parse_script(text: str) -> list[ScriptedMessage]:
    script = _SCRIPT.validate_json(text)
    if not script:
        raise ValueError("FAKE_MODEL_SCRIPT must contain at least one message")
    return script


def script_position(messages: Sequence[BaseMessage]) -> int:
    """How many replies the model has given since the user's last message.

    Every user message starts the script again; each tool round in the same turn moves
    one step on. An approval adds no user message (the graph resumes), so the reply after
    an approved write is the step after the proposal.
    """
    position = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, AIMessage):
            position += 1
    return position


class ScriptedChatModel(FakeMessagesListChatModel):
    """Replies with script step `script_position(input)`, wrapping round past the end.

    The inherited in-memory counter `i` is not used.
    """

    @classmethod
    def from_script(cls, script: Sequence[ScriptedMessage]) -> ScriptedChatModel:
        return cls(responses=to_ai_messages(script))

    @classmethod
    def from_script_file(cls, path: Path) -> ScriptedChatModel:
        # Read once at startup, before the event loop serves requests.
        return cls.from_script(parse_script(path.read_text(encoding="utf-8")))

    @override
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        # The script already decides which tools are called.
        return self

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        step = self.responses[script_position(messages) % len(self.responses)]
        # A fresh copy per reply: add_messages assigns ids in place, and a reused
        # object would make a repeated step replace the earlier one in the history.
        message = step.model_copy(deep=True, update={"id": None})
        return ChatResult(generations=[ChatGeneration(message=message)])
