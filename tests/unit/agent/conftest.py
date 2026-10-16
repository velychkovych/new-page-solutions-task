"""Fakes for the agent tests: a recording chat model and a recording tool provider.

Nothing here touches the network or a real model. Helpers are exposed as fixtures
because tests run with --import-mode=importlib, where sibling modules are not importable.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import Field, JsonValue

from helpdesk.agent.graph import AgentGraph, build_graph
from helpdesk.agent.persistence import open_checkpointer, open_store
from helpdesk.agent.tool_provider import (
    ProgressCallback,
    ToolDefinition,
    ToolResult,
    ToolTransportError,
)
from helpdesk.shared.settings import Settings
from helpdesk.shared.tools import AddCommentArgs, CreateTicketArgs, GetTicketArgs, SearchTicketsArgs


class ScriptExhaustedError(AssertionError):
    pass


class RecordingChatModel(FakeMessagesListChatModel):
    """Scripted replies (tool calls kept), recording every call and every bind_tools."""

    calls: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[Any] = Field(default_factory=list)
    bound_kwargs: dict[str, Any] = Field(default_factory=dict)
    raises: Any = None
    delay_s: float = 0.0
    repeat_last: bool = False

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: Any = None, **kwargs: Any) -> Any:
        self.bound_tools = list(tools)
        self.bound_kwargs = kwargs
        return self

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.raises is not None:
            raise self.raises
        if self.i >= len(self.responses):
            if not (self.repeat_last and self.responses):
                raise ScriptExhaustedError(f"model called {len(self.calls)} times")
            response = self.responses[-1]
        else:
            response = self.responses[self.i]
            self.i += 1
        message = response.model_copy(deep=True, update={"id": None})
        return ChatResult(generations=[ChatGeneration(message=message)])


def ai(text: str = "", *calls: tuple[str, dict[str, Any], str]) -> AIMessage:
    """An AI reply; each call is (name, args, id)."""
    return AIMessage(
        content=text,
        tool_calls=[{"name": name, "args": args, "id": call_id} for name, args, call_id in calls],
    )


@dataclass
class RecordedCall:
    name: str
    args: dict[str, JsonValue]
    idempotency_key: str | None


type Reply = ToolResult | BaseException | Callable[[Mapping[str, JsonValue]], Awaitable[ToolResult]]


def _definition(name: str, description: str, model: type) -> ToolDefinition:
    return ToolDefinition(
        name=name, description=description, input_schema=model.model_json_schema()
    )


@dataclass
class FakeToolProvider:
    """Records every call (name, args, idempotency key) and replies from a per-tool queue."""

    replies: dict[str, list[Reply]] = field(default_factory=dict)
    calls: list[RecordedCall] = field(default_factory=list)
    delay_s: float = 0.0
    # Progress messages sent to a call's `on_progress` (when it passes one) before it replies.
    progress: dict[str, list[str]] = field(default_factory=dict)

    async def list_tools(self) -> Sequence[ToolDefinition]:
        return [
            _definition(
                "search_tickets",
                "Search tickets. Always call before proposing a new ticket, to find duplicates.",
                SearchTicketsArgs,
            ),
            _definition("get_ticket", "Read one ticket with its comments.", GetTicketArgs),
            _definition("create_ticket", "Create a ticket (needs approval).", CreateTicketArgs),
            _definition("add_comment", "Comment on a ticket (needs approval).", AddCommentArgs),
        ]

    async def call(
        self,
        name: str,
        args: Mapping[str, JsonValue],
        *,
        on_progress: ProgressCallback | None = None,
    ) -> ToolResult:
        key = args.get("idempotency_key")
        self.calls.append(RecordedCall(name, dict(args), key if isinstance(key, str) else None))
        if on_progress is not None:
            for message in self.progress.get(name, []):
                await on_progress(message)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        queue = self.replies.get(name)
        if queue:
            reply = queue.pop(0)
            if isinstance(reply, BaseException):
                raise reply
            if isinstance(reply, ToolResult):
                return reply
            return await reply(args)
        return ToolResult(text=json.dumps({"tool": name, "ok": True}), is_error=False)

    def writes(self) -> list[RecordedCall]:
        return [c for c in self.calls if c.name in {"create_ticket", "add_comment"}]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        checkpoint_db_path=tmp_path / "checkpoints.sqlite",
        store_db_path=tmp_path / "store.sqlite",
        model_provider="fake",
        # The app lifespan sets telemetry up: spans go under tmp_path, never ./traces or OTLP.
        traces_dir=tmp_path / "traces",
        otel_exporter_otlp_endpoint=None,
    )


@pytest.fixture
def provider() -> FakeToolProvider:
    return FakeToolProvider()


@pytest.fixture
def spans() -> tuple[TracerProvider, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return tracer_provider, exporter


type GraphFactory = Callable[..., Awaitable[AgentGraph]]


@pytest.fixture
async def make_graph(settings: Settings) -> AsyncIterator[GraphFactory]:
    """Builds graphs on the test's SQLite files; every graph gets its own connections."""
    async with AsyncExitStack() as stack:

        async def factory(
            model: RecordingChatModel, tool_provider: FakeToolProvider, **overrides: Any
        ) -> AgentGraph:
            graph_settings = settings.model_copy(update=overrides) if overrides else settings
            checkpointer = await stack.enter_async_context(
                open_checkpointer(graph_settings.checkpoint_db_path)
            )
            store = await stack.enter_async_context(
                open_store(graph_settings.store_db_path, fact_ttl_days=graph_settings.fact_ttl_days)
            )
            return await build_graph(
                model=model,
                tool_provider=tool_provider,
                settings=graph_settings,
                checkpointer=checkpointer,
                store=store,
            )

        yield factory


type ClientFactory = Callable[..., Awaitable[httpx.AsyncClient]]


@pytest.fixture
async def make_client(
    settings: Settings, spans: tuple[TracerProvider, InMemorySpanExporter]
) -> AsyncIterator[ClientFactory]:
    """An httpx client on the ASGI app, with the app's lifespan running."""
    async with AsyncExitStack() as stack:

        async def factory(
            model: RecordingChatModel, tool_provider: FakeToolProvider, **overrides: Any
        ) -> httpx.AsyncClient:
            from helpdesk.agent.api import create_app  # noqa: PLC0415 - keeps graph tests independent

            app_settings = settings.model_copy(update=overrides) if overrides else settings
            app = create_app(
                settings=app_settings,
                tool_provider=tool_provider,
                model=model,
                tracer=spans[0].get_tracer("tests"),
            )
            await stack.enter_async_context(app.router.lifespan_context(app))
            transport = httpx.ASGITransport(app=app)
            return await stack.enter_async_context(
                httpx.AsyncClient(transport=transport, base_url="http://agent.test")
            )

        yield factory


@pytest.fixture
def fake_model() -> type[RecordingChatModel]:
    return RecordingChatModel


@pytest.fixture
def make_ai() -> Callable[..., AIMessage]:
    return ai


@pytest.fixture
def transport_error() -> type[ToolTransportError]:
    return ToolTransportError
