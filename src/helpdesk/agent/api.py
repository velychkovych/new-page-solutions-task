"""FastAPI app factory for the agent (CONTRACTS §3.1; /chat/stream and / come in Phase 2)."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi import Path as PathParam
from langchain_core.language_models import BaseChatModel
from opentelemetry.trace import Tracer

from helpdesk.agent.graph import build_graph
from helpdesk.agent.memory import Clock, utc_now
from helpdesk.agent.persistence import open_checkpointer, open_store
from helpdesk.agent.service import ChatService
from helpdesk.agent.tool_provider import ToolProvider
from helpdesk.shared.api import ChatRequest, ChatResponse, HealthResponse, SessionView
from helpdesk.shared.ids import PRINCIPAL_ID_PATTERN
from helpdesk.shared.settings import Settings

PrincipalPath = Annotated[str, PathParam(pattern=PRINCIPAL_ID_PATTERN)]


class ServiceNotReadyError(RuntimeError):
    """A request arrived before the lifespan opened the checkpointer and store."""


class _Runtime:
    service: ChatService | None = None


def create_app(
    *,
    settings: Settings,
    tool_provider: ToolProvider,
    model: BaseChatModel,
    tracer: Tracer | None = None,
    clock: Clock = utc_now,
) -> FastAPI:
    """Tests pass a fake tool provider and model; Phase 2 passes the MCP provider."""
    runtime = _Runtime()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Both are async context managers: they stay open for the app's whole life.
        async with (
            open_checkpointer(settings.checkpoint_db_path) as checkpointer,
            open_store(settings.store_db_path, fact_ttl_days=settings.fact_ttl_days) as store,
        ):
            graph = await build_graph(
                model=model,
                tool_provider=tool_provider,
                settings=settings,
                checkpointer=checkpointer,
                store=store,
                clock=clock,
            )
            runtime.service = ChatService(graph=graph, settings=settings, tracer=tracer)
            try:
                yield
            finally:
                runtime.service = None

    def chat_service() -> ChatService:
        if runtime.service is None:
            raise ServiceNotReadyError("agent service is not started")
        return runtime.service

    service_dep = Depends(chat_service)
    app = FastAPI(title="IT helpdesk agent", lifespan=lifespan)

    @app.post("/chat")
    async def chat(body: ChatRequest, service: Annotated[ChatService, service_dep]) -> ChatResponse:
        return await service.chat(body)

    @app.get("/sessions/{user_id}/{session_id}")
    async def session(
        user_id: PrincipalPath,
        session_id: PrincipalPath,
        service: Annotated[ChatService, service_dep],
    ) -> SessionView:
        return await service.session(user_id, session_id)

    @app.get("/health")
    async def health() -> HealthResponse:
        return HealthResponse()

    return app
