"""FastAPI app factory for the agent (CONTRACTS §3.1, §4)."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi import Path as PathParam
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.language_models import BaseChatModel
from opentelemetry.trace import Tracer

from helpdesk.agent.graph import build_graph
from helpdesk.agent.memory import Clock, utc_now
from helpdesk.agent.persistence import open_checkpointer, open_store
from helpdesk.agent.service import ChatService
from helpdesk.agent.tool_provider import ToolProvider
from helpdesk.agent.web_page import (
    FIXTURE_MEDIA_TYPES,
    WebPageError,
    fixture_path,
    render_index,
)
from helpdesk.shared.api import ChatRequest, ChatResponse, HealthResponse, SessionView
from helpdesk.shared.ids import PRINCIPAL_ID_PATTERN
from helpdesk.shared.settings import Settings
from helpdesk.telemetry import fastapi_telemetry_config, instrument_app, telemetry_lifespan

logger = logging.getLogger(__name__)

PrincipalPath = Annotated[str, PathParam(pattern=PRINCIPAL_ID_PATTERN)]

SERVICE_NAME = "agent"


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
    """Tests pass a fake tool provider and model; Phase 2 passes the MCP provider.

    `tracer` is for tests that read spans from a private provider; without it the request
    spans go to the process telemetry that the lifespan sets up.
    """
    runtime = _Runtime()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Telemetry is outermost so it is active for list_tools at startup and for every
        # request, and is flushed only after everything else has shut down. The
        # checkpointer and store are async context managers held for the app's whole life.
        async with (
            telemetry_lifespan(SERVICE_NAME, settings),
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
            service = ChatService(graph=graph, settings=settings, tracer=tracer)
            runtime.service = service
            try:
                yield
            finally:
                runtime.service = None
                # A streamed turn whose client left is still running: let it save its
                # checkpoint before the checkpointer closes.
                await service.wait_for_streamed_turns()

    def chat_service() -> ChatService:
        if runtime.service is None:
            raise ServiceNotReadyError("agent service is not started")
        return runtime.service

    service_dep = Depends(chat_service)
    app = FastAPI(
        title="IT helpdesk agent", telemetry=fastapi_telemetry_config(), lifespan=lifespan
    )
    # Before the app starts: Starlette builds its middleware stack only once.
    instrument_app(app)

    @app.post("/chat")
    async def chat(body: ChatRequest, service: Annotated[ChatService, service_dep]) -> ChatResponse:
        return await service.chat(body)

    @app.post("/chat/stream", response_class=EventSourceResponse)
    async def chat_stream(
        body: ChatRequest, service: Annotated[ChatService, service_dep]
    ) -> AsyncIterator[ServerSentEvent]:
        # The body is validated before this runs, so a bad request is a plain 422 JSON
        # response. FastAPI sets Cache-Control: no-cache and X-Accel-Buffering: no, and
        # sends ": ping" while the turn is quiet.
        async for event in service.stream(body):
            yield ServerSentEvent(event=event.name.value, data=event.data)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        try:
            page = await asyncio.to_thread(render_index, settings.web_dir, settings.jaeger_ui_url)
        except WebPageError as error:
            logger.warning("web page unavailable: %s", error)
            raise HTTPException(status_code=404) from None
        return HTMLResponse(page, headers={"Cache-Control": "no-cache"})

    @app.get("/fixtures/{name}")
    async def fixture(name: str) -> FileResponse:
        path = await asyncio.to_thread(fixture_path, settings.web_dir, name)
        if path is None:
            raise HTTPException(status_code=404)
        return FileResponse(path, media_type=FIXTURE_MEDIA_TYPES[path.suffix])

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
