"""Entry point: `python -m helpdesk.agent` (CONTRACTS.md §7)."""

import asyncio
import logging

import uvicorn

from helpdesk.agent.api import create_app
from helpdesk.agent.mcp_provider import McpToolProvider
from helpdesk.agent.model import ModelConfigError, build_chat_model
from helpdesk.shared.settings import MissingSecretError, Settings


async def serve(settings: Settings) -> None:
    # Built before serving, so a missing model key or fake script stops the process at once.
    model = build_chat_model(settings)
    app = create_app(
        settings=settings, tool_provider=McpToolProvider.from_settings(settings), model=model
    )
    config = uvicorn.Config(
        app,
        host=settings.agent_host,
        port=settings.agent_port,
        log_level=settings.log_level.lower(),
        # The lifespan sets up telemetry, opens the checkpointer and store, and lists the
        # MCP tools. With "auto", uvicorn 0.54 takes an error raised before Starlette
        # reports startup.failed as "lifespan unsupported" and serves without them. With
        # "on", a failed startup (e.g. the MCP server unreachable) exits with status 3.
        lifespan="on",
    )
    await uvicorn.Server(config).serve()


def main(settings: Settings | None = None) -> None:
    settings = settings or Settings()
    logging.basicConfig(
        level=settings.log_level.upper(), format="%(levelname)s %(name)s %(message)s"
    )
    try:
        asyncio.run(serve(settings))
    except (MissingSecretError, ModelConfigError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
