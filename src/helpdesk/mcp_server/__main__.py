"""Entry point: `python -m helpdesk.mcp_server` (CONTRACTS.md §7)."""

import asyncio
import logging

import uvicorn

from helpdesk.mcp_server.mock_client import MockApiClient, create_mock_http_client
from helpdesk.mcp_server.redaction import SecretRedactor
from helpdesk.mcp_server.retry import RetryPolicy
from helpdesk.mcp_server.server import build_server
from helpdesk.shared.settings import MissingSecretError, Settings, require_secret


async def serve(settings: Settings) -> None:
    # Checked before anything else starts: without the token every tool call would fail.
    token = require_secret(settings.mock_api_token, "MOCK_API_TOKEN")
    async with create_mock_http_client(settings, token) as http:
        client = MockApiClient(http, RetryPolicy.from_settings(settings))
        server = build_server(settings, client, redactor=SecretRedactor([token]))
        config = uvicorn.Config(
            server.streamable_http_app(),
            host=settings.mcp_host,
            port=settings.mcp_port,
            log_level=settings.log_level.lower(),
        )
        await uvicorn.Server(config).serve()


def main(settings: Settings | None = None) -> None:
    settings = settings or Settings()
    logging.basicConfig(
        level=settings.log_level.upper(), format="%(levelname)s %(name)s %(message)s"
    )
    try:
        asyncio.run(serve(settings))
    except MissingSecretError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
