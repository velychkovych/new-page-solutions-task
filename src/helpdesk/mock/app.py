"""The mock ticketing API application (CONTRACTS §1)."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI

from helpdesk.mock.admin_routes import build_admin_router
from helpdesk.mock.auth import AuthMiddleware
from helpdesk.mock.chaos import ChaosEngine, ChaosMiddleware, Sleep
from helpdesk.mock.errors import install_error_handlers
from helpdesk.mock.idempotency import IdempotentWriter
from helpdesk.mock.seed import seed_database
from helpdesk.mock.ticket_routes import Clock, TicketServices, build_ticket_router
from helpdesk.shared.settings import Settings, require_secret


def _utc_now() -> datetime:
    return datetime.now(UTC)


def create_app(
    settings: Settings, *, sleep: Sleep = asyncio.sleep, clock: Clock = _utc_now
) -> FastAPI:
    """Build the app. Raises `MissingSecretError` if either token is unset, so the service
    refuses to start rather than run with no auth."""
    api_token = require_secret(settings.mock_api_token, "MOCK_API_TOKEN")
    admin_token = require_secret(settings.mock_admin_token, "MOCK_ADMIN_TOKEN")
    db_path = settings.mock_db_path

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Non-reset seed: fills a new database, leaves an existing one alone.
        await seed_database(db_path)
        yield

    chaos = ChaosEngine(settings)
    app = FastAPI(
        title="Helpdesk mock ticketing API",
        lifespan=lifespan,
        # Not part of the contract; also keeps every non-2xx body an ApiError.
        openapi_url=None,
        docs_url=None,
        redoc_url=None,
    )
    install_error_handlers(app)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    services = TicketServices(
        db_path=db_path, writer=IdempotentWriter(db_path), chaos=chaos, clock=clock
    )
    app.include_router(build_ticket_router(services))
    app.include_router(build_admin_router(db_path, chaos))

    # add_middleware prepends, so the last one added is outermost: auth runs before chaos.
    app.add_middleware(ChaosMiddleware, engine=chaos, sleep=sleep)
    app.add_middleware(AuthMiddleware, api_token=api_token, admin_token=admin_token)
    return app
