"""Admin routes (CONTRACTS §1.3). AuthMiddleware already required both tokens, and chaos
never applies here, so `make chaos-off` works even while chaos is failing every request."""

from pathlib import Path

from fastapi import APIRouter

from helpdesk.mock.chaos import ChaosEngine
from helpdesk.mock.idempotency import writes_by_key
from helpdesk.shared.tickets import ChaosConfig, WritesReport


def build_admin_router(db_path: Path, chaos: ChaosEngine) -> APIRouter:
    router = APIRouter(prefix="/admin")

    @router.post("/chaos")
    async def set_chaos(config: ChaosConfig) -> ChaosConfig:
        chaos.configure(config)
        return chaos.config

    @router.get("/writes")
    async def writes() -> WritesReport:
        by_key = await writes_by_key(db_path)
        return WritesReport(total=sum(by_key.values()), by_key=by_key)

    return router
