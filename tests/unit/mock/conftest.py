"""In-process mock harness: the real app over httpx.ASGITransport, a tmp SQLite file,
a fake clock and a recording sleep. No network, no real waiting."""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
from pydantic import SecretStr

from helpdesk.mock.app import create_app
from helpdesk.shared.settings import Settings

API_TOKEN = "unit-api-token-7f3a"
ADMIN_TOKEN = "unit-admin-token-91c2"
AUTH = {"Authorization": f"Bearer {API_TOKEN}"}
ADMIN = {**AUTH, "X-Admin-Token": ADMIN_TOKEN}
FIXED_NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)

NEW_TICKET: dict[str, Any] = {
    "title": "Monitor flickers",
    "description": "The external monitor flickers every few seconds.",
    "category": "hardware",
    "priority": "low",
    "reporter_id": "alice",
}
NEW_COMMENT: dict[str, Any] = {"author_id": "alice", "body": "Still broken this morning."}
CHAOS_OFF: dict[str, Any] = {
    "enabled": False,
    "rate_limit_rate": 0,
    "error_rate": 0,
    "slow_rate": 0,
    "slow_ms": 0,
}


@dataclass
class FakeClock:
    now: datetime = FIXED_NOW

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@dataclass
class RecordingSleep:
    calls: list[float] = field(default_factory=list)

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@dataclass
class Mock:
    """The harness. Test modules cannot import conftest, so constants ride along here."""

    API_TOKEN: ClassVar[str] = API_TOKEN
    ADMIN_TOKEN: ClassVar[str] = ADMIN_TOKEN
    AUTH: ClassVar[dict[str, str]] = AUTH
    ADMIN: ClassVar[dict[str, str]] = ADMIN
    FIXED_NOW: ClassVar[datetime] = FIXED_NOW
    NEW_TICKET: ClassVar[dict[str, Any]] = NEW_TICKET
    NEW_COMMENT: ClassVar[dict[str, Any]] = NEW_COMMENT

    client: httpx.AsyncClient
    settings: Settings
    clock: FakeClock
    sleep: RecordingSleep

    async def create(
        self, key: str, body: dict[str, Any] | None = None, **headers: str
    ) -> httpx.Response:
        return await self.client.post(
            "/tickets",
            json=NEW_TICKET if body is None else body,
            headers={**AUTH, "Idempotency-Key": key, **headers},
        )

    async def comment(
        self, ticket_id: str, key: str, body: dict[str, Any] | None = None
    ) -> httpx.Response:
        return await self.client.post(
            f"/tickets/{ticket_id}/comments",
            json=NEW_COMMENT if body is None else body,
            headers={**AUTH, "Idempotency-Key": key},
        )

    async def chaos(self, **config: Any) -> httpx.Response:
        response = await self.client.post(
            "/admin/chaos", json={**CHAOS_OFF, **config}, headers=ADMIN
        )
        return response

    async def writes(self) -> dict[str, Any]:
        response = await self.client.get("/admin/writes", headers=ADMIN)
        assert response.status_code == 200
        return response.json()  # type: ignore[no-any-return]

    async def all_ids(self) -> list[str]:
        response = await self.client.get("/tickets", params={"limit": 20}, headers=AUTH)
        ids = [item["id"] for item in response.json()["items"]]
        while (cursor := response.json()["next_cursor"]) is not None:
            response = await self.client.get(
                "/tickets", params={"limit": 20, "cursor": cursor}, headers=AUTH
            )
            ids += [item["id"] for item in response.json()["items"]]
        return ids


type MockFactory = Callable[..., AbstractAsyncContextManager[Mock]]


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # A developer's shell (CHAOS_ENABLED=true, say) must not change unit test behaviour.
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


def make_settings(traces_dir: Path, **overrides: Any) -> Settings:
    # Every app started here sets up telemetry: spans go to a tmp dir, never to OTLP.
    values: dict[str, Any] = {
        "mock_api_token": SecretStr(API_TOKEN),
        "mock_admin_token": SecretStr(ADMIN_TOKEN),
        "traces_dir": traces_dir,
        "otel_exporter_otlp_endpoint": None,
        **overrides,
    }
    return Settings(_env_file=None, **values)


@pytest.fixture
def mock_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings for a test that builds and drives the app itself, outside the harness."""
    _clear_env(monkeypatch)
    return make_settings(tmp_path / "traces", mock_db_path=tmp_path / "mock.sqlite")


@pytest.fixture
def mock_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MockFactory:
    _clear_env(monkeypatch)
    counter = 0

    @asynccontextmanager
    async def factory(**overrides: Any) -> AsyncIterator[Mock]:
        nonlocal counter
        counter += 1
        overrides.setdefault("mock_db_path", tmp_path / f"mock-{counter}.sqlite")
        settings = make_settings(tmp_path / "traces", **overrides)
        clock, sleep = FakeClock(), RecordingSleep()
        app = create_app(settings, sleep=sleep, clock=clock)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://mock"
            ) as client,
        ):
            yield Mock(client=client, settings=settings, clock=clock, sleep=sleep)

    return factory


@pytest.fixture
async def mock(mock_factory: MockFactory) -> AsyncIterator[Mock]:
    async with mock_factory() as harness:
        yield harness
