"""SQLite checkpointer and Store, opened for the lifetime of the agent service."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.base import TTLConfig
from langgraph.store.sqlite.aio import AsyncSqliteStore

from helpdesk.shared.approval import PendingAction, WriteResult
from helpdesk.shared.errors import ErrorCode
from helpdesk.shared.tools import ToolName

# The only non-langchain types the state holds (CONTRACTS §5.2). With an explicit
# allowlist the serializer is strict whatever LANGGRAPH_STRICT_MSGPACK says: anything
# else in a checkpoint is returned as raw data instead of being imported and called.
CHECKPOINT_TYPES: tuple[type, ...] = (PendingAction, WriteResult, ToolName, ErrorCode)

MINUTES_PER_DAY = 24 * 60


def checkpoint_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)


@asynccontextmanager
async def open_checkpointer(path: Path) -> AsyncIterator[AsyncSqliteSaver]:
    # Same as AsyncSqliteSaver.from_conn_string, which cannot take a serializer.
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(str(path)) as conn:
        saver = AsyncSqliteSaver(conn, serde=checkpoint_serializer())
        await saver.setup()
        yield saver


def fact_ttl_config(ttl_days: int) -> TTLConfig:
    # Reads must not extend the TTL: the expiry we filter on is fixed at write time.
    return TTLConfig(default_ttl=float(ttl_days * MINUTES_PER_DAY), refresh_on_read=False)


@asynccontextmanager
async def open_store(path: Path, *, fact_ttl_days: int) -> AsyncIterator[AsyncSqliteStore]:
    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteStore.from_conn_string(
        str(path), ttl=fact_ttl_config(fact_ttl_days)
    ) as store:
        await store.setup()
        yield store
