"""Idempotent writes (CONTRACTS §1.5).

Every write runs under one in-process lock and a `BEGIN IMMEDIATE` transaction, and the
idempotency record is inserted in that same transaction. So a committed write always has
its key, a rolled-back write never does, and a concurrent request with the same key waits,
then finds the stored result instead of executing again.
"""

import asyncio
import hashlib
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import aiosqlite
from pydantic import BaseModel, JsonValue

from helpdesk.mock.db import transaction
from helpdesk.mock.errors import (
    IdempotencyConflictError,
    InvalidIdempotencyKeyError,
    MissingIdempotencyKeyError,
)
from helpdesk.shared.ids import IDEMPOTENCY_KEY_PATTERN

_KEY_RE = re.compile(IDEMPOTENCY_KEY_PATTERN)

type WriteOperation = Callable[[aiosqlite.Connection], Awaitable[BaseModel]]


def validate_key(raw: str | None) -> str:
    if raw is None:
        raise MissingIdempotencyKeyError("send an Idempotency-Key header")
    # fullmatch: `$` alone would also accept a trailing newline.
    if _KEY_RE.fullmatch(raw) is None:
        raise InvalidIdempotencyKeyError("the key must be 1-255 printable ASCII chars, no spaces")
    return raw


def request_fingerprint(method: str, path: str, body: BaseModel) -> str:
    """Method, path and the sha256 of the validated body as canonical (sorted-key) JSON."""
    canonical = json.dumps(
        body.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return f"{method.upper()} {path} {hashlib.sha256(canonical.encode()).hexdigest()}"


@dataclass(frozen=True, slots=True)
class WriteOutcome:
    status_code: int
    body: JsonValue
    # False for a replay: the handler did not run, so fail-after-commit must not apply.
    executed: bool


class IdempotentWriter:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        # BEGIN IMMEDIATE already serialises writers across processes; the lock keeps
        # same-process writers from spinning on SQLITE_BUSY.
        self._lock = asyncio.Lock()

    async def run(
        self, key: str, fingerprint: str, status_code: int, operation: WriteOperation
    ) -> WriteOutcome:
        """Execute `operation` once per key. Only a successful (2xx) result is stored: an
        exception inside `operation` rolls back the write and leaves the key unused."""
        async with self._lock, transaction(self._db_path, immediate=True) as conn:
            async with conn.execute(
                "SELECT fingerprint, status_code, body FROM idempotency_records WHERE key = ?",
                (key,),
            ) as cursor:
                stored = await cursor.fetchone()
            if stored is not None:
                if stored["fingerprint"] != fingerprint:
                    raise IdempotencyConflictError(
                        "this key was already used for a different request"
                    )
                body: JsonValue = json.loads(stored["body"])
                return WriteOutcome(stored["status_code"], body, executed=False)

            result = await operation(conn)
            body = result.model_dump(mode="json")
            await conn.execute(
                "INSERT INTO idempotency_records (key, fingerprint, status_code, body) "
                "VALUES (?, ?, ?, ?)",
                (key, fingerprint, status_code, json.dumps(body, separators=(",", ":"))),
            )
            await conn.execute(
                "INSERT INTO write_counts (key, count) VALUES (?, 1) "
                "ON CONFLICT(key) DO UPDATE SET count = count + 1",
                (key,),
            )
            return WriteOutcome(status_code, body, executed=True)


async def writes_by_key(db_path: Path) -> dict[str, int]:
    async with transaction(db_path) as conn:
        rows = await conn.execute_fetchall("SELECT key, count FROM write_counts ORDER BY key")
    return {row["key"]: row["count"] for row in rows}
