"""SQLite connection handling and schema for the mock.

A fresh connection per operation keeps every request's transaction isolated, and lets
`python -m helpdesk.mock.seed --reset` run from another process (`make seed`) while the
server is up: nothing is cached in memory.
"""

import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    description   TEXT NOT NULL,
    status        TEXT NOT NULL,
    priority      TEXT NOT NULL,
    category      TEXT NOT NULL,
    reporter_id   TEXT NOT NULL,
    assignee_team TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS comments (
    seq        INTEGER PRIMARY KEY,
    id         TEXT NOT NULL UNIQUE,
    ticket_id  TEXT NOT NULL REFERENCES tickets(id),
    author_id  TEXT NOT NULL,
    body       TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS comments_by_ticket ON comments(ticket_id, seq);
CREATE TABLE IF NOT EXISTS idempotency_records (
    key         TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    body        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS write_counts (
    key   TEXT PRIMARY KEY,
    count INTEGER NOT NULL
);
"""

# Order matters only for the foreign key: comments before tickets.
ALL_TABLES = ("comments", "tickets", "idempotency_records", "write_counts")

# Long enough to wait out a concurrent `seed --reset` from another process.
BUSY_TIMEOUT_MS = 5_000


@asynccontextmanager
async def connect(db_path: Path) -> AsyncIterator[aiosqlite.Connection]:
    # isolation_level=None: no implicit BEGIN, so every transaction is explicit and
    # writes can take the lock up front with BEGIN IMMEDIATE.
    async with aiosqlite.connect(db_path, isolation_level=None) as conn:
        conn.row_factory = sqlite3.Row
        await conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        await conn.execute("PRAGMA foreign_keys = ON")
        yield conn


@asynccontextmanager
async def transaction(
    db_path: Path, *, immediate: bool = False
) -> AsyncIterator[aiosqlite.Connection]:
    """Commit on success, roll back on any exception (including a 404 mid-write)."""
    async with connect(db_path) as conn:
        await conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield conn
        except BaseException:
            await conn.execute("ROLLBACK")
            raise
        await conn.execute("COMMIT")


async def init_schema(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with connect(db_path) as conn:
        # WAL lets readers proceed while a write transaction holds the lock.
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.executescript(SCHEMA)
