"""Ticket and comment queries. Every function runs on a caller-owned connection, so a write
and its idempotency record share one transaction (CONTRACTS §1.5)."""

import sqlite3
from datetime import datetime

import aiosqlite

from helpdesk.mock.errors import TicketIdsExhaustedError, TicketNotFoundError
from helpdesk.mock.search import SearchFilter
from helpdesk.shared.tickets import (
    AddCommentRequest,
    Comment,
    CreateTicketRequest,
    Ticket,
    TicketStatus,
    TicketSummary,
)

_TICKET_ID_MAX = 9_999
_TICKET_COLUMNS = (
    "id, title, description, status, priority, category, reporter_id, assignee_team, "
    "created_at, updated_at"
)


def _ticket_id(number: int) -> str:
    return f"INC-{number:04d}"


def _comment_id(seq: int) -> str:
    return f"CMT-{seq:06d}"


def _comment(row: sqlite3.Row) -> Comment:
    return Comment(
        id=row["id"],
        author_id=row["author_id"],
        body=row["body"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _ticket(row: sqlite3.Row, comments: list[Comment]) -> Ticket:
    return Ticket(
        id=row["id"],
        title=row["title"],
        description=row["description"],
        status=row["status"],
        priority=row["priority"],
        category=row["category"],
        reporter_id=row["reporter_id"],
        assignee_team=row["assignee_team"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        comments=comments,
    )


async def get_ticket(conn: aiosqlite.Connection, ticket_id: str) -> Ticket:
    async with conn.execute(
        f"SELECT {_TICKET_COLUMNS} FROM tickets WHERE id = ?",  # noqa: S608 - constant columns
        (ticket_id,),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        raise TicketNotFoundError(f"ticket {ticket_id} does not exist")
    comment_rows = await conn.execute_fetchall(
        "SELECT id, author_id, body, created_at FROM comments WHERE ticket_id = ? ORDER BY seq",
        (ticket_id,),
    )
    return _ticket(row, [_comment(r) for r in comment_rows])


async def search_tickets(conn: aiosqlite.Connection, search: SearchFilter) -> list[TicketSummary]:
    """Every match in `id` order. Matching runs in Python: SQLite's lower() is ASCII-only,
    and the mock holds a few dozen rows."""
    sql = "SELECT id, title, description, status, priority, category, updated_at FROM tickets"
    params: list[str] = []
    if search.statuses:
        sql += f" WHERE status IN ({', '.join('?' * len(search.statuses))})"
        params.extend(status.value for status in search.statuses)
    rows = await conn.execute_fetchall(f"{sql} ORDER BY id", params)
    return [
        TicketSummary(
            id=row["id"],
            title=row["title"],
            status=row["status"],
            priority=row["priority"],
            category=row["category"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
        for row in rows
        if search.matches(row["title"], row["description"])
    ]


async def insert_ticket(conn: aiosqlite.Connection, ticket: Ticket) -> None:
    await conn.execute(
        f"INSERT INTO tickets ({_TICKET_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",  # noqa: S608
        (
            ticket.id,
            ticket.title,
            ticket.description,
            ticket.status.value,
            ticket.priority.value,
            ticket.category.value,
            ticket.reporter_id,
            ticket.assignee_team,
            ticket.created_at.isoformat(),
            ticket.updated_at.isoformat(),
        ),
    )
    for comment in ticket.comments:
        await _insert_comment(conn, ticket.id, comment)


async def _insert_comment(conn: aiosqlite.Connection, ticket_id: str, comment: Comment) -> None:
    await conn.execute(
        "INSERT INTO comments (id, ticket_id, author_id, body, created_at) VALUES (?, ?, ?, ?, ?)",
        (comment.id, ticket_id, comment.author_id, comment.body, comment.created_at.isoformat()),
    )


async def _next_number(conn: aiosqlite.Connection, sql: str) -> int:
    async with conn.execute(sql) as cursor:
        row = await cursor.fetchone()
    current = row[0] if row is not None and row[0] is not None else 0
    return int(current) + 1


async def create_ticket(
    conn: aiosqlite.Connection, request: CreateTicketRequest, now: datetime
) -> Ticket:
    # Sequential ids from the current maximum; the caller holds the write lock.
    number = await _next_number(conn, "SELECT MAX(CAST(substr(id, 5) AS INTEGER)) FROM tickets")
    if number > _TICKET_ID_MAX:
        raise TicketIdsExhaustedError("no ticket ids left")
    ticket = Ticket(
        id=_ticket_id(number),
        title=request.title,
        description=request.description,
        status=TicketStatus.OPEN,
        priority=request.priority,
        category=request.category,
        reporter_id=request.reporter_id,
        assignee_team=None,
        created_at=now,
        updated_at=now,
        comments=[],
    )
    await insert_ticket(conn, ticket)
    return ticket


async def add_comment(
    conn: aiosqlite.Connection, ticket_id: str, request: AddCommentRequest, now: datetime
) -> Comment:
    """Commenting on a resolved or closed ticket is allowed (CONTRACTS §1.3)."""
    cursor = await conn.execute(
        "UPDATE tickets SET updated_at = ? WHERE id = ?", (now.isoformat(), ticket_id)
    )
    if cursor.rowcount == 0:
        raise TicketNotFoundError(f"ticket {ticket_id} does not exist")
    seq = await _next_number(conn, "SELECT MAX(seq) FROM comments")
    comment = Comment(
        id=_comment_id(seq), author_id=request.author_id, body=request.body, created_at=now
    )
    await conn.execute(
        "INSERT INTO comments (seq, id, ticket_id, author_id, body, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (seq, comment.id, ticket_id, comment.author_id, comment.body, now.isoformat()),
    )
    return comment
