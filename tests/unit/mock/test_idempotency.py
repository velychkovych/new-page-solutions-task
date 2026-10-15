"""CONTRACTS §1.5: idempotent writes."""

import asyncio
import hashlib
import json
import sqlite3

import aiosqlite
import pytest

from helpdesk.mock.db import connect
from helpdesk.mock.errors import InvalidIdempotencyKeyError, MissingIdempotencyKeyError
from helpdesk.mock.idempotency import IdempotentWriter, request_fingerprint, validate_key
from helpdesk.mock.repository import create_ticket
from helpdesk.shared.tickets import (
    AddCommentRequest,
    CreateTicketRequest,
    Ticket,
    TicketPriority,
)


async def test_replay_returns_stored_response_without_executing(mock) -> None:
    first = await mock.create("k-replay")
    mock.clock.advance(3600)  # a second execution would get a different timestamp and id
    second = await mock.create("k-replay")
    assert first.status_code == second.status_code == 201
    assert second.json() == first.json()
    assert second.headers["Idempotent-Replay"] == "true"
    assert "Idempotent-Replay" not in first.headers
    assert await mock.writes() == {"total": 1, "by_key": {"k-replay": 1}}
    assert (await mock.all_ids())[-1] == "INC-0016"
    assert len(await mock.all_ids()) == 16


async def test_comment_replay(mock) -> None:
    first = await mock.comment("INC-0003", "k-c")
    second = await mock.comment("INC-0003", "k-c")
    assert second.status_code == 201
    assert second.json() == first.json()
    assert second.headers["Idempotent-Replay"] == "true"
    ticket = (await mock.client.get("/tickets/INC-0003", headers=mock.AUTH)).json()
    assert len(ticket["comments"]) == 2


async def test_fingerprint_uses_the_validated_body(mock) -> None:
    await mock.create("k-canon")
    # Same fields in another order, with whitespace the model strips: same request.
    reordered = dict(reversed(list(mock.NEW_TICKET.items())))
    reordered["title"] = f"  {reordered['title']}  "
    replay = await mock.create("k-canon", reordered)
    assert replay.status_code == 201
    assert replay.headers["Idempotent-Replay"] == "true"


async def test_same_key_different_body_is_409(mock) -> None:
    await mock.create("k-conflict")
    response = await mock.create("k-conflict", {**mock.NEW_TICKET, "priority": "high"})
    assert response.status_code == 409
    assert response.json()["error"] == "idempotency_conflict"
    assert await mock.writes() == {"total": 1, "by_key": {"k-conflict": 1}}
    assert len(await mock.all_ids()) == 16


async def test_same_key_different_route_is_409(mock) -> None:
    await mock.create("k-route")
    comment = await mock.comment("INC-0003", "k-route")
    assert comment.status_code == 409
    assert comment.json()["error"] == "idempotency_conflict"
    other_ticket = await mock.comment("INC-0001", "k-c1")
    assert other_ticket.status_code == 201
    same_body_other_ticket = await mock.comment("INC-0003", "k-c1")
    assert same_body_other_ticket.status_code == 409
    ticket = (await mock.client.get("/tickets/INC-0003", headers=mock.AUTH)).json()
    assert len(ticket["comments"]) == 1
    assert await mock.writes() == {"total": 2, "by_key": {"k-route": 1, "k-c1": 1}}


async def test_missing_key_is_400(mock) -> None:
    for response in (
        await mock.client.post("/tickets", json=mock.NEW_TICKET, headers=mock.AUTH),
        await mock.client.post(
            "/tickets/INC-0003/comments", json=mock.NEW_COMMENT, headers=mock.AUTH
        ),
    ):
        assert response.status_code == 400
        assert response.json()["error"] == "missing_idempotency_key"
    assert await mock.writes() == {"total": 0, "by_key": {}}


@pytest.mark.parametrize("key", ["", "has space", "x" * 256, "tab\there"])
async def test_malformed_key_is_400(mock, key: str) -> None:
    response = await mock.create(key)
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_idempotency_key"


async def test_non_ascii_key_is_400(mock) -> None:
    response = await mock.client.post(
        "/tickets",
        json=mock.NEW_TICKET,
        headers=[
            ("Authorization", mock.AUTH["Authorization"].encode()),
            ("Idempotency-Key", "ключ".encode()),
        ],
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_idempotency_key"


async def test_longest_valid_key_is_accepted(mock) -> None:
    key = "~" * 255
    assert (await mock.create(key)).status_code == 201


async def test_422_is_not_stored(mock) -> None:
    bad = await mock.create("k-later", {**mock.NEW_TICKET, "priority": "urgent"})
    assert bad.status_code == 422
    good = await mock.create("k-later")
    assert good.status_code == 201
    assert "Idempotent-Replay" not in good.headers


@pytest.mark.parametrize("table", ["idempotency_records", "write_counts"])
async def test_write_and_its_key_commit_together(mock, table: str) -> None:
    """§1.5: the failure lands after the write but before its bookkeeping, so a write
    committed ahead of its key would survive here."""
    db_path = mock.settings.mock_db_path
    async with connect(db_path) as conn:
        await conn.execute(
            f"CREATE TRIGGER fail_{table} BEFORE INSERT ON {table} "
            "BEGIN SELECT RAISE(ABORT, 'injected'); END"
        )
    request = CreateTicketRequest.model_validate(mock.NEW_TICKET)

    async def operation(conn: aiosqlite.Connection) -> Ticket:
        return await create_ticket(conn, request, mock.FIXED_NOW)

    writer = IdempotentWriter(db_path)
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        await writer.run("k-atomic", "fp", 201, operation)
    assert len(await mock.all_ids()) == 15
    assert await mock.writes() == {"total": 0, "by_key": {}}
    async with connect(db_path) as conn:
        rows = await conn.execute_fetchall("SELECT key FROM idempotency_records")
    assert list(rows) == []


async def test_concurrent_same_key_executes_once(mock) -> None:
    responses = await asyncio.gather(*(mock.create("k-race") for _ in range(25)))
    assert {r.status_code for r in responses} == {201}
    assert len({r.text for r in responses}) == 1
    replays = [r for r in responses if r.headers.get("Idempotent-Replay") == "true"]
    assert len(replays) == 24
    assert await mock.writes() == {"total": 1, "by_key": {"k-race": 1}}
    assert len(await mock.all_ids()) == 16


async def test_concurrent_different_keys_get_distinct_ids(mock) -> None:
    responses = await asyncio.gather(*(mock.create(f"k-{n}") for n in range(10)))
    ids = sorted(r.json()["id"] for r in responses)
    assert ids == [f"INC-{n:04d}" for n in range(16, 26)]
    report = await mock.writes()
    assert report["total"] == 10
    assert set(report["by_key"].values()) == {1}


async def test_concurrent_same_key_comments_execute_once(mock) -> None:
    responses = await asyncio.gather(*(mock.comment("INC-0001", "k-cr") for _ in range(10)))
    assert {r.status_code for r in responses} == {201}
    ticket = (await mock.client.get("/tickets/INC-0001", headers=mock.AUTH)).json()
    assert len(ticket["comments"]) == 1


def test_validate_key() -> None:
    assert validate_key("3f2c-uuid") == "3f2c-uuid"
    with pytest.raises(MissingIdempotencyKeyError):
        validate_key(None)
    for bad in ("", "a b", "a\n", "\x7f", "x" * 256):
        with pytest.raises(InvalidIdempotencyKeyError):
            validate_key(bad)


def test_fingerprint_covers_method_path_and_body() -> None:
    body = CreateTicketRequest.model_validate(
        {
            "title": "t",
            "description": "d",
            "category": "other",
            "priority": "low",
            "reporter_id": "bob",
        }
    )
    base = request_fingerprint("POST", "/tickets", body)
    assert base == request_fingerprint("post", "/tickets", body)
    assert base != request_fingerprint("POST", "/tickets/INC-0001/comments", body)
    changed = body.model_copy(update={"priority": TicketPriority.HIGH})
    assert base != request_fingerprint("POST", "/tickets", changed)
    comment = AddCommentRequest(author_id="bob", body="b")
    expected = json.dumps({"author_id": "bob", "body": "b"}, sort_keys=True, separators=(",", ":"))
    assert request_fingerprint("POST", "/x", comment).endswith(
        hashlib.sha256(expected.encode()).hexdigest()
    )
