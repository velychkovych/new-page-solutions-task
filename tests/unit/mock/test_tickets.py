"""CONTRACTS §1.2, §1.3, §1.7: read, create and comment routes, and ApiError bodies."""

from typing import Any

import pytest

from helpdesk.shared.tickets import ApiError, CommentCreated, Ticket


async def test_get_ticket_with_comments(mock) -> None:
    response = await mock.client.get("/tickets/INC-0003", headers=mock.AUTH)
    assert response.status_code == 200
    ticket = Ticket.model_validate(response.json())
    assert ticket.reporter_id == "alice"
    assert [c.id for c in ticket.comments] == ["CMT-000003"]


@pytest.mark.parametrize("ticket_id", ["INC-9999", "INC-7", "inc-0003", "garbage"])
async def test_unknown_ticket_is_404(mock, ticket_id: str) -> None:
    response = await mock.client.get(f"/tickets/{ticket_id}", headers=mock.AUTH)
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


async def test_create_ticket(mock) -> None:
    response = await mock.create("k-create")
    assert response.status_code == 201
    assert "Idempotent-Replay" not in response.headers
    ticket = Ticket.model_validate(response.json())
    assert ticket.id == "INC-0016"  # sequential after the 15 seed tickets
    assert ticket.status == "open"
    assert ticket.assignee_team is None
    assert ticket.comments == []
    assert ticket.created_at == ticket.updated_at == mock.FIXED_NOW
    assert ticket.title == mock.NEW_TICKET["title"]
    fetched = await mock.client.get("/tickets/INC-0016", headers=mock.AUTH)
    assert fetched.json() == response.json()
    second = await mock.create("k-create-2")
    assert second.json()["id"] == "INC-0017"


async def test_create_strips_text(mock) -> None:
    body = {**mock.NEW_TICKET, "title": "  Padded title  "}
    response = await mock.create("k-strip", body)
    assert response.json()["title"] == "Padded title"


async def test_add_comment_updates_ticket(mock) -> None:
    mock.clock.advance(60)
    response = await mock.comment("INC-0003", "k-comment")
    assert response.status_code == 201
    created = CommentCreated.model_validate(response.json())
    assert created.ticket_id == "INC-0003"
    assert created.comment.author_id == "alice"
    assert created.comment.body == mock.NEW_COMMENT["body"]
    assert created.comment.created_at == mock.clock.now
    ticket = Ticket.model_validate(
        (await mock.client.get("/tickets/INC-0003", headers=mock.AUTH)).json()
    )
    assert ticket.updated_at == mock.clock.now
    assert ticket.comments[-1] == created.comment  # oldest first
    assert len(ticket.comments) == 2


async def test_comment_ids_are_unique(mock) -> None:
    ids = {(await mock.comment("INC-0001", f"k-{n}")).json()["comment"]["id"] for n in range(5)}
    seed_ids = {
        c["id"]
        for t in ("INC-0002", "INC-0003", "INC-0007", "INC-0008", "INC-0011")
        for c in (await mock.client.get(f"/tickets/{t}", headers=mock.AUTH)).json()["comments"]
    }
    assert len(ids) == 5
    assert not ids & seed_ids


async def test_comment_on_resolved_ticket_is_allowed(mock) -> None:
    response = await mock.comment("INC-0011", "k-resolved")
    assert response.status_code == 201
    ticket = (await mock.client.get("/tickets/INC-0011", headers=mock.AUTH)).json()
    assert ticket["status"] == "resolved"  # commenting does not reopen it


async def test_comment_on_unknown_ticket_is_404_and_not_stored(mock) -> None:
    response = await mock.comment("INC-9999", "k-404")
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"
    assert await mock.writes() == {"total": 0, "by_key": {}}
    # A 404 is not stored: the same key on a real ticket is evaluated fresh.
    retry = await mock.comment("INC-0003", "k-404")
    assert retry.status_code == 201
    assert "Idempotent-Replay" not in retry.headers


@pytest.mark.parametrize(
    "body",
    [
        {**{"title": "x"}, "unknown": 1},
        {"title": "t", "description": "d", "category": "hardware", "priority": "low"},
        {"title": "   ", "description": "d", "category": "hardware", "priority": "low",
         "reporter_id": "alice"},
        {"title": "t" * 201, "description": "d", "category": "hardware", "priority": "low",
         "reporter_id": "alice"},
        {"title": "t", "description": "d", "category": "printers", "priority": "low",
         "reporter_id": "alice"},
        {"title": "t", "description": "d", "category": "hardware", "priority": "low",
         "reporter_id": "alice:admin"},
        {"title": "t", "description": "d", "category": "hardware", "priority": "low",
         "reporter_id": "alice", "status": "closed"},
    ],
)  # fmt: skip
async def test_invalid_create_body_is_422_api_error(mock, body: dict[str, Any]) -> None:
    response = await mock.create("k-422", body)
    assert response.status_code == 422
    error = ApiError.model_validate(response.json())
    assert error.error == "invalid_request"
    assert error.detail


async def test_unknown_comment_field_is_422(mock) -> None:
    response = await mock.comment("INC-0003", "k-c422", {**mock.NEW_COMMENT, "author": "x"})
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"


async def test_non_json_body_is_422(mock) -> None:
    response = await mock.client.post(
        "/tickets",
        content=b"not json",
        headers={**mock.AUTH, "Idempotency-Key": "k", "Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"


async def test_validation_detail_does_not_echo_input(mock) -> None:
    body = {**mock.NEW_TICKET, "reporter_id": "secret value with spaces"}
    response = await mock.create("k-echo", body)
    assert response.status_code == 422
    assert "secret value with spaces" not in response.text
    assert "reporter_id" in response.json()["detail"]


async def test_router_errors_use_api_error(mock) -> None:
    not_found = await mock.client.get("/nope", headers=mock.AUTH)
    assert not_found.status_code == 404
    assert not_found.json() == {"error": "not_found"}
    wrong_method = await mock.client.delete("/tickets/INC-0003", headers=mock.AUTH)
    assert wrong_method.status_code == 405
    ApiError.model_validate(wrong_method.json())
