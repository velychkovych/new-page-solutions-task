"""CONTRACTS §1.4: search matching, status filter, limits and cursor pagination."""

import base64
import json
from typing import Any

import pytest

from helpdesk.mock.cursor import decode_cursor, encode_cursor
from helpdesk.mock.errors import InvalidCursorError
from helpdesk.mock.search import SearchFilter
from helpdesk.mock.seed_data import SEED_TICKETS
from helpdesk.shared.tickets import TicketStatus


async def _get(mock, **params: Any) -> Any:
    response = await mock.client.get("/tickets", params=params, headers=mock.AUTH)
    assert response.status_code == 200, response.text
    return response.json()


async def _walk(mock, **params: Any) -> tuple[list[str], list[int]]:
    """Follow next_cursor to the end; return every id and each page's size."""
    page = await _get(mock, **params)
    ids, sizes = [i["id"] for i in page["items"]], [len(page["items"])]
    while page["next_cursor"] is not None:
        page = await _get(mock, **{**params, "cursor": page["next_cursor"]})
        ids += [i["id"] for i in page["items"]]
        sizes.append(len(page["items"]))
    return ids, sizes


def _expected(q: str | None = None, statuses: tuple[str, ...] = ()) -> list[str]:
    search = SearchFilter.build(q, [TicketStatus(s) for s in statuses])
    return [
        t.id
        for t in SEED_TICKETS
        if (not statuses or t.status in search.statuses) and search.matches(t.title, t.description)
    ]


@pytest.mark.parametrize("limit", [1, 2, 3, 5, 7, 15, 20])
@pytest.mark.parametrize(
    ("q", "statuses"),
    [(None, ()), ("network", ()), ("the", ("open",)), ("", ("open", "resolved")), ("vpn", ())],
)
async def test_pagination_covers_everything_once(
    mock, limit: int, q: str | None, statuses: tuple[str, ...]
) -> None:
    params: dict[str, Any] = {"limit": limit, "status": list(statuses)}
    if q is not None:
        params["q"] = q
    ids, sizes = await _walk(mock, **params)
    expected = _expected(q, statuses)
    assert ids == expected  # every match, exactly once, in id order
    assert len(set(ids)) == len(ids)
    # next_cursor is null exactly at the end: no trailing empty page, no early stop.
    assert all(size == limit for size in sizes[:-1])
    assert 0 < sizes[-1] <= limit or expected == []


async def test_default_limit_is_five_and_network_needs_page_two(mock) -> None:
    first = await _get(mock, q="network")
    assert len(first["items"]) == 5
    assert first["next_cursor"] is not None
    second = await _get(mock, q="network", cursor=first["next_cursor"])
    assert [i["id"] for i in second["items"]] == ["INC-0010", "INC-0014", "INC-0015"]
    assert second["next_cursor"] is None


async def test_exact_multiple_of_limit_has_no_empty_last_page(mock) -> None:
    ids, sizes = await _walk(mock, limit=5)
    assert len(ids) == 15
    assert sizes == [5, 5, 5]


async def test_summary_shape(mock) -> None:
    page = await _get(mock, q="vpn")
    assert page == {
        "items": [
            {
                "id": "INC-0003",
                "title": "VPN keeps dropping",
                "status": "open",
                "priority": "high",
                "category": "network",
                "updated_at": "2026-09-04T08:00:00Z",
            }
        ],
        "next_cursor": None,
    }


@pytest.mark.parametrize(
    ("q", "expected"),
    [
        ("VPN DROPPING", ["INC-0003"]),
        ("vpn   keeps\tdropping", ["INC-0003"]),
        ("vpn printer", []),  # AND, not OR
        ("ignore instructions", ["INC-0013"]),
        ("dropping stable", ["INC-0003"]),  # one term in the title, one in the description
        ("   ", [t.id for t in SEED_TICKETS]),  # blank = everything
        ("battery", ["INC-0007"]),
        ("PRINTER", ["INC-0011"]),
        ("no-such-word", []),
    ],
)
async def test_terms_are_anded_and_case_insensitive(mock, q: str, expected: list[str]) -> None:
    ids, _ = await _walk(mock, q=q, limit=20)
    assert ids == expected


async def test_status_is_repeatable_and_ored(mock) -> None:
    open_ids, _ = await _walk(mock, status="open", limit=20)
    in_progress_ids, _ = await _walk(mock, status="in_progress", limit=20)
    both, _ = await _walk(mock, status=["open", "in_progress"], limit=20)
    assert both == sorted(open_ids + in_progress_ids)
    assert in_progress_ids == ["INC-0005", "INC-0007"]
    page = await _get(mock, status=["resolved", "resolved"], limit=20)
    assert [i["status"] for i in page["items"]] == ["resolved"] * 3


async def test_status_and_query_combine(mock) -> None:
    ids, _ = await _walk(mock, q="network", status="resolved")
    assert ids == ["INC-0008"]


@pytest.mark.parametrize(
    "params",
    [
        {"status": "pending"},
        {"status": ["open", "OPEN"]},
        {"limit": 0},
        {"limit": 21},
        {"limit": -1},
        {"limit": "five"},
        {"q": "x" * 201},
    ],
)
async def test_invalid_query_params_are_422_api_error(mock, params: dict[str, Any]) -> None:
    response = await mock.client.get("/tickets", params=params, headers=mock.AUTH)
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "invalid_request"
    assert set(body) == {"error", "detail"}
    assert isinstance(body["detail"], str)
    assert body["detail"]


async def test_q_of_200_chars_is_allowed(mock) -> None:
    page = await _get(mock, q="x" * 200)
    assert page == {"items": [], "next_cursor": None}


async def test_cursor_is_bound_to_q_and_status(mock) -> None:
    cursor = (await _get(mock, q="network", limit=2))["next_cursor"]
    assert (await _get(mock, q="network", limit=2, cursor=cursor))["items"]
    # Changing only the limit keeps the cursor valid: limit is not part of the search.
    assert (await _get(mock, q="network", limit=3, cursor=cursor))["items"]
    for params in (
        {"q": "vpn"},
        {},
        {"q": "network", "status": "open"},
        {"q": "network wifi"},
    ):
        response = await mock.client.get(
            "/tickets", params={**params, "limit": 2, "cursor": cursor}, headers=mock.AUTH
        )
        assert response.status_code == 400, params
        assert response.json()["error"] == "invalid_cursor"


async def test_cursor_survives_param_reordering_and_case(mock) -> None:
    cursor = (await _get(mock, q="Network", status=["open", "resolved"], limit=2))["next_cursor"]
    page = await _get(mock, q="network", status=["resolved", "open"], limit=2, cursor=cursor)
    assert page["items"]


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


@pytest.mark.parametrize(
    "cursor",
    [
        "",
        "not-a-cursor",
        "%%%",
        _b64(b"not json"),
        _b64(b"[]"),
        _b64(json.dumps({"o": -1, "f": "x"}).encode()),
        _b64(json.dumps({"o": "5", "f": 3, "extra": 1}).encode()),
        "é",
    ],
)
async def test_malformed_cursor_is_400(mock, cursor: str) -> None:
    response = await mock.client.get("/tickets", params={"cursor": cursor}, headers=mock.AUTH)
    assert response.status_code == 400
    assert response.json() == {"error": "invalid_cursor", "detail": "the cursor is malformed"}


async def test_new_tickets_land_on_the_last_page(mock) -> None:
    first = await _get(mock, limit=5)
    created = await mock.create("k-new")
    assert created.status_code == 201
    ids, _ = await _walk(mock, limit=5, cursor=first["next_cursor"])
    assert ids[-1] == created.json()["id"]
    assert ids[0] == "INC-0006"  # earlier pages did not shift


def test_cursor_round_trip() -> None:
    search = SearchFilter.build("a b", [TicketStatus.OPEN])
    assert decode_cursor(encode_cursor(10, search), search) == 10
    with pytest.raises(InvalidCursorError):
        decode_cursor(encode_cursor(10, search), SearchFilter.build("a", [TicketStatus.OPEN]))


def test_search_filter_normalises() -> None:
    a = SearchFilter.build("  VPN  Drops ", [TicketStatus.OPEN, TicketStatus.CLOSED])
    b = SearchFilter.build("vpn drops", [TicketStatus.CLOSED, TicketStatus.OPEN, TicketStatus.OPEN])
    assert a == b
    assert a.terms == ("vpn", "drops")
    assert SearchFilter.build("STRASSE", []).matches("Straße", "")
