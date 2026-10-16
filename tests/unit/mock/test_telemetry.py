"""CONTRACTS §6 for the mock: telemetry is set up when the app starts, server spans land in
traces/mock.jsonl, a caller's traceparent is joined, and no token ever reaches a span."""

import json
import re
from pathlib import Path
from typing import Any

import pytest

TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
SPAN_ID_RE = re.compile(r"^[0-9a-f]{16}$")
CALLER_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
CALLER_SPAN_ID = "00f067aa0ba902b7"
# Wrong credentials are not configured secrets, so only the header rule keeps them out.
WRONG_BEARER = "wrong-bearer-value-3e9a17"
WRONG_ADMIN = "wrong-admin-value-8d2f40"


def _jsonl(mock: Any) -> Path:
    return mock.settings.traces_dir / "mock.jsonl"


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _status_code(record: dict[str, Any]) -> int | None:
    attributes = record["attributes"]
    value = attributes.get("http.response.status_code", attributes.get("http.status_code"))
    return None if value is None else int(value)


async def test_a_request_writes_a_server_span_for_service_mock(mock_factory: Any) -> None:
    async with mock_factory() as mock:
        response = await mock.client.get("/tickets/INC-0003", headers=mock.AUTH)
        assert response.status_code == 200
    # Read after the lifespan exits: its shutdown flushes the batch processor.
    (server,) = _records(_jsonl(mock))
    assert server["name"] == "GET /tickets/{ticket_id}"
    assert server["service"] == "mock"
    assert TRACE_ID_RE.fullmatch(server["trace_id"])
    assert server["trace_id"] != "0" * 32
    assert SPAN_ID_RE.fullmatch(server["span_id"])
    assert server["parent_span_id"] is None
    assert server["status"] == "ok"
    assert _status_code(server) == 200


async def test_an_incoming_traceparent_is_joined(mock_factory: Any) -> None:
    async with mock_factory() as mock:
        response = await mock.client.get(
            "/tickets",
            params={"q": "vpn"},
            headers={**mock.AUTH, "traceparent": f"00-{CALLER_TRACE_ID}-{CALLER_SPAN_ID}-01"},
        )
        assert response.status_code == 200
    (server,) = _records(_jsonl(mock))
    assert server["name"] == "GET /tickets"
    assert server["trace_id"] == CALLER_TRACE_ID
    assert server["parent_span_id"] == CALLER_SPAN_ID


async def test_health_is_not_traced(mock_factory: Any) -> None:
    async with mock_factory() as mock:
        for _ in range(3):
            assert (await mock.client.get("/health")).status_code == 200
        # One traced request, so an empty result cannot come from a missing setup.
        assert (await mock.client.get("/tickets", headers=mock.AUTH)).status_code == 200
    assert [r["name"] for r in _records(_jsonl(mock))] == ["GET /tickets"]


async def test_no_token_or_credential_header_reaches_the_traces(
    mock_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Header capture switched on for everything, as a misconfigured deployment could do:
    # the exporter, not the instrumentation default, must keep the credentials out.
    monkeypatch.setenv("OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER_REQUEST", ".*")
    monkeypatch.setenv("OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER_RESPONSE", ".*")
    async with mock_factory() as mock:
        client, auth, admin = mock.client, mock.AUTH, mock.ADMIN
        statuses = [
            (await client.get("/tickets")).status_code,
            (await client.get("/tickets", headers={"Authorization": WRONG_BEARER})).status_code,
            (
                await client.get(
                    "/admin/writes", headers={"Authorization": f"Bearer {mock.ADMIN_TOKEN}"}
                )
            ).status_code,
            (
                await client.get("/admin/writes", headers={**auth, "X-Admin-Token": WRONG_ADMIN})
            ).status_code,
            (
                await client.get("/admin/writes", headers={**auth, "X-Admin-Token": mock.API_TOKEN})
            ).status_code,
            # A token in the URL is scrubbed as a value, wherever it lands.
            (await client.get("/tickets", params={"q": mock.API_TOKEN}, headers=auth)).status_code,
            (await client.get("/admin/writes", headers=admin)).status_code,
        ]
        assert statuses == [401, 401, 401, 403, 403, 200, 200]

        assert (await mock.chaos(enabled=True, rate_limit_rate=1)).status_code == 200
        assert (await client.get("/tickets", headers=auth)).status_code == 429
        assert (await mock.chaos(enabled=True, error_rate=1)).status_code == 200
        assert (await client.get("/tickets", headers=auth)).status_code in {500, 503}
        assert (await mock.chaos(enabled=True, fail_after_commit_rate=1)).status_code == 200
        assert (await mock.create("trace-key-1")).status_code == 503

    raw = _jsonl(mock).read_text(encoding="utf-8")
    for value in (mock.API_TOKEN, mock.ADMIN_TOKEN, WRONG_BEARER, WRONG_ADMIN):
        assert value not in raw
    assert "authorization" not in raw.lower()
    assert "admin_token" not in raw.lower()

    records = _records(_jsonl(mock))
    traced = sorted(code for r in records if (code := _status_code(r)) is not None)
    assert {401, 403, 429} <= set(traced)
    # The chaos 5xx and the fail-after-commit 503.
    assert len([c for c in traced if c >= 500]) == 2
    # The capture really was on: a harmless header made it into the spans.
    assert any(key.endswith("idempotency_key") for r in records for key in r["attributes"]), (
        "header capture was off, so the test proves nothing"
    )
