"""CONTRACTS §1.1: bearer auth, admin token, /health open, tokens never echoed."""

import logging
from pathlib import Path

import pytest
from pydantic import SecretStr

from helpdesk.mock import __main__ as mock_main
from helpdesk.mock.app import create_app
from helpdesk.shared.settings import MissingSecretError, Settings

PROTECTED = [
    ("GET", "/tickets"),
    ("GET", "/tickets/INC-0003"),
    ("POST", "/tickets"),
    ("POST", "/tickets/INC-0003/comments"),
    ("POST", "/admin/chaos"),
    ("GET", "/admin/writes"),
    ("GET", "/no-such-route"),
]


async def test_health_needs_no_auth(mock) -> None:
    response = await mock.client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(("method", "path"), PROTECTED)
@pytest.mark.parametrize(
    "authorization",
    [None, "Bearer wrong-token", "Bearer ", "Bearer  {token}", "Basic {token}", "{token}"],
)
async def test_missing_or_wrong_bearer_is_401(
    mock, method: str, path: str, authorization: str | None
) -> None:
    headers = {"X-Admin-Token": mock.ADMIN_TOKEN, "Idempotency-Key": "k-1"}
    if authorization is not None:
        headers["Authorization"] = authorization.format(token=mock.API_TOKEN)
    response = await mock.client.request(method, path, headers=headers, json=mock.NEW_TICKET)
    assert response.status_code == 401
    assert response.json() == {"error": "unauthorized"}


async def test_bearer_scheme_is_case_insensitive(mock) -> None:
    response = await mock.client.get(
        "/tickets", headers={"Authorization": f"bearer {mock.API_TOKEN}"}
    )
    assert response.status_code == 200


async def test_unauthenticated_request_gets_401_even_under_full_chaos(mock) -> None:
    await mock.chaos(enabled=True, error_rate=1.0, slow_rate=1.0, slow_ms=1000)
    for _ in range(20):
        response = await mock.client.get("/tickets")
        assert response.status_code == 401
        assert response.json() == {"error": "unauthorized"}
    # Auth ran first, so chaos never drew a slow response for these requests.
    assert mock.sleep.calls == []


@pytest.mark.parametrize("admin_token", [None, "wrong", ""])
@pytest.mark.parametrize(("method", "path"), [("POST", "/admin/chaos"), ("GET", "/admin/writes")])
async def test_admin_routes_need_admin_token(
    mock, method: str, path: str, admin_token: str | None
) -> None:
    headers = dict(mock.AUTH)
    if admin_token is not None:
        headers["X-Admin-Token"] = admin_token
    body = {"enabled": False, "rate_limit_rate": 0, "error_rate": 0, "slow_rate": 0, "slow_ms": 0}
    response = await mock.client.request(method, path, headers=headers, json=body)
    assert response.status_code == 403
    assert response.json() == {"error": "forbidden"}


async def test_admin_token_alone_is_not_enough(mock) -> None:
    response = await mock.client.get("/admin/writes", headers={"X-Admin-Token": mock.ADMIN_TOKEN})
    assert response.status_code == 401


async def test_admin_token_does_not_open_ticket_routes(mock) -> None:
    response = await mock.client.get(
        "/tickets", headers={"Authorization": f"Bearer {mock.ADMIN_TOKEN}"}
    )
    assert response.status_code == 401


async def test_tokens_are_never_echoed_or_logged(mock, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    await mock.chaos(enabled=True, rate_limit_rate=0.3, error_rate=0.3)
    responses = [
        await mock.client.get("/tickets", headers=mock.AUTH),
        await mock.client.get("/tickets", headers={"Authorization": "Bearer nope"}),
        await mock.client.get("/admin/writes", headers={**mock.AUTH, "X-Admin-Token": "nope"}),
        await mock.client.post("/tickets", headers=mock.AUTH, json={"bad": "body"}),
        await mock.create("k-echo"),
        await mock.client.get("/admin/writes", headers=mock.ADMIN),
    ]
    for response in responses:
        for secret in (mock.API_TOKEN, mock.ADMIN_TOKEN):
            assert secret not in response.text
            assert all(secret not in value for value in response.headers.values())
    for secret in (mock.API_TOKEN, mock.ADMIN_TOKEN):
        assert secret not in caplog.text


@pytest.mark.parametrize("missing", ["mock_api_token", "mock_admin_token"])
def test_startup_refuses_without_tokens(tmp_path: Path, missing: str) -> None:
    tokens = {"mock_api_token": SecretStr("a"), "mock_admin_token": SecretStr("b")}
    for value in (None, SecretStr("")):
        settings = Settings(
            _env_file=None,
            mock_db_path=tmp_path / "m.sqlite",
            traces_dir=tmp_path / "traces",
            otel_exporter_otlp_endpoint=None,
            **{**tokens, missing: value},
        )
        with pytest.raises(MissingSecretError, match=missing.upper()):
            create_app(settings)
    assert not (tmp_path / "m.sqlite").exists()
    assert not (tmp_path / "traces").exists()


def test_entry_point_exits_without_tokens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Run from an empty dir so a developer's .env cannot supply the tokens.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MOCK_API_TOKEN", raising=False)
    monkeypatch.setenv("MOCK_ADMIN_TOKEN", "admin-secret-value")
    with pytest.raises(SystemExit) as exc_info:
        mock_main.main()
    message = str(exc_info.value.code)
    assert "MOCK_API_TOKEN is not set" in message
    assert "admin-secret-value" not in message
