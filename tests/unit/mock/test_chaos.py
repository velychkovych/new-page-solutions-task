"""CONTRACTS §1.6: seeded chaos middleware and fail-after-commit (PLAN §8 row 5)."""

import random
from collections import Counter
from http import HTTPStatus
from typing import Any

import pytest

from helpdesk.mock.chaos import ChaosEngine, ChaosOutcome
from helpdesk.shared.settings import Settings
from helpdesk.shared.tickets import ChaosConfig

RATES: dict[str, Any] = {
    "enabled": True,
    "rate_limit_rate": 0.2,
    "error_rate": 0.3,
    "slow_rate": 0.1,
    "slow_ms": 1500,
}
CALLS = 1_000
TOLERANCE = 0.05


def _engine(chaos_seed: int = 0, **config: Any) -> ChaosEngine:
    engine = ChaosEngine(Settings(_env_file=None, chaos_seed=chaos_seed))
    engine.configure(ChaosConfig.model_validate({**RATES, **config}))
    return engine


def test_engine_rates_are_roughly_respected() -> None:
    engine = _engine(seed=1234)
    decisions = [engine.decide() for _ in range(CALLS)]
    outcomes = Counter(d.outcome for d in decisions)
    assert abs(outcomes[ChaosOutcome.RATE_LIMITED] / CALLS - 0.2) < TOLERANCE
    errors = outcomes[ChaosOutcome.SERVER_ERROR] + outcomes[ChaosOutcome.SERVICE_UNAVAILABLE]
    assert abs(errors / CALLS - 0.3) < TOLERANCE
    # 500 vs 503 is a fair coin among the errors.
    assert abs(outcomes[ChaosOutcome.SERVER_ERROR] / errors - 0.5) < 0.1
    assert abs(sum(d.slow for d in decisions) / CALLS - 0.1) < TOLERANCE


def test_draw_order_matches_the_contract() -> None:
    """§1.6: slow first, then one uniform `u`, then the 500/503 choice only on an error.
    Comparing against a hand-driven `random.Random` pins the order, not just the rates."""
    rates = {"rate_limit_rate": 0.3, "error_rate": 0.3, "slow_rate": 0.5}
    engine = _engine(seed=2024, **rates)
    rng = random.Random(2024)  # noqa: S311
    expected = []
    for _ in range(500):
        slow = rng.random() < rates["slow_rate"]
        u = rng.random()
        if u < rates["rate_limit_rate"]:
            outcome = ChaosOutcome.RATE_LIMITED
        elif u < rates["rate_limit_rate"] + rates["error_rate"]:
            status = rng.choice((HTTPStatus.INTERNAL_SERVER_ERROR, HTTPStatus.SERVICE_UNAVAILABLE))
            outcome = (
                ChaosOutcome.SERVER_ERROR
                if status is HTTPStatus.INTERNAL_SERVER_ERROR
                else ChaosOutcome.SERVICE_UNAVAILABLE
            )
        else:
            outcome = ChaosOutcome.PASS
        expected.append((slow, outcome))
    assert [(d.slow, d.outcome) for d in (engine.decide() for _ in range(500))] == expected


def test_same_seed_same_sequence_and_reconfigure_restarts_it() -> None:
    a, b = _engine(chaos_seed=7), _engine(seed=7)
    first = [a.decide() for _ in range(200)]
    assert first == [b.decide() for _ in range(200)]
    assert first != [_engine(seed=8).decide() for _ in range(200)]
    a.configure(a.config)  # POST /admin/chaos re-creates the RNG
    assert [a.decide() for _ in range(200)] == first


def test_null_seed_falls_back_to_chaos_seed() -> None:
    from_env = _engine(chaos_seed=99)  # config seed null -> CHAOS_SEED
    explicit = _engine(chaos_seed=0, seed=99)
    assert [from_env.decide() for _ in range(100)] == [explicit.decide() for _ in range(100)]


def test_disabled_engine_never_fails() -> None:
    engine = _engine(enabled=False, rate_limit_rate=1.0, error_rate=0.0, slow_rate=1.0)
    engine.configure(engine.config.model_copy(update={"fail_after_commit_rate": 1.0}))
    assert all(engine.decide().outcome is ChaosOutcome.PASS for _ in range(50))
    assert not any(engine.decide().slow for _ in range(50))
    assert not engine.fail_after_commit()


async def _statuses(mock, calls: int) -> list[int]:
    return [
        (await mock.client.get("/tickets/INC-0001", headers=mock.AUTH)).status_code
        for _ in range(calls)
    ]


async def test_http_chaos_rates_over_1000_calls(mock) -> None:
    response = await mock.chaos(**{**RATES, "seed": 42})
    assert response.status_code == 200
    statuses = Counter(await _statuses(mock, CALLS))
    assert set(statuses) <= {200, 429, 500, 503}
    assert abs(statuses[429] / CALLS - 0.2) < TOLERANCE
    assert abs((statuses[500] + statuses[503]) / CALLS - 0.3) < TOLERANCE
    assert abs(statuses[200] / CALLS - 0.5) < TOLERANCE
    assert abs(len(mock.sleep.calls) / CALLS - 0.1) < TOLERANCE
    assert set(mock.sleep.calls) == {1.5}


async def test_http_chaos_is_reproducible_for_a_seed(mock_factory) -> None:
    runs = []
    for _ in range(2):
        async with mock_factory() as mock:
            await mock.chaos(**{**RATES, "seed": 5})
            runs.append((await _statuses(mock, 200), len(mock.sleep.calls)))
    assert runs[0] == runs[1]
    async with mock_factory() as mock:
        await mock.chaos(**{**RATES, "seed": 6})
        assert (await _statuses(mock, 200)) != runs[0][0]


async def test_chaos_from_env_at_startup_is_seeded(mock_factory) -> None:
    env = {
        "chaos_enabled": True,
        "chaos_rate_limit_rate": 0.2,
        "chaos_error_rate": 0.3,
        "chaos_seed": 11,
    }
    runs = []
    for _ in range(2):
        async with mock_factory(**env) as mock:
            runs.append(await _statuses(mock, 100))
    assert runs[0] == runs[1]
    assert set(runs[0]) > {200}


async def test_429_and_5xx_bodies(mock_factory) -> None:
    async with mock_factory(chaos_retry_after_s=3) as mock:
        await mock.chaos(enabled=True, rate_limit_rate=1.0)
        limited = await mock.client.get("/tickets", headers=mock.AUTH)
        assert limited.status_code == 429
        assert limited.json() == {"error": "rate_limited"}
        assert limited.headers["Retry-After"] == "3"
        await mock.chaos(enabled=True, error_rate=1.0)
        failed = [await mock.client.get("/tickets", headers=mock.AUTH) for _ in range(40)]
        assert {r.status_code for r in failed} == {500, 503}
        assert all(r.json() == {"error": "upstream_error"} for r in failed)


async def test_slow_response_sleeps_then_succeeds(mock) -> None:
    await mock.chaos(enabled=True, slow_rate=1.0, slow_ms=2500)
    response = await mock.client.get("/tickets/INC-0003", headers=mock.AUTH)
    assert response.status_code == 200
    assert mock.sleep.calls == [2.5]


async def test_health_and_admin_are_exempt(mock) -> None:
    await mock.chaos(enabled=True, error_rate=1.0, slow_rate=1.0, slow_ms=1000)
    for _ in range(20):
        assert (await mock.client.get("/health")).status_code == 200
        assert (await mock.client.get("/admin/writes", headers=mock.ADMIN)).status_code == 200
    assert mock.sleep.calls == []
    # chaos-off still works while chaos fails every ticket request.
    assert (await mock.chaos()).status_code == 200
    assert (await mock.client.get("/tickets", headers=mock.AUTH)).status_code == 200


async def test_disabled_chaos_does_nothing(mock) -> None:
    await mock.chaos(enabled=False, rate_limit_rate=1.0, slow_rate=1.0, slow_ms=1000)
    assert set(await _statuses(mock, 20)) == {200}
    assert mock.sleep.calls == []


async def test_write_failed_by_chaos_is_not_persisted(mock) -> None:
    await mock.chaos(enabled=True, error_rate=1.0)
    failed = await mock.create("k-chaos")
    assert failed.status_code in {500, 503}
    comment = await mock.comment("INC-0003", "k-chaos-c")
    assert comment.status_code in {500, 503}
    await mock.chaos()
    assert len(await mock.all_ids()) == 15
    assert await mock.writes() == {"total": 0, "by_key": {}}
    ticket = (await mock.client.get("/tickets/INC-0003", headers=mock.AUTH)).json()
    assert len(ticket["comments"]) == 1
    # The key was never stored, so a retry executes fresh.
    retry = await mock.create("k-chaos")
    assert retry.status_code == 201
    assert "Idempotent-Replay" not in retry.headers


async def test_fail_after_commit_then_replay_returns_original_201(mock) -> None:
    await mock.chaos(enabled=True, fail_after_commit_rate=1.0)
    lost = await mock.create("k-lost")
    assert lost.status_code == 503
    assert lost.json() == {"error": "upstream_error"}
    # The write committed even though the response was lost.
    stored = await mock.client.get("/tickets/INC-0016", headers=mock.AUTH)
    assert stored.status_code == 200
    # The rate is still 1.0, but a replay is not an executed write, so it is not failed.
    retry = await mock.create("k-lost")
    assert retry.status_code == 201
    assert retry.headers["Idempotent-Replay"] == "true"
    assert retry.json() == stored.json()
    assert await mock.writes() == {"total": 1, "by_key": {"k-lost": 1}}


async def test_fail_after_commit_on_comments(mock) -> None:
    await mock.chaos(enabled=True, fail_after_commit_rate=1.0)
    assert (await mock.comment("INC-0003", "k-lc")).status_code == 503
    retry = await mock.comment("INC-0003", "k-lc")
    assert retry.status_code == 201
    assert retry.headers["Idempotent-Replay"] == "true"
    ticket = (await mock.client.get("/tickets/INC-0003", headers=mock.AUTH)).json()
    assert len(ticket["comments"]) == 2


async def test_fail_after_commit_needs_chaos_enabled(mock_factory) -> None:
    async with mock_factory(chaos_fail_after_commit_rate=1.0) as mock:
        assert (await mock.create("k-off")).status_code == 201


async def test_fail_after_commit_from_env(mock_factory) -> None:
    async with mock_factory(chaos_enabled=True, chaos_fail_after_commit_rate=1.0) as mock:
        assert (await mock.create("k-env")).status_code == 503
        assert (await mock.create("k-env")).status_code == 201


async def test_admin_chaos_returns_active_config(mock) -> None:
    response = await mock.chaos(**{**RATES, "seed": 3, "fail_after_commit_rate": 0.5})
    assert response.json() == {**RATES, "seed": 3, "fail_after_commit_rate": 0.5}
    defaults = await mock.chaos(enabled=True)
    assert defaults.json()["seed"] is None
    assert defaults.json()["fail_after_commit_rate"] == 0.0


@pytest.mark.parametrize(
    "config",
    [
        {"rate_limit_rate": 0.6, "error_rate": 0.5},
        {"slow_rate": 1.5},
        {"slow_ms": 60_001},
        {"error_rate": -0.1},
        {"fail_after_commit_rate": 2},
        {"unknown": True},
    ],
)
async def test_admin_chaos_rejects_invalid_config(mock, config: dict[str, Any]) -> None:
    response = await mock.chaos(enabled=True, **config)
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"


async def test_admin_chaos_rejects_missing_fields(mock) -> None:
    response = await mock.client.post("/admin/chaos", json={"enabled": True}, headers=mock.ADMIN)
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"
