"""Chaos profiles: named in one place, and the moderate seed really does what it promises."""

import itertools

from evals.cases import load_cases
from evals.chaos_profiles import (
    ALL_5XX,
    CHAOS_OFF,
    MODERATE,
    ChaosProfile,
    enables_chaos,
    resolve_chaos,
)
from helpdesk.mock.chaos import ChaosEngine, ChaosOutcome
from helpdesk.mock.cursor import encode_cursor
from helpdesk.mock.search import SearchFilter
from helpdesk.shared.settings import Settings
from helpdesk.shared.tickets import ChaosConfig

# More mock requests than any case makes, retries included.
REQUESTS = 30


def _outcomes(config: ChaosConfig, n: int = REQUESTS) -> list[ChaosOutcome]:
    engine = ChaosEngine(Settings(_env_file=None))  # type: ignore[call-arg]
    engine.configure(config)
    return [engine.decide().outcome for _ in range(n)]


def test_resolve_named_profiles() -> None:
    assert resolve_chaos(ChaosProfile.OFF) == CHAOS_OFF
    assert resolve_chaos(ChaosProfile.MODERATE) == MODERATE
    assert resolve_chaos(ChaosProfile.ALL_5XX) == ALL_5XX


def test_resolve_passes_an_explicit_config_through() -> None:
    config = ChaosConfig(enabled=True, rate_limit_rate=1, error_rate=0, slow_rate=0, slow_ms=0)
    assert resolve_chaos(config) is config


def test_enables_chaos() -> None:
    assert not enables_chaos(None)
    assert not enables_chaos(ChaosProfile.OFF)
    assert enables_chaos(ChaosProfile.MODERATE)
    assert enables_chaos(ChaosProfile.ALL_5XX)


def test_moderate_rates_are_moderate_and_seeded() -> None:
    assert MODERATE.enabled
    assert 0 < MODERATE.rate_limit_rate + MODERATE.error_rate < 0.5
    assert MODERATE.seed is not None
    assert MODERATE.slow_ms < 5_000  # under the per-attempt timeout


def test_moderate_seed_fails_the_first_request_so_a_retry_is_always_seen() -> None:
    outcomes = _outcomes(MODERATE)
    assert outcomes[0] is not ChaosOutcome.PASS


def test_moderate_seed_never_fails_twice_within_three_requests() -> None:
    # So no tool call can fail all three attempts: the task can always complete.
    failures = [
        i for i, outcome in enumerate(_outcomes(MODERATE)) if outcome is not ChaosOutcome.PASS
    ]
    assert len(failures) >= 3  # it is still chaos
    assert all(b - a >= 3 for a, b in itertools.pairwise(failures))


def test_moderate_seed_shows_both_a_rate_limit_and_a_server_error_early() -> None:
    early = set(_outcomes(MODERATE, 6))
    assert ChaosOutcome.RATE_LIMITED in early
    assert early & {ChaosOutcome.SERVER_ERROR, ChaosOutcome.SERVICE_UNAVAILABLE}


def test_all_5xx_fails_every_request_with_a_server_error() -> None:
    outcomes = set(_outcomes(ALL_5XX))
    assert outcomes <= {ChaosOutcome.SERVER_ERROR, ChaosOutcome.SERVICE_UNAVAILABLE}


def test_the_pagination_case_script_carries_the_real_second_page_cursor() -> None:
    (case,) = [c for c in load_cases() if c.id == "pagination_cursor"]
    second_search = case.fake[0][1].tool_calls[0]
    assert second_search.args["cursor"] == encode_cursor(20, SearchFilter.build("network", []))
