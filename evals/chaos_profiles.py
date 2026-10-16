"""The named chaos profiles a case can ask for. They are defined here and nowhere else."""

from enum import StrEnum

from helpdesk.shared.tickets import ChaosConfig


class ChaosProfile(StrEnum):
    OFF = "off"
    MODERATE = "moderate"
    # PLAN calls this case "100% 503". The mock answers a chaos error with 500 or 503 at
    # random (CONTRACTS §1.6), so the honest name is "every request fails with a 5xx".
    ALL_5XX = "all_5xx"


type ChaosSetting = ChaosProfile | ChaosConfig

CHAOS_OFF = ChaosConfig(enabled=False, rate_limit_rate=0, error_rate=0, slow_rate=0, slow_ms=0)

# Seed 5177 makes the mock's draw sequence fail the 1st request (503) and the 4th (429),
# and keeps any two failures at least 3 requests apart for the first 30 requests
# (tests/unit/evals/test_chaos_profiles.py). So the first tool call always shows a retry,
# and no call can fail three times in a row: the task can complete with rates this high.
MODERATE_SEED = 5177
MODERATE = ChaosConfig(
    enabled=True,
    rate_limit_rate=0.15,
    error_rate=0.25,
    slow_rate=0.1,
    # Well under the 5 s per-attempt timeout, so a slow response is never a timeout.
    slow_ms=1000,
    seed=MODERATE_SEED,
)

ALL_5XX = ChaosConfig(
    enabled=True, rate_limit_rate=0, error_rate=1.0, slow_rate=0, slow_ms=0, seed=0
)

PROFILES: dict[ChaosProfile, ChaosConfig] = {
    ChaosProfile.OFF: CHAOS_OFF,
    ChaosProfile.MODERATE: MODERATE,
    ChaosProfile.ALL_5XX: ALL_5XX,
}


def resolve_chaos(setting: ChaosSetting) -> ChaosConfig:
    return PROFILES[setting] if isinstance(setting, ChaosProfile) else setting


def enables_chaos(setting: ChaosSetting | None) -> bool:
    return setting is not None and resolve_chaos(setting).enabled
