"""Chaos injection for the mock (CONTRACTS §1.6; PLAN §8 row 5).

`ChaosMiddleware` sits inside `AuthMiddleware`, so an unauthenticated request always
gets 401, and outside the route handler, so a request failed by chaos never reaches it and
a write failed this way is never persisted. Fail-after-commit is the one decision made
after the handler: the write route asks `ChaosEngine.fail_after_commit()` once its
transaction has committed.
"""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum, auto
from http import HTTPStatus

from starlette.types import ASGIApp, Receive, Scope, Send

from helpdesk.mock.auth import is_admin, is_public
from helpdesk.mock.errors import ApiErrorCode, error_response
from helpdesk.shared.settings import Settings
from helpdesk.shared.tickets import ChaosConfig

logger = logging.getLogger(__name__)

type Sleep = Callable[[float], Awaitable[None]]

_UPSTREAM_ERROR_STATUSES = (HTTPStatus.INTERNAL_SERVER_ERROR, HTTPStatus.SERVICE_UNAVAILABLE)


class ChaosOutcome(Enum):
    PASS = auto()
    RATE_LIMITED = auto()
    SERVER_ERROR = auto()
    SERVICE_UNAVAILABLE = auto()


@dataclass(frozen=True, slots=True)
class ChaosDecision:
    slow: bool
    outcome: ChaosOutcome


def config_from_settings(settings: Settings) -> ChaosConfig:
    return ChaosConfig(
        enabled=settings.chaos_enabled,
        rate_limit_rate=settings.chaos_rate_limit_rate,
        error_rate=settings.chaos_error_rate,
        slow_rate=settings.chaos_slow_rate,
        slow_ms=settings.chaos_slow_ms,
        seed=None,
        fail_after_commit_rate=settings.chaos_fail_after_commit_rate,
    )


def is_chaos_exempt(path: str) -> bool:
    return is_public(path) or is_admin(path)


class ChaosEngine:
    """Owns the single RNG behind every chaos choice, so a seed fixes the whole sequence."""

    def __init__(self, settings: Settings) -> None:
        self._default_seed = settings.chaos_seed
        self.retry_after_s = settings.chaos_retry_after_s
        self.configure(config_from_settings(settings))

    @property
    def config(self) -> ChaosConfig:
        return self._config

    def configure(self, config: ChaosConfig) -> None:
        """Replace the whole config and re-create the RNG, so a run restarts its sequence."""
        self._config = config
        seed = self._default_seed if config.seed is None else config.seed
        self._rng = random.Random(seed)  # noqa: S311 - reproducible fault injection, not crypto

    def decide(self) -> ChaosDecision:
        """Draw in the contract's fixed order: slow, then one uniform draw for 429/5xx."""
        config = self._config
        if not config.enabled:
            return ChaosDecision(slow=False, outcome=ChaosOutcome.PASS)
        slow = self._rng.random() < config.slow_rate
        u = self._rng.random()
        if u < config.rate_limit_rate:
            outcome = ChaosOutcome.RATE_LIMITED
        elif u < config.rate_limit_rate + config.error_rate:
            status = self._rng.choice(_UPSTREAM_ERROR_STATUSES)
            outcome = (
                ChaosOutcome.SERVER_ERROR
                if status is HTTPStatus.INTERNAL_SERVER_ERROR
                else ChaosOutcome.SERVICE_UNAVAILABLE
            )
        else:
            outcome = ChaosOutcome.PASS
        return ChaosDecision(slow=slow, outcome=outcome)

    def fail_after_commit(self) -> bool:
        """Called only after an executed (not replayed) write has committed."""
        config = self._config
        if not config.enabled:
            return False
        return self._rng.random() < config.fail_after_commit_rate


class ChaosMiddleware:
    def __init__(self, app: ASGIApp, *, engine: ChaosEngine, sleep: Sleep = asyncio.sleep) -> None:
        self.app = app
        self.engine = engine
        # Injectable so tests exercise slow responses without really waiting.
        self.sleep = sleep

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or is_chaos_exempt(scope["path"]):
            await self.app(scope, receive, send)
            return
        decision = self.engine.decide()
        if decision.slow:
            await self.sleep(self.engine.config.slow_ms / 1000)
        match decision.outcome:
            case ChaosOutcome.PASS:
                await self.app(scope, receive, send)
                return
            case ChaosOutcome.RATE_LIMITED:
                response = error_response(
                    HTTPStatus.TOO_MANY_REQUESTS,
                    ApiErrorCode.RATE_LIMITED,
                    headers={"Retry-After": str(self.engine.retry_after_s)},
                )
            case ChaosOutcome.SERVER_ERROR:
                response = error_response(
                    HTTPStatus.INTERNAL_SERVER_ERROR, ApiErrorCode.UPSTREAM_ERROR
                )
            case ChaosOutcome.SERVICE_UNAVAILABLE:
                response = error_response(
                    HTTPStatus.SERVICE_UNAVAILABLE, ApiErrorCode.UPSTREAM_ERROR
                )
        logger.info("chaos: %s %s -> %d", scope["method"], scope["path"], response.status_code)
        await response(scope, receive, send)
