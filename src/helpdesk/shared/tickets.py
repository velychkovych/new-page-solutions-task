"""Wire models for the mock ticketing REST API (CONTRACTS.md §1)."""

from enum import StrEnum
from typing import Annotated, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from helpdesk.shared.ids import TicketId, UserId
from helpdesk.shared.settings import Rate

IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
IDEMPOTENT_REPLAY_HEADER = "Idempotent-Replay"
ADMIN_TOKEN_HEADER = "X-Admin-Token"  # noqa: S105 - a header name, not a secret

SEARCH_LIMIT_MIN = 1
SEARCH_LIMIT_MAX = 20
SEARCH_LIMIT_DEFAULT = 5

Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
LongText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=5000)]


class TicketStatus(StrEnum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class TicketPriority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketCategory(StrEnum):
    NETWORK = "network"
    HARDWARE = "hardware"
    SOFTWARE = "software"
    ACCESS = "access"
    OTHER = "other"


class _Wire(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Comment(_Wire):
    id: str
    author_id: UserId
    body: str
    created_at: AwareDatetime


class Ticket(_Wire):
    id: TicketId
    title: Title
    description: LongText
    status: TicketStatus
    priority: TicketPriority
    category: TicketCategory
    reporter_id: UserId
    assignee_team: str | None
    created_at: AwareDatetime
    updated_at: AwareDatetime
    comments: list[Comment]


class TicketSummary(_Wire):
    id: TicketId
    title: Title
    status: TicketStatus
    priority: TicketPriority
    category: TicketCategory
    updated_at: AwareDatetime


class SearchResult(_Wire):
    items: list[TicketSummary]
    next_cursor: str | None


class CreateTicketRequest(_Wire):
    title: Title
    description: LongText
    category: TicketCategory
    priority: TicketPriority
    reporter_id: UserId


class AddCommentRequest(_Wire):
    author_id: UserId
    body: LongText


class CommentCreated(_Wire):
    ticket_id: TicketId
    comment: Comment


class ChaosConfig(_Wire):
    enabled: bool
    rate_limit_rate: Rate
    error_rate: Rate
    slow_rate: Rate
    slow_ms: Annotated[int, Field(ge=0, le=60_000)]
    # null = use CHAOS_SEED (default 0), so every run is seeded.
    seed: int | None = None
    fail_after_commit_rate: Rate = 0.0

    @model_validator(mode="after")
    def _failure_rates_fit_one_draw(self) -> Self:
        # One uniform draw decides 429 vs 5xx vs pass, so the two rates must fit in [0, 1].
        if self.rate_limit_rate + self.error_rate > 1.0:
            raise ValueError("rate_limit_rate + error_rate must be <= 1")
        return self


class WritesReport(_Wire):
    """Executed (not replayed) writes per idempotency key; every value should be 1."""

    total: int
    by_key: dict[str, int]


class ApiError(_Wire):
    error: str
    detail: str | None = None
