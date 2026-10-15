"""The MCP tool contract (CONTRACTS.md §2): names, arguments, injected arguments."""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from helpdesk.shared.ids import IdempotencyKey, TicketId, UserId
from helpdesk.shared.tickets import (
    SEARCH_LIMIT_DEFAULT,
    SEARCH_LIMIT_MAX,
    SEARCH_LIMIT_MIN,
    LongText,
    TicketCategory,
    TicketPriority,
    TicketStatus,
    Title,
)


class ToolName(StrEnum):
    SEARCH_TICKETS = "search_tickets"
    GET_TICKET = "get_ticket"
    CREATE_TICKET = "create_ticket"
    ADD_COMMENT = "add_comment"


READ_TOOLS: frozenset[ToolName] = frozenset({ToolName.SEARCH_TICKETS, ToolName.GET_TICKET})
WRITE_TOOLS: frozenset[ToolName] = frozenset({ToolName.CREATE_TICKET, ToolName.ADD_COMMENT})

# Present in the MCP schema, stripped from the schema the model sees, and filled in
# by the agent's execute_write node. The model never chooses whose ticket it writes.
INJECTED_ARGS: Mapping[ToolName, tuple[str, ...]] = MappingProxyType(
    {
        ToolName.CREATE_TICKET: ("reporter_id", "idempotency_key"),
        ToolName.ADD_COMMENT: ("author_id", "idempotency_key"),
    }
)

SearchQuery = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SearchTicketsArgs(_Args):
    query: SearchQuery
    status: list[TicketStatus] | None = None
    cursor: str | None = None
    page_size: Annotated[int, Field(ge=SEARCH_LIMIT_MIN, le=SEARCH_LIMIT_MAX)] = (
        SEARCH_LIMIT_DEFAULT
    )


class GetTicketArgs(_Args):
    ticket_id: TicketId


class CreateTicketArgs(_Args):
    """Model-visible arguments of create_ticket."""

    title: Title
    description: LongText
    category: TicketCategory
    priority: TicketPriority


class CreateTicketInput(CreateTicketArgs):
    """Full MCP input of create_ticket, including the injected arguments."""

    reporter_id: UserId
    idempotency_key: IdempotencyKey


class AddCommentArgs(_Args):
    """Model-visible arguments of add_comment."""

    ticket_id: TicketId
    body: LongText


class AddCommentInput(AddCommentArgs):
    """Full MCP input of add_comment, including the injected arguments."""

    author_id: UserId
    idempotency_key: IdempotencyKey
