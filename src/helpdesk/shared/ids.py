"""Identifier formats shared by the mock, the MCP server and the agent."""

import re
from typing import Annotated

from pydantic import StringConstraints

# [0-9] rather than \d: \d also matches non-ASCII digits (e.g. Arabic-Indic).
TICKET_ID_PATTERN = r"^INC-[0-9]{4}$"
PRINCIPAL_ID_PATTERN = r"^[a-zA-Z0-9_-]{1,64}$"
# Printable ASCII without spaces, so a key survives an HTTP header unchanged.
IDEMPOTENCY_KEY_PATTERN = r"^[\x21-\x7e]{1,255}$"

TicketId = Annotated[str, StringConstraints(pattern=TICKET_ID_PATTERN)]
UserId = Annotated[str, StringConstraints(pattern=PRINCIPAL_ID_PATTERN)]
SessionId = Annotated[str, StringConstraints(pattern=PRINCIPAL_ID_PATTERN)]
IdempotencyKey = Annotated[str, StringConstraints(pattern=IDEMPOTENCY_KEY_PATTERN)]

_TICKET_ID_RE = re.compile(TICKET_ID_PATTERN)
_PRINCIPAL_ID_RE = re.compile(PRINCIPAL_ID_PATTERN)


def is_ticket_id(value: str) -> bool:
    # fullmatch, because `$` alone also matches before a trailing newline.
    return _TICKET_ID_RE.fullmatch(value) is not None


def is_principal_id(value: str) -> bool:
    return _PRINCIPAL_ID_RE.fullmatch(value) is not None


def thread_id(user_id: str, session_id: str) -> str:
    """Checkpointer thread id. Neither part may contain ':', so ids cannot collide."""
    for name, value in (("user_id", user_id), ("session_id", session_id)):
        if not is_principal_id(value):
            raise ValueError(f"{name} must match {PRINCIPAL_ID_PATTERN}")
    return f"{user_id}:{session_id}"
