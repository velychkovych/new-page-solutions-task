"""Opaque pagination cursors: base64url JSON of the offset and a `(q, status)` fingerprint.

The fingerprint stops a cursor issued for one search from silently paging through another
(CONTRACTS §1.4). The cursor is not signed: it only has to be opaque, not tamper-proof.
"""

import base64
import hashlib
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from helpdesk.mock.errors import InvalidCursorError
from helpdesk.mock.search import SearchFilter


class _CursorPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    o: Annotated[int, Field(ge=0)]
    f: str


def filter_fingerprint(search: SearchFilter) -> str:
    return hashlib.sha256(search.model_dump_json().encode()).hexdigest()[:16]


def encode_cursor(offset: int, search: SearchFilter) -> str:
    payload = _CursorPayload(o=offset, f=filter_fingerprint(search))
    return base64.urlsafe_b64encode(payload.model_dump_json().encode()).decode().rstrip("=")


def decode_cursor(cursor: str, search: SearchFilter) -> int:
    """Return the offset, or raise `InvalidCursorError` for a malformed or foreign cursor."""
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        payload = _CursorPayload.model_validate_json(raw)
    # binascii.Error and pydantic's ValidationError are both ValueErrors.
    except ValueError:
        raise InvalidCursorError("the cursor is malformed") from None
    if payload.f != filter_fingerprint(search):
        raise InvalidCursorError("the cursor was issued for a different q/status") from None
    return payload.o
