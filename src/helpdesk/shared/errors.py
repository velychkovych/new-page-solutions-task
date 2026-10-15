"""The tool error contract: `ERROR <CODE>: <what happened>. Next step: <what to do>.`"""

import re
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict


class ErrorCode(StrEnum):
    NOT_FOUND = "NOT_FOUND"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    CONFLICT = "CONFLICT"
    UPSTREAM_UNAVAILABLE = "UPSTREAM_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    AUTH = "AUTH"


# Defaults only; a tool may give a more specific next step.
DEFAULT_NEXT_STEP: Mapping[ErrorCode, str] = MappingProxyType(
    {
        ErrorCode.NOT_FOUND: "Check the id with the user, or search for the ticket",
        ErrorCode.INVALID_ARGUMENT: "Fix the arguments and call the tool again",
        ErrorCode.CONFLICT: "Do not retry; tell the user the change was not applied",
        ErrorCode.UPSTREAM_UNAVAILABLE: (
            "Tell the user the ticketing system is unavailable and to try again later"
        ),
        ErrorCode.TIMEOUT: "Tell the user the ticketing system timed out and to try again later",
        ErrorCode.AUTH: "Tell the user to contact an administrator",
    }
)

_TOOL_ERROR_RE = re.compile(
    r"ERROR (?P<code>[A-Z_]+): (?P<what>.+?)\. Next step: (?P<next_step>.+)\.", re.DOTALL
)


class ToolError(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: ErrorCode
    what: str
    next_step: str


def format_tool_error(code: ErrorCode, what: str, next_step: str | None = None) -> str:
    step = DEFAULT_NEXT_STEP[code] if next_step is None else next_step
    return f"ERROR {code.value}: {what.strip().rstrip('.')}. Next step: {step.strip().rstrip('.')}."


def parse_tool_error(text: str) -> ToolError | None:
    """Return the parsed error, or None if `text` is not in the contract format."""
    match = _TOOL_ERROR_RE.fullmatch(text.strip())
    if match is None or match["code"] not in ErrorCode.__members__:
        return None
    return ToolError(
        code=ErrorCode(match["code"]), what=match["what"], next_step=match["next_step"]
    )
