"""The `ERROR <CODE>: ...` texts the agent writes itself (CONTRACTS §2.4, §5.3).

Each one names the tool and the concrete cause and gives one action, so the model can
act on it. None of them holds exception text, a stack trace or a raw tool output.
"""

from collections.abc import Sequence

from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorDetails

from helpdesk.agent.routing import WRITE_TOOL_NAMES
from helpdesk.shared.errors import ErrorCode, format_tool_error
from helpdesk.shared.memory import REMEMBER_FACT_TOOL, FactKey
from helpdesk.shared.tools import INJECTED_ARGS

# Fixed by CONTRACTS §5.3; what was combined goes into the next step.
ONE_CHANGE_AT_A_TIME_WHAT = "propose one change at a time, after your reads"

# A write that may or may not have been saved (CONTRACTS §2.4).
UNCONFIRMED_WRITE_NEXT_STEP = (
    "Tell the user the change could not be confirmed and may or may not have been saved; "
    "do not make up a ticket id, and do not propose the change again unless the user asks"
)

_INJECTED_NAMES: frozenset[str] = frozenset(
    name for args in INJECTED_ARGS.values() for name in args
)


def _names(names: Sequence[str]) -> str:
    return ", ".join(names)


def unknown_tool_error(name: str, available: Sequence[str]) -> str:
    return format_tool_error(
        ErrorCode.INVALID_ARGUMENT,
        f"There is no tool named {name!r}, so the call was not run",
        f"Use only these tools: {_names(available)}",
    )


def rejected_batch_error(names: Sequence[str], defect: str | None = None) -> str:
    """For each call of a reply that held more than one write, or a write next to other calls.

    CONTRACTS §5.3 fixes the text for every call of such a reply, broken ones included; a
    call's own `defect` (cut off, no id, unparseable JSON) is added to its next step, so the
    model still learns what to fix in it.
    """
    writes = [name for name in names if name in WRITE_TOOL_NAMES]
    others = [name for name in names if name not in WRITE_TOOL_NAMES]
    combined = (
        f"{len(writes)} writes ({_names(writes)})" if len(writes) > 1 else f"the write {writes[0]}"
    )
    if others:
        combined += f" with {_names(others)}"
    also = f", and {defect}" if defect is not None else ""
    return format_tool_error(
        ErrorCode.INVALID_ARGUMENT,
        ONE_CHANGE_AT_A_TIME_WHAT,
        f"None of these calls ran, because one message combined {combined}{also}; make your "
        "read calls first, then send the single write as the only call in a message of its own",
    )


def truncated_call_defect(name: str) -> str:
    return f"the reply was cut off at the output token limit, so the {name} call may be incomplete"


def missing_id_defect(name: str) -> str:
    return f"the {name} call had no tool call id"


def unparseable_args_defect(name: str) -> str:
    return f"the arguments of the {name} call are not a valid JSON object"


def unparseable_args_error(name: str) -> str:
    return format_tool_error(
        ErrorCode.INVALID_ARGUMENT,
        f"The arguments of the {name} call are not a valid JSON object, so the call was not run",
        f"Call {name} again with its arguments as one JSON object that matches its input schema",
    )


def missing_id_error(name: str) -> str:
    return format_tool_error(
        ErrorCode.INVALID_ARGUMENT,
        f"The {name} call arrived without a tool call id, so it was not run",
        f"Call {name} again if you still need its result",
    )


def truncated_call_error(name: str) -> str:
    return format_tool_error(
        ErrorCode.INVALID_ARGUMENT,
        f"Your reply was cut off at the output token limit, so the {name} call may be "
        "incomplete and was not run",
        f"Call {name} again with shorter arguments, and keep any text before the call short",
    )


def _problem(item: ErrorDetails, tool: str) -> str:
    name = ".".join(str(part) for part in item["loc"]) or "arguments"
    match item["type"]:
        case "extra_forbidden" if name in _INJECTED_NAMES:
            return f"`{name}` is not an argument of {tool}: the agent fills it in itself"
        case "extra_forbidden":
            return f"`{name}` is not an argument of {tool}"
        case "missing":
            return f"`{name}` is required"
        case _:
            return f"`{name}`: {item['msg']}"


def invalid_args_error(
    tool: str, error: ValidationError, model: type[BaseModel], *, next_step: str | None = None
) -> str:
    """Names each bad argument and why; pydantic's messages carry the allowed values."""
    problems = "; ".join(
        _problem(item, tool) for item in error.errors(include_url=False, include_input=False)
    )
    allowed = _names(list(model.model_fields))
    return format_tool_error(
        ErrorCode.INVALID_ARGUMENT,
        f"{tool} got invalid arguments ({problems}), so it was not run",
        next_step or f"Call {tool} again with only these arguments, fixed as described: {allowed}",
    )


REMEMBER_FACT_NEXT_STEP = (
    f"Call {REMEMBER_FACT_TOOL} again with only `key`, one of "
    f"{_names([key.value for key in FactKey])}, and `value`, 1 to 200 characters; "
    "save nothing else"
)


def transport_failure(tool: str, *, write: bool) -> str:
    what = f"{tool} could not reach the ticketing tools: the connection to the MCP server failed"
    if not write:
        return format_tool_error(ErrorCode.UPSTREAM_UNAVAILABLE, what)
    return format_tool_error(
        ErrorCode.UPSTREAM_UNAVAILABLE,
        f"{what}, so the change could not be confirmed",
        UNCONFIRMED_WRITE_NEXT_STEP,
    )


def internal_failure(tool: str, cause: str, *, write: bool) -> str:
    """A failure inside the agent or its tool provider: a bug, never retried here."""
    what = f"{tool} failed inside the helpdesk agent: {cause}"
    if not write:
        return format_tool_error(ErrorCode.INTERNAL, what)
    return format_tool_error(
        ErrorCode.INTERNAL,
        f"{what}, so the change could not be confirmed",
        UNCONFIRMED_WRITE_NEXT_STEP,
    )
