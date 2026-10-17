"""Typed tool failures and their mapping to the error contract (CONTRACTS.md §2.4).

Every text names the tool and the concrete cause, and gives one next step that fits it.
For a write it also says whether the change was applied: a 4xx means the mock did not run
it, while a timeout, an unavailable or misbehaving upstream, or a bug in this server leave
the outcome unknown.

Error text is built only from our own wording, argument and tool names, HTTP status
codes, attempt counts and exception type names. It never includes a raw upstream body, an
exception message, a stack trace or a header value. A name the model sent is echoed only
when it is a plain identifier, so the model's text cannot add a second "Next step:".
"""

import re
from collections.abc import Collection, Mapping, Sequence
from typing import Any

import httpx
from pydantic import BaseModel, StringConstraints, ValidationError

from helpdesk.mcp_server.retry import UpstreamCallError
from helpdesk.shared.errors import ErrorCode, format_tool_error
from helpdesk.shared.ids import IDEMPOTENCY_KEY_PATTERN, PRINCIPAL_ID_PATTERN, TICKET_ID_PATTERN
from helpdesk.shared.tickets import ApiError
from helpdesk.shared.tools import INJECTED_ARGS, WRITE_TOOLS

# The model must not claim either outcome, and must not make up a ticket id. It must not
# propose the write again on its own either. The agent reuses the idempotency key only for
# a re-proposal with identical arguments; with any other arguments, a write that did
# commit would be applied twice.
WRITE_NOT_CONFIRMED_NEXT_STEP = (
    "Tell the user the change could not be confirmed and may or may not have been applied;"
    " do not propose it again unless the user asks (then with exactly the same arguments),"
    " and do not make up a ticket id"
)
WRITE_INTERNAL_NEXT_STEP = (
    "Tell the user the change could not be confirmed and to contact an administrator;"
    " do not propose it again and do not make up a ticket id"
)
MISCONFIGURED_NEXT_STEP = (
    "Tell the user the helpdesk is misconfigured and to contact an administrator"
)
_REJECTED_NEXT_STEP = (
    "Do not repeat this call with the same arguments; tell the user the ticketing system"
    " rejected the request"
)
_NOT_APPLIED_ADMIN_NEXT_STEP = (
    "Do not retry; tell the user the change was not applied and to contact an administrator"
)
_MAX_VALIDATION_ISSUES = 5
# Any other name the model sent is described, never echoed.
_SHOWN_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
# Why a value of the right shape is still rejected, where the rule alone does not say:
# strict mode rejects 7.0 and "7" for an integer.
_TYPE_REASONS: Mapping[str, str] = {
    "int_type": "sent as a JSON integer without quotes or a decimal point",
}

# Patterns read better as a description than as a regex.
_PATTERN_RULES: Mapping[str, str] = {
    TICKET_ID_PATTERN: "INC- followed by four digits, for example INC-0042",
    PRINCIPAL_ID_PATTERN: "1 to 64 letters, digits, '_' or '-'",
    IDEMPOTENCY_KEY_PATTERN: "1 to 255 printable ASCII characters without spaces",
}
# Arguments whose schema ("null or text") says too little about what is valid.
_ARGUMENT_RULES: Mapping[str, str] = {
    "cursor": "null or the next_cursor of an earlier search_tickets result for the same query"
    " and status",
}


class ToolCallError(Exception):
    """A failure that the tool reports to the model as `ERROR <CODE>: ...`.

    `cause` is the concrete reason, without the tool name, which `text` adds. When
    `outcome_unknown` is set, a write is reported as unconfirmed rather than not applied.
    """

    def __init__(
        self,
        code: ErrorCode,
        cause: str,
        *,
        next_step: str | None = None,
        write_next_step: str | None = None,
        outcome_unknown: bool = False,
    ) -> None:
        super().__init__(cause)
        self.code = code
        self.cause = cause
        self.next_step = next_step
        self.outcome_unknown = outcome_unknown
        if write_next_step is None and outcome_unknown:
            write_next_step = WRITE_NOT_CONFIRMED_NEXT_STEP
        self.write_next_step = write_next_step if write_next_step is not None else next_step

    def text(self, tool: str) -> str:
        if tool not in WRITE_TOOLS:
            return format_tool_error(self.code, f"{tool} failed: {self.cause}", self.next_step)
        if self.outcome_unknown:
            what = f"{tool} could not confirm whether the change was applied: {self.cause}"
        else:
            what = f"{tool} failed and the change was not applied: {self.cause}"
        return format_tool_error(self.code, what, self.write_next_step)


def _join(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _attempts(attempts: int) -> str:
    return "1 attempt" if attempts == 1 else f"{attempts} attempts"


def _shown(name: str) -> bool:
    return _SHOWN_NAME.fullmatch(name) is not None


# --- argument validation ---------------------------------------------------------------


def _resolve(prop: Mapping[str, Any], defs: Mapping[str, Any]) -> Mapping[str, Any]:
    ref = prop.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        resolved: Mapping[str, Any] = defs[ref.removeprefix("#/$defs/")]
        return resolved
    return prop


def _rule(prop: Mapping[str, Any], defs: Mapping[str, Any]) -> str:
    """What a value must be, from the advertised JSON schema, so the text and the schema
    the client sees cannot drift apart."""
    prop = _resolve(prop, defs)
    if options := prop.get("anyOf"):
        rules = [_rule(option, defs) for option in options if option.get("type") != "null"]
        nullable = len(rules) < len(options)
        return ("null or " if nullable else "") + " or ".join(rules)
    if enum := prop.get("enum"):
        return "one of: " + ", ".join(str(value) for value in enum)
    match prop.get("type"):
        case "integer":
            rule = f"a whole number from {prop['minimum']} to {prop['maximum']}"
        case "array":
            rule = f"a list whose items are each {_rule(prop['items'], defs)}"
        case "string" if (pattern := prop.get("pattern")) is not None:
            rule = _PATTERN_RULES.get(pattern, f"text matching the pattern {pattern}")
        case "string" if "maxLength" in prop:
            rule = f"text of {prop.get('minLength', 0)} to {prop['maxLength']} characters"
        case "string":
            rule = "text"
        case _:
            rule = "a valid value"
    return rule


def _stripped_arguments(input_model: type[BaseModel]) -> set[str]:
    """Arguments whose whitespace is stripped before the length check. The JSON schema does
    not say so, and without it "   " looks like valid text of 1 to 200 characters."""
    return {
        field.alias or name
        for name, field in input_model.model_fields.items()
        if any(isinstance(m, StringConstraints) and m.strip_whitespace for m in field.metadata)
    }


def argument_rules(input_model: type[BaseModel]) -> dict[str, str]:
    """Each argument's rule, e.g. {"page_size": "a whole number from 1 to 20"}."""
    schema = input_model.model_json_schema(by_alias=True)
    defs: Mapping[str, Any] = schema.get("$defs", {})
    stripped = _stripped_arguments(input_model)
    rules: dict[str, str] = {}
    for name, prop in schema["properties"].items():
        rule = _ARGUMENT_RULES.get(name) or _rule(prop, defs)
        if name in stripped:
            rule += ", not counting leading and trailing whitespace"
        rules[name] = rule
    return rules


def _broken_rule(name: str, error_type: str, rule: str) -> str:
    if error_type == "missing":
        return f"{name} is required and must be {rule}"
    if reason := _TYPE_REASONS.get(error_type):
        return f"{name} must be {rule}, {reason}"
    return f"{name} must be {rule}"


def _fix_step(tool: str, wrong: Sequence[str], unknown: Sequence[str], hidden: int) -> str:
    fixes: list[str] = []
    if wrong:
        fixes.append(f"fix {_join(wrong)}")
    removals = list(unknown)
    if hidden:
        other = "every other" if removals else "every"
        removals.append(f"{other} argument {tool} does not take")
    if removals:
        fixes.append(f"remove {_join(removals)}")
    step = f"{', '.join(fixes)}, then call {tool} again"
    return step[0].upper() + step[1:]


def invalid_arguments(
    tool: str, input_model: type[BaseModel], exc: ValidationError
) -> ToolCallError:
    """One issue per argument, with the rule it broke. The rejected value is left out: it
    can be long, and the model already has it."""
    rules = argument_rules(input_model)
    issues: dict[str, str] = {}
    unknown: dict[str, str] = {}
    hidden = 0
    for error in exc.errors(include_url=False, include_input=False):
        if not error["loc"]:
            # Not reachable over MCP, whose arguments are always an object.
            issues.setdefault("arguments", "the arguments must be a JSON object")
            continue
        name = str(error["loc"][0])
        if name not in rules:
            if _shown(name):
                unknown[name] = f"{name} is not an argument of {tool}"
            else:
                hidden += 1
        else:
            issues.setdefault(name, _broken_rule(name, error["type"], rules[name]))
    # Schema order, then unknown arguments, so the text does not depend on input order.
    order = list(rules)
    wrong = sorted(issues, key=lambda name: order.index(name) if name in order else -1)
    issues = {name: issues[name] for name in wrong} | unknown
    if hidden:
        names = "1 argument whose name is" if hidden == 1 else f"{hidden} arguments whose names are"
        verb = "is not an argument" if hidden == 1 else "are not arguments"
        issues["<hidden>"] = (
            f"{names} not shown {verb} of {tool}, which takes only {_join(list(rules))}"
        )

    shown = list(issues.values())[:_MAX_VALIDATION_ISSUES]
    cause = "; ".join(shown)
    if len(issues) > _MAX_VALIDATION_ISSUES:
        cause += f"; and {len(issues) - _MAX_VALIDATION_ISSUES} more"

    injected = {arg for name, args in INJECTED_ARGS.items() if name == tool for arg in args}
    if injected & issues.keys():
        # The agent fills these in, so the model cannot fix them by calling again.
        return ToolCallError(
            ErrorCode.INVALID_ARGUMENT, cause, next_step=_NOT_APPLIED_ADMIN_NEXT_STEP
        )
    return ToolCallError(
        ErrorCode.INVALID_ARGUMENT, cause, next_step=_fix_step(tool, wrong, list(unknown), hidden)
    )


# --- mock responses --------------------------------------------------------------------


def _api_error_code(response: httpx.Response) -> str | None:
    try:
        return ApiError.model_validate_json(response.content).error
    except ValidationError:
        return None


def error_for_status(
    response: httpx.Response, *, attempts: int, ticket_id: str | None = None
) -> ToolCallError:
    """Map a final non-2xx mock response. Retryable statuses never get here."""
    status = response.status_code
    match status:
        case 404:
            return _not_found(ticket_id)
        case 401 | 403:
            return ToolCallError(
                ErrorCode.AUTH,
                f"the ticketing system rejected the MCP server's credentials (HTTP {status}),"
                " so the MCP server is misconfigured",
                next_step=MISCONFIGURED_NEXT_STEP,
            )
        case 409:
            return ToolCallError(
                ErrorCode.CONFLICT,
                "its idempotency key was already used for a different change (HTTP 409)",
            )
        case 400:
            return _bad_request(_api_error_code(response))
        case 422:
            return ToolCallError(
                ErrorCode.INVALID_ARGUMENT,
                "the ticketing system rejected the arguments as invalid (HTTP 422)",
                next_step=_REJECTED_NEXT_STEP,
                write_next_step=_NOT_APPLIED_ADMIN_NEXT_STEP,
            )
        case _:
            return ToolCallError(
                ErrorCode.UPSTREAM_UNAVAILABLE,
                f"the ticketing system sent an unexpected HTTP {status} response"
                f" ({_attempts(attempts)})",
                outcome_unknown=True,
            )


def _not_found(ticket_id: str | None) -> ToolCallError:
    if ticket_id is None:
        # Only a ticket path can be missing; anything else means a wrong mock URL.
        return ToolCallError(
            ErrorCode.NOT_FOUND,
            "the ticketing system answered HTTP 404 for this request, so the MCP server is"
            " misconfigured",
            next_step=MISCONFIGURED_NEXT_STEP,
        )
    return ToolCallError(
        ErrorCode.NOT_FOUND,
        f"ticket {ticket_id} does not exist",
        next_step=f"Tell the user ticket {ticket_id} does not exist and ask them to check the id",
    )


def _bad_request(api_code: str | None) -> ToolCallError:
    match api_code:
        case "invalid_cursor":
            return ToolCallError(
                ErrorCode.INVALID_ARGUMENT,
                "the cursor does not belong to this search: it is malformed, or it came from"
                " a search with a different query or status (HTTP 400)",
                next_step="Call search_tickets again with the same query and status and no cursor",
            )
        case "missing_idempotency_key" | "invalid_idempotency_key":
            # The key was validated here first, so the agent and the mock disagree on it.
            return ToolCallError(
                ErrorCode.INVALID_ARGUMENT,
                f"the ticketing system rejected the idempotency key ({api_code}, HTTP 400)",
                next_step=_NOT_APPLIED_ADMIN_NEXT_STEP,
            )
        case _:
            return ToolCallError(
                ErrorCode.INVALID_ARGUMENT,
                "the ticketing system rejected the request as invalid (HTTP 400)",
                next_step=_REJECTED_NEXT_STEP,
                write_next_step=_NOT_APPLIED_ADMIN_NEXT_STEP,
            )


def malformed_response(status: int, *, attempts: int) -> ToolCallError:
    return ToolCallError(
        ErrorCode.UPSTREAM_UNAVAILABLE,
        f"the ticketing system sent an HTTP {status} response in an unexpected format"
        f" ({_attempts(attempts)})",
        outcome_unknown=True,
    )


# --- retry outcomes --------------------------------------------------------------------


def _last_attempt(exc: UpstreamCallError) -> str:
    if exc.last_status is not None:
        outcome = f"got HTTP {exc.last_status}"
    else:
        outcome = exc.last_failure or "got no response"
    if exc.attempts == 1:
        return f"1 attempt, which {outcome}"
    return f"{exc.attempts} attempts; the last one {outcome}"


def upstream_unavailable(exc: UpstreamCallError) -> ToolCallError:
    return ToolCallError(
        ErrorCode.UPSTREAM_UNAVAILABLE,
        f"the ticketing system is unavailable ({_last_attempt(exc)})",
        outcome_unknown=True,
    )


def deadline_exceeded(exc: UpstreamCallError, deadline_s: float) -> ToolCallError:
    return ToolCallError(
        ErrorCode.TIMEOUT,
        f"the ticketing system did not answer within the {deadline_s:g} s deadline"
        f" ({_attempts(exc.attempts)})",
        outcome_unknown=True,
    )


# --- unknown tools ---------------------------------------------------------------------


def unknown_tool(name: str, tools: Collection[str]) -> str:
    """FastMCP's own text is "Unknown tool: <name>", outside the contract and echoing the
    model's text as-is."""
    tool = f"no tool named {name}" if _shown(name) else "no tool with the requested name"
    return format_tool_error(
        ErrorCode.INVALID_ARGUMENT,
        f"there is {tool}, so nothing was run; the tools are {_join(list(tools))}",
        "Call one of the listed tools instead",
    )


# --- bugs ------------------------------------------------------------------------------


def internal_error() -> ToolCallError:
    """An exception no other mapping covers: a bug here, not the arguments or the mock."""
    return ToolCallError(
        ErrorCode.INTERNAL,
        "an unexpected internal error occurred in the MCP server (a bug, not caused by the"
        " arguments or the ticketing system)",
        write_next_step=WRITE_INTERNAL_NEXT_STEP,
        outcome_unknown=True,
    )
