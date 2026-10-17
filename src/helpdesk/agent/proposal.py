"""Turning a write tool call into a PendingAction (CONTRACTS §5.3 `propose`)."""

import uuid

from langchain_core.messages import ToolCall
from pydantic import BaseModel, JsonValue, ValidationError

from helpdesk.agent.tool_errors import invalid_args_error, missing_id_error
from helpdesk.shared.approval import PendingAction, WriteResult
from helpdesk.shared.errors import ErrorCode
from helpdesk.shared.tools import AddCommentArgs, CreateTicketArgs, ToolName

_ARGS_MODELS: dict[ToolName, type[BaseModel]] = {
    ToolName.CREATE_TICKET: CreateTicketArgs,
    ToolName.ADD_COMMENT: AddCommentArgs,
}

_SUMMARY_BODY_CHARS = 120

# Write errors after which the change may or may not have been saved (CONTRACTS §2.4).
_OUTCOME_UNKNOWN_CODES: frozenset[ErrorCode] = frozenset(
    {ErrorCode.UPSTREAM_UNAVAILABLE, ErrorCode.TIMEOUT, ErrorCode.INTERNAL}
)


class InvalidProposalError(ValueError):
    """The model's write arguments do not match the model-visible schema."""

    def __init__(self, tool_error: str) -> None:
        super().__init__(tool_error)
        self.tool_error = tool_error


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def summarize(tool: ToolName, args: BaseModel) -> str:
    """A deterministic, human-readable line describing the write."""
    match args:
        case CreateTicketArgs():
            kind = f"{args.priority.value}-priority {args.category.value}"
            return f'Create a {kind} ticket: "{args.title}"'
        case AddCommentArgs():
            return (
                f'Add a comment to {args.ticket_id}: "{_shorten(args.body, _SUMMARY_BODY_CHARS)}"'
            )
    raise ValueError(f"no summary for {tool}")  # pragma: no cover - _ARGS_MODELS is exhaustive


def idempotency_key_for(
    tool: ToolName, args: dict[str, JsonValue], last_write: WriteResult | None
) -> str:
    """A new key, unless this re-proposes the last write and that write's outcome is unknown.

    That write may have been saved. Sending its key again makes the mock replay the stored
    result instead of saving the change twice; if it was never saved, it runs once now.
    Only identical args qualify: the mock rejects a reused key with a different body.
    """
    if (
        last_write is not None
        and not last_write.ok
        and last_write.error_code in _OUTCOME_UNKNOWN_CODES
        and last_write.tool == tool
        and last_write.args == args
    ):
        return last_write.idempotency_key
    return str(uuid.uuid4())


def build_pending_action(call: ToolCall, last_write: WriteResult | None) -> PendingAction:
    """Validate a write call and create its pending action.

    This is the only place an idempotency key is created or reused. It runs before the
    interrupt, in its own node, so resuming never mints a second key.
    """
    tool = ToolName(call["name"])
    model = _ARGS_MODELS[tool]
    try:
        args = model.model_validate(call["args"])
    except ValidationError as error:
        raise InvalidProposalError(invalid_args_error(tool.value, error, model)) from error
    if not call["id"]:
        # The agent node gives every call an id, so this only guards the type.
        raise InvalidProposalError(missing_id_error(tool.value))
    visible_args: dict[str, JsonValue] = args.model_dump(mode="json")
    return PendingAction(
        action_id=str(uuid.uuid4()),
        idempotency_key=idempotency_key_for(tool, visible_args, last_write),
        tool=tool,
        args=visible_args,
        summary=summarize(tool, args),
        tool_call_id=call["id"],
    )
