"""Turning a write tool call into a PendingAction (CONTRACTS §5.3 `propose`)."""

import uuid

from langchain_core.messages import ToolCall
from pydantic import BaseModel, ValidationError

from helpdesk.shared.approval import PendingAction
from helpdesk.shared.errors import ErrorCode, format_tool_error
from helpdesk.shared.tools import AddCommentArgs, CreateTicketArgs, ToolName

_ARGS_MODELS: dict[ToolName, type[BaseModel]] = {
    ToolName.CREATE_TICKET: CreateTicketArgs,
    ToolName.ADD_COMMENT: AddCommentArgs,
}

_SUMMARY_BODY_CHARS = 120


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


def _validation_summary(error: ValidationError) -> str:
    parts = [
        f"{'.'.join(str(p) for p in item['loc']) or 'arguments'}: {item['msg']}"
        for item in error.errors(include_url=False, include_input=False)
    ]
    return "; ".join(parts)


def build_pending_action(call: ToolCall) -> PendingAction:
    """Validate a write call and create its pending action.

    This is the only place an idempotency key is created. It runs before the
    interrupt, in its own node, so resuming never mints a second key.
    """
    tool = ToolName(call["name"])
    try:
        args = _ARGS_MODELS[tool].model_validate(call["args"])
    except ValidationError as error:
        raise InvalidProposalError(
            format_tool_error(
                ErrorCode.INVALID_ARGUMENT,
                f"Invalid {tool.value} arguments ({_validation_summary(error)})",
            )
        ) from error
    if call["id"] is None:
        raise InvalidProposalError(
            format_tool_error(ErrorCode.INVALID_ARGUMENT, "The tool call has no id")
        )
    return PendingAction(
        action_id=str(uuid.uuid4()),
        idempotency_key=str(uuid.uuid4()),
        tool=tool,
        args=args.model_dump(mode="json"),
        summary=summarize(tool, args),
        tool_call_id=call["id"],
    )
