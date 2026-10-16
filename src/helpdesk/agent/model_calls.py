"""The tool calls in one model reply, valid or not, as the graph answers them (CONTRACTS §5.3).

The Claude API needs exactly one tool_result for every tool_use it sent, on the next
request. langchain-anthropic 1.7.6 (`_format_messages`) sends a tool_use for every
`tool_calls` entry and for every `invalid_tool_calls` entry that has an id and a name (its
input is the parsed arguments, or `{}`). So the graph answers both lists, and the agent
node first makes sure every call has an id and a name, so none can be skipped there.
"""

import json
import uuid
from dataclasses import dataclass

from langchain_core.messages import AIMessage, InvalidToolCall
from pydantic import JsonValue

# Ids we mint match Anthropic's `^[a-zA-Z0-9_-]+$`, so the SDK never rewrites them.
_GENERATED_ID_PREFIX = "helpdesk_call_"
# Stands in for a missing name on an invalid call, which the formatter would otherwise drop.
UNNAMED_TOOL = "unnamed_tool"
# The `error` we put on a call that arrived without an id.
MISSING_ID = "the tool call had no id"

# Anthropic stop reasons that mean the reply was cut off mid-output; "length" is the
# OpenAI finish_reason for the same thing.
TRUNCATED_STOP_REASONS: frozenset[str] = frozenset({"max_tokens", "model_context_window_exceeded"})
TRUNCATED_FINISH_REASONS: frozenset[str] = frozenset({"length"})

_TOOL_BLOCK_TYPES = frozenset({"tool_use", "tool_call", "invalid_tool_call"})


@dataclass(frozen=True, slots=True)
class ModelCall:
    id: str
    name: str
    # The parsed arguments; `{}` when the model's JSON could not be parsed.
    args: dict[str, JsonValue]
    # Why the call is unusable as sent (unparseable arguments, no id); None for a valid call.
    invalid: str | None = None


def new_call_id() -> str:
    return f"{_GENERATED_ID_PREFIX}{uuid.uuid4().hex}"


def _parsed_args(raw: str | None) -> dict[str, JsonValue]:
    try:
        parsed: object = json.loads(raw or "{}")
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def normalize_response(message: AIMessage) -> AIMessage:
    """Give every tool call an id and a name before the reply is stored.

    A call without an id cannot be paired with its result, so it becomes an invalid call
    with a generated id: it is answered with an error and never run.
    """
    valid = [call for call in message.tool_calls if call["id"]]
    invalid: list[InvalidToolCall] = [
        InvalidToolCall(
            type="invalid_tool_call",
            id=None,
            name=call["name"],
            args=json.dumps(call["args"]),
            error=MISSING_ID,
        )
        for call in message.tool_calls
        if not call["id"]
    ]
    invalid.extend(message.invalid_tool_calls)
    if len(valid) == len(message.tool_calls) and all(c["id"] and c["name"] for c in invalid):
        return message
    named: list[InvalidToolCall] = []
    for call in invalid:
        fixed = call.copy()
        fixed["id"] = call["id"] or new_call_id()
        fixed["name"] = call["name"] or UNNAMED_TOOL
        named.append(fixed)
    return message.model_copy(update={"tool_calls": valid, "invalid_tool_calls": named})


def _content_order(message: AIMessage) -> dict[str, int]:
    """Position of each tool block in the reply content, which keeps the model's order."""
    order: dict[str, int] = {}
    if isinstance(message.content, list):
        for block in message.content:
            if isinstance(block, dict) and block.get("type") in _TOOL_BLOCK_TYPES:
                block_id = block.get("id")
                if isinstance(block_id, str):
                    order.setdefault(block_id, len(order))
    return order


def model_calls(message: AIMessage) -> list[ModelCall]:
    """Every call in the reply, valid and invalid, in the order the model made them.

    Without content blocks to order by, valid calls come before invalid ones. Calls still
    without an id (only in checkpoints older than `normalize_response`) cannot be answered
    and are left out.
    """
    calls = [
        ModelCall(id=call["id"], name=call["name"], args=dict(call["args"]))
        for call in message.tool_calls
        if call["id"]
    ]
    calls.extend(
        ModelCall(
            id=call["id"],
            name=call["name"] or UNNAMED_TOOL,
            args=_parsed_args(call["args"]),
            invalid=call["error"] or "the arguments are not valid JSON",
        )
        for call in message.invalid_tool_calls
        if call["id"]
    )
    order = _content_order(message)
    ranked = sorted(enumerate(calls), key=lambda item: (order.get(item[1].id, len(order)), item[0]))
    return [call for _, call in ranked]


def is_truncated(message: AIMessage) -> bool:
    """The reply stopped at the output limit, so its text or last tool call may be cut off."""
    metadata = message.response_metadata
    return (
        metadata.get("stop_reason") in TRUNCATED_STOP_REASONS
        or metadata.get("finish_reason") in TRUNCATED_FINISH_REASONS
    )
