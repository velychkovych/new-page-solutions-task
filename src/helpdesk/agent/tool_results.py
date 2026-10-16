"""One call to the tool provider, with whatever comes back turned into a well-formed result.

The rule for failures:
- An MCP tool error in the contract format is passed on exactly.
- `ToolTransportError` → UPSTREAM_UNAVAILABLE, at once: the MCP server is the only retry
  layer.
- Any other `Exception`, and any result that breaks the contract (not a `ToolResult`,
  empty text, an error not in the `ERROR <CODE>` format, a write result that is not a JSON
  object) → INTERNAL. A write says the change could not be confirmed.
- A `BaseException` that is not an `Exception` (cancellation from the turn deadline,
  process exit) is never caught. The node then never finishes, the checkpoint stays before
  `execute_write`, and the next request re-runs the write with the same idempotency key
  (CONTRACTS §5.3), instead of recording an outcome nobody saw.
"""

import json
import logging
from collections.abc import Mapping

from pydantic import JsonValue

from helpdesk.agent.tool_errors import internal_failure, transport_failure
from helpdesk.agent.tool_provider import ToolProvider, ToolResult, ToolTransportError
from helpdesk.shared.errors import parse_tool_error

logger = logging.getLogger(__name__)


def _is_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except ValueError:
        return False


def _contract_problem(result: ToolResult, *, write: bool) -> str | None:
    """Why `result` cannot be passed on as it is, or None when it can."""
    if not result.text.strip():
        return "the tool returned an empty result"
    if result.is_error:
        if parse_tool_error(result.text) is None:
            # Not echoed: it could be a raw upstream body or a stack trace (CONTRACTS §2.4).
            return "the tool returned an error that is not in the ERROR format"
        return None
    if write and not _is_json_object(result.text):
        return "the tool returned a result that is not a JSON object"
    return None


async def call_tool(
    provider: ToolProvider, name: str, args: Mapping[str, JsonValue], *, write: bool
) -> ToolResult:
    try:
        # `object`: a provider that breaks its protocol is handled, not trusted.
        result: object = await provider.call(name, args)
    except ToolTransportError:
        return ToolResult(text=transport_failure(name, write=write), is_error=True)
    except Exception as error:
        # Only the type: an exception message can hold arguments or a token.
        logger.warning("tool %s raised %s", name, type(error).__name__)
        cause = f"the tool provider raised {type(error).__name__}"
        return ToolResult(text=internal_failure(name, cause, write=write), is_error=True)
    if isinstance(result, ToolResult):
        problem = _contract_problem(result, write=write)
        if problem is None:
            return result
    else:
        problem = "the tool provider returned no tool result"
    logger.warning("tool %s broke the result contract: %s", name, problem)
    return ToolResult(text=internal_failure(name, problem, write=write), is_error=True)
