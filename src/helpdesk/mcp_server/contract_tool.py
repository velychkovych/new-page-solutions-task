"""FastMCP tools whose schema and validation come from the shared contract models.

Why not `@mcp.tool()`: in mcp 1.30.0, `Tool.run` validates arguments through
`FuncMetadata.call_fn_with_arg_validation` and wraps *any* failure as
`ToolError("Error executing tool <name>: <pydantic text>")`, which the low-level server
then returns with that prefix. CONTRACTS §2.4 needs the exact `ERROR INVALID_ARGUMENT: ...`
text, so `ContractTool.run` hands the raw arguments to a `ToolRunner`, which validates
them against the shared `*Args`/`*Input` model and always returns a `CallToolResult`.
The advertised `inputSchema` is that same model's JSON schema, so clients still see
every constraint (patterns, lengths, enums, bounds).
"""

import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, cast

from mcp.server.fastmcp import Context
from mcp.server.fastmcp.tools import Tool
from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase, FuncMetadata
from mcp.types import CallToolResult, TextContent
from opentelemetry.trace import Span, Status, StatusCode, Tracer
from pydantic import BaseModel, ValidationError, create_model

from helpdesk.mcp_server.errors import ToolCallError, invalid_arguments, unexpected_failure
from helpdesk.mcp_server.redaction import SecretRedactor, truncate_utf8
from helpdesk.mcp_server.retry import OnRetry
from helpdesk.shared import trace_attributes as attr

logger = logging.getLogger(__name__)

type ToolContext = Context[Any, Any, Any]
# Takes the validated input and a retry-progress callback; returns a shared wire model.
type ToolHandler[A: BaseModel] = Callable[[A, OnRetry], Awaitable[BaseModel]]


def retry_progress(context: ToolContext | None) -> OnRetry:
    """MCP progress before each retry, only when the client sent a progress token."""

    async def notify(attempt: int, max_attempts: int) -> None:
        if context is None:
            return
        try:
            meta = context.request_context.meta
        except ValueError:
            # No MCP request in flight (the tool was called directly, e.g. in a test).
            return
        if meta is None or meta.progressToken is None:
            return
        try:
            await context.report_progress(
                attempt - 1, max_attempts, message=f"Retrying (attempt {attempt} of {max_attempts})"
            )
        except Exception:
            # A lost progress message must not fail (or duplicate) the tool call itself.
            logger.debug("could not send retry progress", exc_info=True)

    return notify


def text_result(text: str, *, is_error: bool) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=is_error)


class ToolRunner[A: BaseModel]:
    """Validates, calls the handler, maps failures, and records the `mcp.tool` span."""

    def __init__(
        self,
        name: str,
        input_model: type[A],
        handler: ToolHandler[A],
        *,
        tracer: Tracer,
        redactor: SecretRedactor,
        result_max_bytes: int,
    ) -> None:
        self.name = name
        self.input_model = input_model
        self._handler = handler
        self._tracer = tracer
        self._redact = redactor
        self._result_max_bytes = result_max_bytes

    async def __call__(
        self, arguments: dict[str, Any], context: ToolContext | None
    ) -> CallToolResult:
        started = time.perf_counter()
        with self._tracer.start_as_current_span(
            attr.SPAN_TOOL, attributes={attr.TOOL_NAME: self.name, attr.TOOL_ATTEMPTS: 0}
        ) as span:
            # The injected ids and the idempotency key are not secrets, so they stay in the
            # span for debugging; the redactor still guarantees no secret value does.
            span.set_attribute(
                attr.TOOL_ARGS,
                self._redact(json.dumps(arguments, separators=(",", ":"), default=str)),
            )
            result = await self._run(arguments, context, span)
        logger.info(
            "tool %s finished: %s in %d ms",
            self.name,
            "error" if result.isError else "ok",
            (time.perf_counter() - started) * 1000,
        )
        return result

    async def _run(
        self, arguments: dict[str, Any], context: ToolContext | None, span: Span
    ) -> CallToolResult:
        try:
            args = self.input_model.model_validate(arguments)
        except ValidationError as exc:
            # Validated here, so the mock is never called with bad arguments.
            return self._fail(span, invalid_arguments(self.name, exc))
        try:
            value = await self._handler(args, retry_progress(context))
        except ToolCallError as exc:
            return self._fail(span, exc)
        except Exception as exc:
            # Last resort. Without it the low-level server returns str(exc) as the tool
            # text, which breaks the error contract and may carry upstream bytes.
            logger.error("tool %s failed unexpectedly: %s", self.name, type(exc).__name__)
            return self._fail(span, unexpected_failure(self.name))
        text = self._redact(value.model_dump_json())
        span.set_attribute(attr.TOOL_RESULT, truncate_utf8(text, self._result_max_bytes))
        return text_result(text, is_error=False)

    def _fail(self, span: Span, error: ToolCallError) -> CallToolResult:
        text = self._redact(error.text)
        span.set_attribute(attr.TOOL_ERROR_CODE, error.code.value)
        span.set_attribute(attr.TOOL_RESULT, truncate_utf8(text, self._result_max_bytes))
        span.set_status(Status(StatusCode.ERROR, error.code.value))
        return text_result(text, is_error=True)


class ContractTool(Tool):
    """A FastMCP `Tool` that skips FastMCP's own argument validation (see module doc)."""

    async def run(
        self,
        arguments: dict[str, Any],
        context: ToolContext | None = None,
        convert_result: bool = False,
    ) -> CallToolResult:
        # The low-level server returns a CallToolResult as-is, so `convert_result` has
        # nothing to do here.
        result: CallToolResult = await self.fn(arguments, context)
        return result


def _arg_model(input_model: type[BaseModel]) -> type[ArgModelBase]:
    # `Tool` requires FuncMetadata. It is not used for validation here, but it mirrors the
    # contract model so anything in FastMCP that reads it sees the same fields.
    model = create_model(f"{input_model.__name__}Arguments", __base__=(input_model, ArgModelBase))
    return cast(type[ArgModelBase], model)


def contract_tool[A: BaseModel](runner: ToolRunner[A], description: str) -> ContractTool:
    return ContractTool(
        fn=runner,
        name=runner.name,
        description=description,
        parameters=runner.input_model.model_json_schema(by_alias=True),
        fn_metadata=FuncMetadata(arg_model=_arg_model(runner.input_model)),
        is_async=True,
    )
