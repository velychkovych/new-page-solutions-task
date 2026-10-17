"""Graph nodes (CONTRACTS §5.3).

No node has a RetryPolicy and no node retries a tool: the MCP server is the only
tool retry layer. Transport failures become UPSTREAM_UNAVAILABLE results at once.

The reads and execute_write report each provider call on LangGraph's custom stream
(`tool_activity`) for /chat/stream. That is not graph state, and approval reports nothing.
"""

import asyncio
import json
import logging
import time
from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore
from langgraph.types import Command, interrupt
from pydantic import JsonValue, ValidationError

from helpdesk.agent.decision import approves
from helpdesk.agent.history import model_messages
from helpdesk.agent.memory import Clock, load_facts
from helpdesk.agent.model_calls import (
    MISSING_ID,
    ModelCall,
    is_truncated,
    model_calls,
    normalize_response,
)
from helpdesk.agent.prompt import system_message
from helpdesk.agent.prompt_caching import cacheable_system_message
from helpdesk.agent.proposal import InvalidProposalError, build_pending_action
from helpdesk.agent.routing import (
    LOCAL_TOOL_NAMES,
    READ_TOOL_NAMES,
    WRITE_TOOL_NAMES,
    Node,
    ToolCallBatch,
    classify_tool_calls,
    last_ai_message,
    proposable,
)
from helpdesk.agent.run_config import user_id_from_config
from helpdesk.agent.state import AgentState
from helpdesk.agent.tool_activity import tool_activity
from helpdesk.agent.tool_errors import (
    REMEMBER_FACT_NEXT_STEP,
    internal_failure,
    invalid_args_error,
    missing_id_defect,
    missing_id_error,
    rejected_batch_error,
    truncated_call_defect,
    truncated_call_error,
    unknown_tool_error,
    unparseable_args_defect,
    unparseable_args_error,
)
from helpdesk.agent.tool_messages import declined_message, elapsed_ms, tool_message
from helpdesk.agent.tool_provider import ToolProvider
from helpdesk.agent.tool_results import call_tool
from helpdesk.shared.approval import PendingAction, WriteResult
from helpdesk.shared.errors import parse_tool_error
from helpdesk.shared.memory import RememberFactArgs
from helpdesk.shared.tools import INJECTED_ARGS

logger = logging.getLogger(__name__)

IDEMPOTENCY_KEY_ARG = "idempotency_key"


class GraphStateError(RuntimeError):
    """A node ran in a state the topology should make impossible."""


def injected_args(pending: PendingAction, user_id: str) -> dict[str, JsonValue]:
    """reporter_id/author_id from the request, the key from the pending action."""
    return {
        name: pending.idempotency_key if name == IDEMPOTENCY_KEY_ARG else user_id
        for name in INJECTED_ARGS[pending.tool]
    }


def _parse_result(text: str) -> JsonValue:
    try:
        parsed: JsonValue = json.loads(text)
    except ValueError:
        return text
    return parsed


def _defect(call: ModelCall, message: AIMessage) -> str | None:
    """What is wrong with the call itself, most basic cause first: a cut-off reply explains
    its unparseable calls."""
    if is_truncated(message):
        return truncated_call_defect(call.name)
    if call.invalid == MISSING_ID:
        return missing_id_defect(call.name)
    if call.invalid is not None:
        return unparseable_args_defect(call.name)
    return None


def _rejected(call: ModelCall, message: AIMessage, calls: Sequence[ModelCall]) -> str | None:
    """The error for a call that must not run, or None when it may.

    A rejected batch gives every call the batch text of CONTRACTS §5.3, with the call's own
    defect in its next step. Outside a batch, a broken call gets an error for its defect.
    """
    if classify_tool_calls(calls) is ToolCallBatch.REJECTED:
        return rejected_batch_error([c.name for c in calls], _defect(call, message))
    if is_truncated(message):
        return truncated_call_error(call.name)
    if call.invalid == MISSING_ID:
        return missing_id_error(call.name)
    if call.invalid is not None:
        return unparseable_args_error(call.name)
    if call.name in WRITE_TOOL_NAMES:
        # A lone, whole write goes to propose; it never runs here.
        return rejected_batch_error([call.name])
    return None


class AgentNodes:
    def __init__(
        self,
        *,
        model: BaseChatModel,
        bound_model: Runnable[LanguageModelInput, AIMessage],
        tool_provider: ToolProvider,
        tool_names: Sequence[str],
        remember_fact_tool: BaseTool,
        history_token_budget: int,
        slow_tool_notice_s: float,
        clock: Clock,
    ) -> None:
        self._model = model
        self._bound_model = bound_model
        self._tools = tool_provider
        # The names bound to the model, in order: what an unknown-tool error offers.
        self._tool_names = tuple(tool_names)
        self._remember_fact = remember_fact_tool
        self._history_token_budget = history_token_budget
        self._slow_tool_notice_s = slow_tool_notice_s
        self._clock = clock

    async def load_context(
        self, state: AgentState, config: RunnableConfig, store: BaseStore
    ) -> AgentState:
        user_id = user_id_from_config(config)
        facts = await load_facts(store, user_id, now=self._clock())
        return {"messages": [system_message(facts)], "user_id": user_id}

    async def agent(self, state: AgentState, config: RunnableConfig) -> AgentState:
        messages = model_messages(
            state.get("messages", []), token_budget=self._history_token_budget
        )
        if messages and isinstance(messages[0], SystemMessage):
            messages[0] = cacheable_system_message(messages[0], self._model)
        response = await self._bound_model.ainvoke(messages, config)
        return {"messages": [normalize_response(response)]}

    async def tools(
        self, state: AgentState, config: RunnableConfig, store: BaseStore
    ) -> AgentState:
        """One ToolMessage per call of the last reply, in the model's order."""
        message = last_ai_message(state)
        if message is None:
            raise GraphStateError("tools ran without an AI message")
        calls = model_calls(message)
        results: list[AnyMessage] = list(
            await asyncio.gather(
                *(self._answer(call, message, calls, config, store) for call in calls)
            )
        )
        return {"messages": results}

    async def _answer(
        self,
        call: ModelCall,
        message: AIMessage,
        calls: Sequence[ModelCall],
        config: RunnableConfig,
        store: BaseStore,
    ) -> ToolMessage:
        rejected = _rejected(call, message, calls)
        if rejected is not None:
            return tool_message(call_id=call.id, name=call.name, text=rejected, is_error=True)
        if call.name in LOCAL_TOOL_NAMES:
            return await self._remember(call, config, store)
        if call.name in READ_TOOL_NAMES and call.name in self._tool_names:
            return await self._read(call)
        # Only a tool the model was given, and that we know to be a read, ever runs.
        text = unknown_tool_error(call.name, self._tool_names)
        return tool_message(call_id=call.id, name=call.name, text=text, is_error=True)

    async def _read(self, call: ModelCall) -> ToolMessage:
        with tool_activity(
            call.id, call.name, slow_after_s=self._slow_tool_notice_s
        ) as on_progress:
            started = time.perf_counter()
            result = await call_tool(
                self._tools, call.name, call.args, write=False, on_progress=on_progress
            )
            latency = elapsed_ms(started, time.perf_counter())
        return tool_message(
            call_id=call.id,
            name=call.name,
            text=result.text,
            is_error=result.is_error,
            latency_ms=latency,
        )

    async def _remember(
        self, call: ModelCall, config: RunnableConfig, store: BaseStore
    ) -> ToolMessage:
        started = time.perf_counter()
        try:
            args = RememberFactArgs.model_validate(call.args)
        except ValidationError as error:
            # The schema has no user id on purpose; extra arguments are rejected here.
            text = invalid_args_error(
                call.name, error, RememberFactArgs, next_step=REMEMBER_FACT_NEXT_STEP
            )
            return tool_message(call_id=call.id, name=call.name, text=text, is_error=True)
        try:
            # InjectedStore: the store is passed as a trusted argument, never by the model.
            output = await self._remember_fact.ainvoke(
                {
                    "type": "tool_call",
                    "id": call.id,
                    "name": call.name,
                    "args": {"key": args.key, "value": args.value, "store": store},
                },
                config,
            )
        except Exception as error:
            logger.warning("remember_fact raised %s", type(error).__name__)
            cause = f"saving the fact raised {type(error).__name__}"
            text = internal_failure(call.name, cause, write=False)
            return tool_message(
                call_id=call.id,
                name=call.name,
                text=text,
                is_error=True,
                latency_ms=elapsed_ms(started, time.perf_counter()),
            )
        text = output.text if isinstance(output, ToolMessage) else str(output)
        return tool_message(
            call_id=call.id,
            name=call.name,
            text=str(text),
            is_error=False,
            latency_ms=elapsed_ms(started, time.perf_counter()),
        )

    async def propose(self, state: AgentState) -> AgentState:
        message = last_ai_message(state)
        if message is None or not proposable(message):
            raise GraphStateError("propose needs exactly one complete write tool call")
        call = message.tool_calls[0]
        try:
            pending = build_pending_action(call, state.get("last_write"))
        except InvalidProposalError as error:
            invalid = tool_message(
                call_id=call["id"] or "", name=call["name"], text=error.tool_error, is_error=True
            )
            return {"messages": [invalid], "pending_action": None}
        return {"pending_action": pending}

    async def approval(self, state: AgentState) -> Command[str]:
        # Only the interrupt: this node re-runs from the top on resume.
        pending = state.get("pending_action")
        if pending is None:
            raise GraphStateError("approval ran without a pending action")
        decision = interrupt(pending)
        return Command(
            goto=Node.EXECUTE_WRITE if approves(decision, pending) else Node.RECORD_DECLINE
        )

    async def execute_write(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """Run the approved write once; see `tool_results` for which failures are caught.

        A cancellation or process exit mid-call is not: the node then never finishes, so
        the next request re-runs it with the same idempotency key.
        """
        pending = state.get("pending_action")
        if pending is None:
            raise GraphStateError("execute_write ran without a pending action")
        args: dict[str, JsonValue] = {
            **pending.args,
            **injected_args(pending, user_id_from_config(config)),
        }
        with tool_activity(
            pending.tool_call_id, pending.tool.value, slow_after_s=self._slow_tool_notice_s
        ) as on_progress:
            started = time.perf_counter()
            result = await call_tool(
                self._tools, pending.tool.value, args, write=True, on_progress=on_progress
            )
            latency = elapsed_ms(started, time.perf_counter())
        error = parse_tool_error(result.text) if result.is_error else None
        last_write = WriteResult(
            action_id=pending.action_id,
            tool=pending.tool,
            idempotency_key=pending.idempotency_key,
            args=pending.args,
            ok=not result.is_error,
            result=None if result.is_error else _parse_result(result.text),
            error_code=error.code if error is not None else None,
            error_message=result.text if result.is_error else None,
        )
        reply = tool_message(
            call_id=pending.tool_call_id,
            name=pending.tool.value,
            text=result.text,
            is_error=result.is_error,
            latency_ms=latency,
        )
        return {"pending_action": None, "last_write": last_write, "messages": [reply]}

    async def record_decline(self, state: AgentState) -> AgentState:
        pending = state.get("pending_action")
        if pending is None:
            raise GraphStateError("record_decline ran without a pending action")
        declined = declined_message(call_id=pending.tool_call_id, name=pending.tool.value)
        return {"pending_action": None, "messages": [declined]}
