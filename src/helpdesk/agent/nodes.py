"""Graph nodes (CONTRACTS §5.3).

No node has a RetryPolicy and no node retries a tool: the MCP server is the only
tool retry layer. Transport failures become UPSTREAM_UNAVAILABLE results at once.
"""

import asyncio
import json
import time
from collections.abc import Mapping

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolCall, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore
from langgraph.types import Command, interrupt
from pydantic import JsonValue, ValidationError

from helpdesk.agent.decision import approves
from helpdesk.agent.history import model_messages
from helpdesk.agent.memory import Clock, load_facts
from helpdesk.agent.prompt import system_message
from helpdesk.agent.prompt_caching import cacheable_system_message
from helpdesk.agent.proposal import InvalidProposalError, build_pending_action
from helpdesk.agent.routing import (
    LOCAL_TOOL_NAMES,
    ONE_CHANGE_AT_A_TIME,
    READ_TOOL_NAMES,
    Node,
    ToolCallBatch,
    classify_tool_calls,
    last_ai_message,
    unknown_tool_error,
)
from helpdesk.agent.run_config import user_id_from_config
from helpdesk.agent.state import AgentState
from helpdesk.agent.tool_messages import (
    TRANSPORT_FAILURE,
    USER_DECLINED,
    elapsed_ms,
    tool_message,
)
from helpdesk.agent.tool_provider import ToolProvider, ToolResult, ToolTransportError
from helpdesk.shared.approval import PendingAction, WriteResult
from helpdesk.shared.errors import ErrorCode, format_tool_error, parse_tool_error
from helpdesk.shared.memory import RememberFactArgs
from helpdesk.shared.tools import INJECTED_ARGS

IDEMPOTENCY_KEY_ARG = "idempotency_key"


class GraphStateError(RuntimeError):
    """A node ran in a state the topology should make impossible."""


def _call_id(call: ToolCall) -> str:
    return call["id"] or ""


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


class AgentNodes:
    def __init__(
        self,
        *,
        model: BaseChatModel,
        bound_model: Runnable[LanguageModelInput, AIMessage],
        tool_provider: ToolProvider,
        remember_fact_tool: BaseTool,
        history_token_budget: int,
        clock: Clock,
    ) -> None:
        self._model = model
        self._bound_model = bound_model
        self._tools = tool_provider
        self._remember_fact = remember_fact_tool
        self._history_token_budget = history_token_budget
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
        return {"messages": [response]}

    async def tools(
        self, state: AgentState, config: RunnableConfig, store: BaseStore
    ) -> AgentState:
        message = last_ai_message(state)
        if message is None:
            raise GraphStateError("tools ran without an AI message")
        calls = message.tool_calls
        if classify_tool_calls(calls) is ToolCallBatch.REJECTED:
            rejected: list[AnyMessage] = [
                tool_message(
                    call_id=_call_id(call),
                    name=call["name"],
                    text=ONE_CHANGE_AT_A_TIME,
                    is_error=True,
                )
                for call in calls
            ]
            return {"messages": rejected}
        results: list[AnyMessage] = list(
            await asyncio.gather(*(self._run_call(call, config, store) for call in calls))
        )
        return {"messages": results}

    async def _run_call(
        self, call: ToolCall, config: RunnableConfig, store: BaseStore
    ) -> ToolMessage:
        name = call["name"]
        if name in READ_TOOL_NAMES:
            return await self._call_provider(call["name"], _call_id(call), call["args"])
        if name in LOCAL_TOOL_NAMES:
            return await self._remember(call, config, store)
        return tool_message(
            call_id=_call_id(call), name=name, text=unknown_tool_error(name), is_error=True
        )

    async def _call_provider(
        self, name: str, call_id: str, args: Mapping[str, JsonValue]
    ) -> ToolMessage:
        started = time.perf_counter()
        result = await self._provider_result(name, args)
        return tool_message(
            call_id=call_id,
            name=name,
            text=result.text,
            is_error=result.is_error,
            latency_ms=elapsed_ms(started, time.perf_counter()),
        )

    async def _provider_result(self, name: str, args: Mapping[str, JsonValue]) -> ToolResult:
        try:
            return await self._tools.call(name, args)
        except ToolTransportError:
            return ToolResult(text=TRANSPORT_FAILURE, is_error=True)

    async def _remember(
        self, call: ToolCall, config: RunnableConfig, store: BaseStore
    ) -> ToolMessage:
        started = time.perf_counter()
        try:
            args = RememberFactArgs.model_validate(call["args"])
        except ValidationError:
            # The schema has no user id on purpose; extra arguments are rejected here.
            text = format_tool_error(
                ErrorCode.INVALID_ARGUMENT,
                "remember_fact takes only `key` (team, office_location, device, "
                "preferred_contact) and `value` (1-200 characters)",
            )
            return tool_message(call_id=_call_id(call), name=call["name"], text=text, is_error=True)
        # InjectedStore: the store is passed as a trusted argument, never by the model.
        output = await self._remember_fact.ainvoke(
            {
                "type": "tool_call",
                "id": _call_id(call),
                "name": call["name"],
                "args": {"key": args.key, "value": args.value, "store": store},
            },
            config,
        )
        text = output.text if isinstance(output, ToolMessage) else str(output)
        return tool_message(
            call_id=_call_id(call),
            name=call["name"],
            text=str(text),
            is_error=False,
            latency_ms=elapsed_ms(started, time.perf_counter()),
        )

    async def propose(self, state: AgentState) -> AgentState:
        message = last_ai_message(state)
        if message is None or len(message.tool_calls) != 1:
            raise GraphStateError("propose needs exactly one write tool call")
        call = message.tool_calls[0]
        try:
            pending = build_pending_action(call)
        except InvalidProposalError as error:
            invalid = tool_message(
                call_id=_call_id(call), name=call["name"], text=error.tool_error, is_error=True
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
        pending = state.get("pending_action")
        if pending is None:
            raise GraphStateError("execute_write ran without a pending action")
        args: dict[str, JsonValue] = {
            **pending.args,
            **injected_args(pending, user_id_from_config(config)),
        }
        started = time.perf_counter()
        result = await self._provider_result(pending.tool.value, args)
        latency = elapsed_ms(started, time.perf_counter())
        error = parse_tool_error(result.text) if result.is_error else None
        last_write = WriteResult(
            action_id=pending.action_id,
            tool=pending.tool,
            idempotency_key=pending.idempotency_key,
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
        declined = tool_message(
            call_id=pending.tool_call_id,
            name=pending.tool.value,
            text=USER_DECLINED,
            is_error=False,
        )
        return {"pending_action": None, "messages": [declined]}
