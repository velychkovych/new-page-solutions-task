"""The custom StateGraph of CONTRACTS §5.1 (deliberately not create_react_agent).

START → load_context → agent ─┬─ no tool calls ─────────────────→ END
                              ├─ reads / remember_fact → tools → agent
                              └─ one write → propose → approval ─┬─ approve → execute_write → agent
                                                                 └─ decline → record_decline → agent
"""

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from helpdesk.agent.memory import REMEMBER_FACT_DEFINITION, Clock, build_remember_fact_tool, utc_now
from helpdesk.agent.nodes import AgentNodes
from helpdesk.agent.prompt_caching import bind_tool_definitions
from helpdesk.agent.routing import Node, route_after_agent, route_after_propose
from helpdesk.agent.state import AgentState
from helpdesk.agent.tool_provider import ToolDefinition, ToolProvider
from helpdesk.shared.settings import Settings
from helpdesk.shared.tools import INJECTED_ARGS

type AgentGraph = CompiledStateGraph[AgentState, None, AgentState, AgentState]


class InjectedArgumentVisibleError(ValueError):
    """A tool definition shows the model an argument the agent must inject."""


def ensure_no_injected_args(definitions: Sequence[ToolDefinition]) -> None:
    """Fail at startup rather than let the model choose a user id or a key."""
    for definition in definitions:
        properties = definition.input_schema.get("properties")
        visible = set(properties) if isinstance(properties, dict) else set()
        for args in INJECTED_ARGS.values():
            leaked = visible.intersection(args)
            if leaked:
                raise InjectedArgumentVisibleError(
                    f"{definition.name} exposes injected argument(s) {sorted(leaked)}"
                )


async def build_graph(
    *,
    model: BaseChatModel,
    tool_provider: ToolProvider,
    settings: Settings,
    checkpointer: BaseCheckpointSaver[str],
    store: BaseStore,
    clock: Clock = utc_now,
) -> AgentGraph:
    definitions = [*await tool_provider.list_tools(), REMEMBER_FACT_DEFINITION]
    ensure_no_injected_args(definitions)
    nodes = AgentNodes(
        model=model,
        bound_model=bind_tool_definitions(model, definitions),
        tool_provider=tool_provider,
        tool_names=[definition.name for definition in definitions],
        remember_fact_tool=build_remember_fact_tool(ttl_days=settings.fact_ttl_days, clock=clock),
        history_token_budget=settings.history_token_budget,
        clock=clock,
    )

    builder = StateGraph(AgentState)
    builder.add_node(Node.LOAD_CONTEXT, nodes.load_context)
    builder.add_node(Node.AGENT, nodes.agent)
    builder.add_node(Node.TOOLS, nodes.tools)
    builder.add_node(Node.PROPOSE, nodes.propose)
    builder.add_node(
        Node.APPROVAL, nodes.approval, destinations=(Node.EXECUTE_WRITE, Node.RECORD_DECLINE)
    )
    builder.add_node(Node.EXECUTE_WRITE, nodes.execute_write)
    builder.add_node(Node.RECORD_DECLINE, nodes.record_decline)

    builder.add_edge(START, Node.LOAD_CONTEXT)
    builder.add_edge(Node.LOAD_CONTEXT, Node.AGENT)
    builder.add_conditional_edges(Node.AGENT, route_after_agent, [Node.TOOLS, Node.PROPOSE, END])
    builder.add_edge(Node.TOOLS, Node.AGENT)
    builder.add_conditional_edges(Node.PROPOSE, route_after_propose, [Node.APPROVAL, Node.AGENT])
    builder.add_edge(Node.EXECUTE_WRITE, Node.AGENT)
    builder.add_edge(Node.RECORD_DECLINE, Node.AGENT)
    return builder.compile(checkpointer=checkpointer, store=store)
