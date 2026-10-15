"""Snapshot of the tool list the MCP server advertises (CONTRACTS §2.2, §2.3).

An accidental change to a tool name, description or input schema fails this test. If the
change is intended (and CONTRACTS.md agrees), regenerate the snapshot and commit it:

    UPDATE_SNAPSHOTS=1 uv run pytest tests/unit/mcp_server/test_tool_schemas.py

then review the diff of tests/unit/mcp_server/snapshots/tool_list.json.
"""

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from tests.unit.mcp_server.conftest import SessionFactory

from helpdesk.shared.tools import (
    INJECTED_ARGS,
    AddCommentInput,
    CreateTicketInput,
    GetTicketArgs,
    SearchTicketsArgs,
    ToolName,
)

SNAPSHOT = Path(__file__).parent / "snapshots" / "tool_list.json"
CONTRACT_INPUTS: dict[str, type[BaseModel]] = {
    ToolName.SEARCH_TICKETS: SearchTicketsArgs,
    ToolName.GET_TICKET: GetTicketArgs,
    ToolName.CREATE_TICKET: CreateTicketInput,
    ToolName.ADD_COMMENT: AddCommentInput,
}


async def listed_tools(mcp_session: SessionFactory) -> list[dict[str, Any]]:
    async with mcp_session() as session:
        result = await session.list_tools()
    return [tool.model_dump(mode="json", by_alias=True, exclude_none=True) for tool in result.tools]


async def test_tool_list_matches_snapshot(mcp_session: SessionFactory) -> None:
    tools = await listed_tools(mcp_session)
    rendered = json.dumps(tools, indent=2, sort_keys=True) + "\n"
    if os.environ.get("UPDATE_SNAPSHOTS") == "1":
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(rendered, encoding="utf-8")
    assert rendered == SNAPSHOT.read_text(encoding="utf-8"), (
        "Tool list changed. If intended, regenerate: "
        "UPDATE_SNAPSHOTS=1 uv run pytest tests/unit/mcp_server/test_tool_schemas.py"
    )


async def test_input_schemas_are_the_shared_contract_models(mcp_session: SessionFactory) -> None:
    tools = {tool["name"]: tool for tool in await listed_tools(mcp_session)}
    assert set(tools) == {name.value for name in ToolName}
    for name, model in CONTRACT_INPUTS.items():
        assert tools[name]["inputSchema"] == model.model_json_schema(by_alias=True)


async def test_injected_args_are_required_in_the_mcp_schema(mcp_session: SessionFactory) -> None:
    tools = {tool["name"]: tool for tool in await listed_tools(mcp_session)}
    for name, injected in INJECTED_ARGS.items():
        assert set(injected) <= set(tools[name]["inputSchema"]["required"])


async def test_constraints_stay_visible(mcp_session: SessionFactory) -> None:
    tools = {tool["name"]: tool for tool in await listed_tools(mcp_session)}
    search = tools["search_tickets"]["inputSchema"]["properties"]
    assert search["page_size"] == {
        "default": 5,
        "maximum": 20,
        "minimum": 1,
        "title": "Page Size",
        "type": "integer",
    }
    ticket_id = tools["get_ticket"]["inputSchema"]["properties"]["ticket_id"]
    assert ticket_id["pattern"] == "^INC-[0-9]{4}$"
    assert tools["create_ticket"]["inputSchema"]["additionalProperties"] is False


async def test_descriptions_say_when_to_use(mcp_session: SessionFactory) -> None:
    tools = {tool["name"]: tool for tool in await listed_tools(mcp_session)}
    assert (
        "Always call before proposing a new ticket, to find duplicates."
        in tools["search_tickets"]["description"]
    )
    for tool in tools.values():
        assert "Use it" in tool["description"]
