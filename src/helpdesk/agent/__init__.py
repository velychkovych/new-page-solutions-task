"""Track C: LangGraph agent and its HTTP API.

`python -m helpdesk.agent` serves `api.create_app(...)` with the model from
`model.build_chat_model` and the MCP tool provider (`mcp_provider.McpToolProvider`).
Tests pass a fake tool provider and a fake model to `create_app` instead.
"""
