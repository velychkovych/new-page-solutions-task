"""Span names and attribute keys shared by all services (CONTRACTS.md §6).

LLM spans come from OpenInference instrumentation and use its attribute names
(e.g. llm.token_count.prompt_details.cache_read); only our own spans are listed here.
"""

# Span names
SPAN_REQUEST = "agent.request"
SPAN_TOOL = "mcp.tool"
SPAN_APPROVAL = "agent.approval"
EVENT_RETRY = "retry.attempt"

# Request span
USER_ID = "helpdesk.user_id"
SESSION_ID = "helpdesk.session_id"
THREAD_ID = "helpdesk.thread_id"
REQUEST_STATUS = "helpdesk.status"
PENDING_ACTION_ID = "helpdesk.pending_action_id"

# Tool span
TOOL_NAME = "helpdesk.tool.name"
TOOL_ARGS = "helpdesk.tool.args"
TOOL_RESULT = "helpdesk.tool.result"
TOOL_ERROR_CODE = "helpdesk.tool.error_code"
TOOL_ATTEMPTS = "helpdesk.tool.attempts"

# Retry event
RETRY_ATTEMPT = "attempt"
RETRY_HTTP_STATUS = "http.status_code"
RETRY_WAIT_MS = "wait_ms"

# Approval span
APPROVAL_ACTION_ID = "helpdesk.approval.action_id"
APPROVAL_DECISION = "helpdesk.approval.decision"
APPROVAL_IDEMPOTENCY_KEY = "helpdesk.approval.idempotency_key"
