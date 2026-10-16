"""User-facing replies for turns that end in an error (CONTRACTS §3.4).

The reply never contains exception text or a stack trace; the details go to the
error span and the log.
"""

import anthropic
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langgraph.errors import GraphRecursionError, NodeCancelledError

# Only true when no write ran in the turn: deadline_reply picks the honest variant.
DEADLINE_REPLY = (
    "Sorry, this request took too long and was stopped. Nothing was changed by the "
    "unfinished step. Please try again."
)
# The write call was in flight: the server may already have saved it. The next request
# re-runs it with the same idempotency key, so it is finished without a duplicate.
DEADLINE_WRITE_UNFINISHED_REPLY = (
    "Sorry, this request took too long and was stopped while your approved change was "
    "being saved. It may already have gone through. Send any message and I will finish "
    "it; it will not be made twice."
)
DEADLINE_AFTER_WRITE_REPLY = (
    "Your approved change was saved, but the rest of this request took too long and was "
    "stopped. Please ask again if you need anything else."
)
# A node cancelled itself (LangGraph raises NodeCancelledError for a CancelledError that
# came from inside a node, such as a tool client's own cancel scope). The same three
# cases as the deadline: the unfinished node wrote nothing, or it was the write.
INTERRUPTED_REPLY = (
    "Sorry, this request was interrupted before it finished. Nothing was changed by the "
    "unfinished step. Please try again."
)
INTERRUPTED_WRITE_UNFINISHED_REPLY = (
    "Sorry, this request was interrupted while your approved change was being saved. It "
    "may already have gone through. Send any message and I will finish it; it will not be "
    "made twice."
)
INTERRUPTED_AFTER_WRITE_REPLY = (
    "Your approved change was saved, but the rest of this request was interrupted. Please "
    "ask again if you need anything else."
)
RECURSION_REPLY = (
    "Sorry, I could not finish this request within my step limit. Please try again, "
    "perhaps with a more specific question."
)
RATE_LIMIT_REPLY = (
    "The AI model is receiving too many requests right now. Please try again in a minute."
)
OVERLOADED_REPLY = (
    "The AI model service is overloaded or unavailable at the moment. Please try again shortly."
)
AUTH_REPLY = (
    "The AI model service rejected the helpdesk's credentials. Please contact an administrator."
)
MODEL_TIMEOUT_REPLY = "The AI model service did not respond in time. Please try again."
CONTEXT_REPLY = "This conversation has grown too long for the model. Please start a new session."
MODEL_ERROR_REPLY = "The AI model service returned an error. Please try again."
# A bug: /chat answers 500, but a stream has already sent 200 and must still end with done.
UNEXPECTED_ERROR_REPLY = (
    "Sorry, something went wrong on our side while handling this request. Please reload "
    "the conversation before you try again."
)

# Most specific first: the langchain-anthropic classes subclass both hierarchies,
# and APITimeoutError subclasses APIConnectionError.
_MODEL_ERROR_REPLIES: tuple[tuple[tuple[type[BaseException], ...], str], ...] = (
    ((ContextOverflowError,), CONTEXT_REPLY),
    ((ModelRateLimitError, anthropic.RateLimitError), RATE_LIMIT_REPLY),
    (
        (
            ModelAuthenticationError,
            ModelPermissionDeniedError,
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
        ),
        AUTH_REPLY,
    ),
    ((ModelTimeoutError, anthropic.APITimeoutError), MODEL_TIMEOUT_REPLY),
    ((ModelConnectionError, anthropic.APIConnectionError), OVERLOADED_REPLY),
    (
        (ModelAPIError, anthropic.OverloadedError, anthropic.InternalServerError),
        OVERLOADED_REPLY,
    ),
    ((ModelError, anthropic.APIError), MODEL_ERROR_REPLY),
)


def model_error_reply(error: BaseException) -> str | None:
    """The reply for a model API error, or None if `error` is not one."""
    for types, reply in _MODEL_ERROR_REPLIES:
        if isinstance(error, types):
            return reply
    return None


def is_interruption(error: BaseException) -> bool:
    return isinstance(error, NodeCancelledError)


def turn_error_reply(error: BaseException) -> str | None:
    """The reply for a known turn failure, or None for a bug (which should surface as 500)."""
    if isinstance(error, GraphRecursionError):
        return RECURSION_REPLY
    return model_error_reply(error)
