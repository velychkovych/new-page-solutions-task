"""POST /chat/stream: one turn as the SSE events of CONTRACTS §4.

The turn itself belongs to `ChatService` and is the same one /chat runs. This module only
turns what that run streams into events, in the contract's order. The run streams
LangGraph's "messages", "updates" and "custom" modes (`STREAM_MODES`), and the turn ends
with its `ChatResponse`.

- `token`: model text from the agent node, sent as it arrives. Only text content counts:
  `BaseMessage.text` keeps only `type: "text"` blocks, so Anthropic's tool_use blocks and
  `input_json_delta` argument fragments never become tokens. When the turn ends, the text
  the service adds itself (the proposal ask, re-ask, stale-decision reply, cut-off note,
  blank-answer fallback, truncation note) is sent as tokens too. So the tokens end with
  exactly /chat's reply.
- Model text in an intermediate reply (one that also calls a read tool, such as "Let me
  look for existing tickets.") is streamed as well. It is sent before anyone knows that
  tool calls follow, and the session view lists it as an assistant message too. /chat's
  reply holds only the last reply's text. So the tokens are the earlier replies' texts,
  each followed by a blank line, and then /chat's reply. Each reply's text is trimmed
  like /chat's.
- `tool_start` / `tool_end`: built with the same helpers as /chat's `tool_calls`
  (`transcript.tool_call_record`). `tool_start` comes from the node just before the
  provider call (`tool_activity`). A call that never ran (rejected, unparseable, unknown,
  remember_fact) gets its `tool_start` right before its `tool_end`. A declined write gets
  neither, because /chat does not list it either.
- `status`: the MCP server's retry progress, and one notice for a slow call.
- `approval_required`, `error`, `done`: from the final response, so `done.status` is
  /chat's status by construction.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langgraph.types import StreamMode

from helpdesk.agent.model_calls import ModelCall, model_calls
from helpdesk.agent.routing import Node
from helpdesk.agent.tool_activity import ToolStarted, ToolStatus
from helpdesk.agent.transcript import call_index, call_view, tool_call_record
from helpdesk.agent.turn_errors import UNEXPECTED_ERROR_REPLY
from helpdesk.shared.api import (
    ChatResponse,
    ChatStatus,
    DoneData,
    ErrorData,
    PendingActionView,
    SseEvent,
    StatusData,
    TokenData,
    ToolEndData,
    ToolStartData,
)

logger = logging.getLogger(__name__)

STREAM_MODES: Final[list[StreamMode]] = ["messages", "updates", "custom"]
# Between the texts of two model replies in one turn, and before text the service adds.
REPLY_SEPARATOR: Final = "\n\n"
_SEGMENT_KEY: Final = "langgraph_checkpoint_ns"
_NODE_KEY: Final = "langgraph_node"

type EventData = (
    TokenData | ToolStartData | ToolEndData | StatusData | PendingActionView | ErrorData | DoneData
)


@dataclass(frozen=True, slots=True)
class StreamEvent:
    name: SseEvent
    data: EventData


class TurnEvents:
    """Collects one streamed turn's events in a queue that the HTTP response drains.

    The queue decouples the graph run from the connection: the run never waits for the
    client. If the client disconnects, the run keeps writing here until it ends.
    """

    def __init__(self, trace_id: str) -> None:
        self._queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        self._trace_id = trace_id
        self._calls: dict[str, ModelCall] = {}
        # Calls with a tool_start and no tool_end yet. Ids are removed again at the end:
        # a model may reuse an id in a later reply, and that call gets its own pair.
        self._open_calls: set[str] = set()
        # One segment is one model call (one run of the agent node).
        self._segment: str | None = None
        self._segment_text = ""
        self._segment_chunked = False
        self._segment_whole = False
        # Whitespace held back until more text follows, so each reply is trimmed like /chat's.
        self._held_space = ""
        self._sent_text = False
        self._done = False

    # --- the turn ------------------------------------------------------------

    def begin(self, trace_id: str, history: Sequence[BaseMessage]) -> None:
        """The request span exists and the lock is held. `history` is the thread before
        the turn: an approved write's call lives in an earlier turn's reply."""
        self._trace_id = trace_id
        self._calls = call_index(history)

    def graph_chunk(self, mode: str, chunk: object) -> None:
        """One item of `astream(stream_mode=STREAM_MODES)`, already split into (mode, chunk)."""
        if self._done:
            return
        match mode:
            case "messages":
                self._on_message(chunk)
            case "updates":
                self._on_updates(chunk)
            case "custom":
                self._on_signal(chunk)
            case _:
                pass

    def finish(self, response: ChatResponse) -> None:
        if self._done:
            return
        if response.status is ChatStatus.ERROR:
            self._put(SseEvent.ERROR, ErrorData(message=response.reply))
        else:
            self._token(self._reply_tail(response.reply))
            if response.pending_action is not None:
                self._put(SseEvent.APPROVAL_REQUIRED, response.pending_action)
        self._end(response.status, response.trace_id)

    def close(self) -> None:
        """The turn ended without a response (a bug, or cancelled): still end the stream."""
        if self._done:
            return
        self._put(SseEvent.ERROR, ErrorData(message=UNEXPECTED_ERROR_REPLY))
        self._end(ChatStatus.ERROR, self._trace_id)

    async def __aiter__(self) -> AsyncIterator[StreamEvent]:
        while True:
            event = await self._queue.get()
            yield event
            if event.name is SseEvent.DONE:
                return

    # --- model text ----------------------------------------------------------

    def _on_message(self, chunk: object) -> None:
        if not (isinstance(chunk, tuple) and len(chunk) == 2):
            return
        message, metadata = chunk
        if not isinstance(message, AIMessage) or not isinstance(metadata, dict):
            return
        if metadata.get(_NODE_KEY) != Node.AGENT:
            return
        self._enter_segment(str(metadata.get(_SEGMENT_KEY, "")))
        if isinstance(message, AIMessageChunk):
            self._segment_chunked = True
        elif self._segment_chunked or self._segment_whole:
            # The finished reply of a streamed call (or the node's copy of it): its text
            # was already sent chunk by chunk.
            return
        else:
            # A model that does not stream sends its reply once, whole.
            self._segment_whole = True
        self._model_text(str(message.text))

    def _enter_segment(self, key: str) -> None:
        if key == self._segment:
            return
        self._segment = key
        self._segment_text = ""
        self._segment_chunked = False
        self._segment_whole = False
        self._held_space = ""

    def _model_text(self, text: str) -> None:
        if not self._segment_text:
            text = text.lstrip()
        body = text.rstrip()
        if not body:
            if self._segment_text:
                self._held_space += text
            return
        prefix = REPLY_SEPARATOR if not self._segment_text and self._sent_text else ""
        sent = self._held_space + body
        self._held_space = text[len(body) :]
        self._segment_text += sent
        self._token(prefix + sent)

    def _reply_tail(self, reply: str) -> str:
        """What the tokens still need so that they end with /chat's reply.

        The last reply's text is a prefix of /chat's reply when the service only appended
        to it (the proposal ask, the truncation note) or used it as it is.
        """
        if self._segment_text and reply.startswith(self._segment_text):
            return reply[len(self._segment_text) :]
        return (REPLY_SEPARATOR if self._sent_text else "") + reply

    # --- tools ---------------------------------------------------------------

    def _on_updates(self, chunk: object) -> None:
        if not isinstance(chunk, dict):
            return
        for update in chunk.values():
            # Interrupts come as `{"__interrupt__": (Interrupt, ...)}`; Command-only nodes
            # as None. Neither carries messages.
            if isinstance(update, dict):
                self._on_node_messages(update.get("messages"))

    def _on_node_messages(self, messages: object) -> None:
        if not isinstance(messages, Iterable) or isinstance(messages, str | bytes):
            return
        for message in messages:
            if isinstance(message, AIMessage):
                self._calls.update((call.id, call) for call in model_calls(message))
            elif isinstance(message, ToolMessage):
                self._tool_end(message)

    def _on_signal(self, signal: object) -> None:
        match signal:
            case ToolStarted(call_id=call_id, name=name):
                self._tool_start(call_id, name)
            case ToolStatus(message=message):
                self._put(SseEvent.STATUS, StatusData(message=message))
            case _:
                pass

    def _tool_start(self, call_id: str, fallback_name: str) -> None:
        if call_id in self._open_calls:
            return
        self._open_calls.add(call_id)
        name, args = call_view(call_id, self._calls, fallback_name)
        self._put(SseEvent.TOOL_START, ToolStartData(call_id=call_id, name=name, args=args))

    def _tool_end(self, message: ToolMessage) -> None:
        record = tool_call_record(message, self._calls)
        if record is None:
            return
        call_id = message.tool_call_id
        self._tool_start(call_id, record.name)
        self._open_calls.discard(call_id)
        self._put(
            SseEvent.TOOL_END,
            ToolEndData(
                call_id=call_id,
                result_summary=record.result_summary,
                error=record.error,
                latency_ms=record.latency_ms,
            ),
        )

    # --- output --------------------------------------------------------------

    def _token(self, text: str) -> None:
        if text:
            self._sent_text = True
            self._put(SseEvent.TOKEN, TokenData(text=text))

    def _end(self, status: ChatStatus, trace_id: str) -> None:
        self._put(SseEvent.DONE, DoneData(trace_id=trace_id, status=status))
        self._done = True

    def _put(self, name: SseEvent, data: EventData) -> None:
        if not self._done:
            self._queue.put_nowait(StreamEvent(name=name, data=data))
