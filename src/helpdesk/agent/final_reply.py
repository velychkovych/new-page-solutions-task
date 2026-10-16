"""The reply for a turn that ends on the model's answer, also when that answer is unusable.

The user always gets a non-empty, honest reply: a blank answer gets a fixed fallback that
never claims anything was done, and a cut-off answer says that it was cut off.
"""

from enum import StrEnum

from langchain_core.messages import AIMessage

from helpdesk.agent.model_calls import is_truncated

EMPTY_REPLY = (
    "Sorry, I could not come up with a reply to that. Please try again, or rephrase your request."
)
EMPTY_AFTER_WRITE_REPLY = (
    "Your approved change was saved, but I could not write a reply about it. Ask me about "
    "the ticket to see its details."
)
EMPTY_AFTER_FAILED_WRITE_REPLY = (
    "I could not confirm that your approved change was saved, and I could not write a "
    "reply about it. Please check the ticket before you try again."
)
TRUNCATED_NOTE = (
    "(My reply was cut off because it reached the length limit. Ask me to continue, or "
    "for a shorter answer.)"
)


class TurnWrite(StrEnum):
    """What happened to an approved write during this turn."""

    NONE = "none"
    SAVED = "saved"
    FAILED = "failed"


_EMPTY_REPLIES: dict[TurnWrite, str] = {
    TurnWrite.NONE: EMPTY_REPLY,
    TurnWrite.SAVED: EMPTY_AFTER_WRITE_REPLY,
    TurnWrite.FAILED: EMPTY_AFTER_FAILED_WRITE_REPLY,
}


def final_reply(message: AIMessage | None, write: TurnWrite = TurnWrite.NONE) -> str:
    text = str(message.text).strip() if message is not None else ""
    if message is None or not text:
        return _EMPTY_REPLIES[write]
    if is_truncated(message):
        return f"{text}\n\n{TRUNCATED_NOTE}"
    return text
