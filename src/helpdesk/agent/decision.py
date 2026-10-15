"""The value the API resumes the approval interrupt with (CONTRACTS §5.3)."""

from pydantic import BaseModel, ConfigDict, ValidationError

from helpdesk.shared.approval import PendingAction


class ResumeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action_id: str
    approve: bool


def resume_value(action_id: str, *, approve: bool) -> dict[str, str | bool]:
    return ResumeDecision(action_id=action_id, approve=approve).model_dump()


def approves(value: object, pending: PendingAction) -> bool:
    """Only a well-formed approval for this exact action executes the write.

    Anything else takes the decline path, so a malformed or mismatched resume can
    never cause a write.
    """
    try:
        decision = ResumeDecision.model_validate(value)
    except ValidationError:
        return False
    return decision.approve and decision.action_id == pending.action_id
