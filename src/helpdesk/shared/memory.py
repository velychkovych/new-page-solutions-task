"""Cross-session user facts (CONTRACTS.md §5.4)."""

from enum import StrEnum
from typing import Annotated

from pydantic import AwareDatetime, BaseModel, ConfigDict, StringConstraints

from helpdesk.shared.ids import is_principal_id


class FactKey(StrEnum):
    TEAM = "team"
    OFFICE_LOCATION = "office_location"
    DEVICE = "device"
    PREFERRED_CONTACT = "preferred_contact"


REMEMBER_FACT_TOOL = "remember_fact"

FactValue = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class RememberFactArgs(BaseModel):
    """Model-visible arguments of remember_fact. There is no user_id here on purpose."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: FactKey
    value: FactValue


class UserFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: FactKey
    value: FactValue
    updated_at: AwareDatetime
    expires_at: AwareDatetime


def facts_namespace(user_id: str) -> tuple[str, str, str]:
    if not is_principal_id(user_id):
        raise ValueError("invalid user_id")
    return ("users", user_id, "facts")
