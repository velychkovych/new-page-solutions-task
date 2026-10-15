import pytest
from pydantic import TypeAdapter, ValidationError

from helpdesk.shared.ids import IdempotencyKey, TicketId, UserId, is_ticket_id, thread_id


@pytest.mark.parametrize("value", ["INC-0001", "INC-9999"])
def test_valid_ticket_ids(value: str) -> None:
    assert is_ticket_id(value)
    assert TypeAdapter(TicketId).validate_python(value) == value


@pytest.mark.parametrize(
    "value", ["INC-1", "inc-0001", "INC-00001", "INC-0001\n", "INC-١٢٣٤", " INC-0001", ""]
)
def test_invalid_ticket_ids(value: str) -> None:
    assert not is_ticket_id(value)
    with pytest.raises(ValidationError):
        TypeAdapter(TicketId).validate_python(value)


@pytest.mark.parametrize("value", ["alice", "user_1", "A-b_9", "x" * 64])
def test_valid_user_ids(value: str) -> None:
    assert TypeAdapter(UserId).validate_python(value) == value


@pytest.mark.parametrize("value", ["", "x" * 65, "a:b", "a b", "ä", "alice\n"])
def test_invalid_user_ids(value: str) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(UserId).validate_python(value)


def test_thread_id_joins_validated_parts() -> None:
    assert thread_id("alice", "s1") == "alice:s1"


@pytest.mark.parametrize(("user_id", "session_id"), [("a:b", "c"), ("a", "b:c"), ("", "s")])
def test_thread_id_rejects_ids_that_could_collide(user_id: str, session_id: str) -> None:
    with pytest.raises(ValueError, match="must match"):
        thread_id(user_id, session_id)


@pytest.mark.parametrize("value", ["", "has space", "k" * 256, "tab\tkey"])
def test_invalid_idempotency_keys(value: str) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(IdempotencyKey).validate_python(value)


def test_uuid4_is_a_valid_idempotency_key() -> None:
    key = "3f1c2b9e-8a7d-4c6b-9f0e-1a2b3c4d5e6f"
    assert TypeAdapter(IdempotencyKey).validate_python(key) == key
