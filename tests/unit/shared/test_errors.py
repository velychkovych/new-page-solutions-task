import pytest

from helpdesk.shared.errors import (
    DEFAULT_NEXT_STEP,
    ErrorCode,
    ToolError,
    format_tool_error,
    parse_tool_error,
)


def test_format_matches_contract_shape() -> None:
    text = format_tool_error(ErrorCode.NOT_FOUND, "Ticket INC-0042 does not exist.", "Ask the user")
    assert text == "ERROR NOT_FOUND: Ticket INC-0042 does not exist. Next step: Ask the user."


def test_format_uses_default_next_step() -> None:
    text = format_tool_error(ErrorCode.AUTH, "The ticketing system rejected our credentials")
    assert text.endswith(f"Next step: {DEFAULT_NEXT_STEP[ErrorCode.AUTH]}.")


def test_every_code_has_a_default_next_step() -> None:
    assert set(DEFAULT_NEXT_STEP) == set(ErrorCode)


@pytest.mark.parametrize("code", list(ErrorCode))
def test_round_trip(code: ErrorCode) -> None:
    text = format_tool_error(code, "Something happened. Twice", "Do a thing")
    assert parse_tool_error(text) == ToolError(
        code=code, what="Something happened. Twice", next_step="Do a thing"
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "all good",
        "ERROR BOGUS: x. Next step: y.",
        "Error executing tool get_ticket: ERROR NOT_FOUND: x. Next step: y.",
        "ERROR NOT_FOUND: x.",
    ],
)
def test_parse_rejects_non_contract_text(text: str) -> None:
    assert parse_tool_error(text) is None
