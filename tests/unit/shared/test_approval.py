import pytest

from helpdesk.shared.approval import APPROVE_WORDS, DECLINE_WORDS, classify_decision_text


@pytest.mark.parametrize("text", ["approve", "approved", "yes", "y", "ok", "confirm"])
def test_approve_words(text: str) -> None:
    assert classify_decision_text(text) is True


@pytest.mark.parametrize("text", ["decline", "declined", "no", "n", "cancel", "reject"])
def test_decline_words(text: str) -> None:
    assert classify_decision_text(text) is False


@pytest.mark.parametrize("text", ["  YES ", "\tNo\n", "Ok"])
def test_trims_and_lowercases(text: str) -> None:
    assert classify_decision_text(text) is not None


@pytest.mark.parametrize(
    "text", ["yes please", "yes.", "sure", "nope", "", "approve it", "what does this do?"]
)
def test_anything_else_is_not_a_decision(text: str) -> None:
    assert classify_decision_text(text) is None


def test_word_lists_do_not_overlap() -> None:
    assert not APPROVE_WORDS & DECLINE_WORDS
