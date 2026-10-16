"""Case files: the real ones load and cover PLAN's 13 cases; bad ones are rejected."""

from pathlib import Path
from typing import Any

import pytest
from tests.unit.evals.conftest import BASE_CASE, make_case

from evals.cases import (
    CASES_DIR,
    Case,
    CaseFileError,
    DecisionTurn,
    MessageTurn,
    load_cases,
    parse_case,
    select_cases,
)
from evals.chaos_profiles import ChaosProfile
from helpdesk.shared.memory import FactKey
from helpdesk.shared.tickets import ChaosConfig

PLAN_CASES = set(range(1, 14))


def _parse(yaml_text: str, name: str = "unit_case") -> Case:
    return parse_case(yaml_text, Path(f"{name}.yaml"))


# --- the real case files -----------------------------------------------------------------


def test_every_case_file_loads() -> None:
    cases = load_cases()
    assert len(cases) == len(list(CASES_DIR.glob("*.yaml")))


def test_the_cases_cover_every_plan_minimum_case() -> None:
    covered = {number for case in load_cases() for number in case.plan_case}
    assert covered == PLAN_CASES


def test_case_ids_are_unique() -> None:
    ids = [case.id for case in load_cases()]
    assert len(ids) == len(set(ids))


def test_cases_are_listed_in_plan_order() -> None:
    firsts = [min(case.plan_case) for case in load_cases()]
    assert firsts == sorted(firsts)


def test_some_case_approves_with_the_text_form() -> None:
    assert any(
        isinstance(turn, MessageTurn) and turn.message.strip().lower() == "yes"
        for case in load_cases()
        for turn in case.turns
    )


def test_some_case_approves_and_some_declines_with_the_structured_form() -> None:
    decisions = {
        turn.decision
        for case in load_cases()
        for turn in case.turns
        if isinstance(turn, DecisionTurn)
    }
    assert decisions == {"approve", "decline"}


def test_the_safety_cases_check_writes_from_the_mock() -> None:
    by_plan = {number: case for case in load_cases() for number in case.plan_case}
    for number in (1, 2, 3, 5, 6, 7, 8, 11, 12, 13):
        assert by_plan[number].expect.writes_executed is not None, number
    assert by_plan[6].expect.no_write_before_approval
    assert by_plan[11].expect.retries_seen
    assert by_plan[12].expect.no_invented_ticket_ids


# --- validation --------------------------------------------------------------------------


def test_plan_shorthand_facts_mean_the_case_user() -> None:
    case = make_case(setup={"facts": {"team": "finance"}})
    assert case.setup.facts == {"alice": {FactKey.TEAM: "finance"}}


def test_per_user_facts_are_kept() -> None:
    case = make_case(setup={"facts": {"bob": {"device": "ThinkPad"}}})
    assert case.setup.facts == {"bob": {FactKey.DEVICE: "ThinkPad"}}


def test_mixed_fact_forms_are_rejected() -> None:
    with pytest.raises(ValueError, match="not a mix"):
        make_case(setup={"facts": {"team": "finance", "bob": {"team": "x"}}})


def test_unknown_fact_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="team"):
        make_case(setup={"facts": {"salary": "a lot"}})


def test_yaml_off_is_the_off_profile() -> None:
    case = _parse(
        "id: unit_case\ntitle: t\nplan_case: 1\nuser_id: alice\nsetup: {chaos: off}\n"
        "turns: [{message: hi, chaos: moderate}]\nfake: [[{content: ok}]]\n"
    )
    assert case.setup.chaos is ChaosProfile.OFF
    assert case.turns[0].chaos is ChaosProfile.MODERATE
    assert case.uses_chaos


def test_an_explicit_chaos_config_is_accepted() -> None:
    config = {
        "enabled": True,
        "rate_limit_rate": 0.5,
        "error_rate": 0,
        "slow_rate": 0,
        "slow_ms": 0,
    }
    case = make_case(setup={"chaos": config})
    assert isinstance(case.setup.chaos, ChaosConfig)
    assert case.uses_chaos


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"surprise": 1}, "Extra inputs are not permitted"),
        ({"plan_case": 14}, "less than or equal to 13"),
        ({"id": "Bad-Id"}, "String should match pattern"),
        ({"turns": []}, "at least 1 item"),
        ({"turns": [{"msg": "hi"}]}, "either `message` or `decision`"),
        ({"turns": [{"message": True}]}, "valid string"),  # YAML's bare `yes`
        ({"turns": [{"decision": "approve"}], "fake": []}, "nothing was said in session"),
        ({"turns": [{"message": "hi"}, {"decision": "maybe"}]}, "'approve' or 'decline'"),
        ({"fake": []}, "`fake` has 0 scripts, but 1 turns reach the model"),
        ({"fake": [[]]}, "at least 1 item"),
        ({"fake": [[{"content": "x", "extra": 1}]]}, "Extra inputs"),
        ({"expect": {"tools_called_in_order_contains": ["close_ticket"]}}, "search_tickets"),
        ({"expect": {"tool_error_codes_include": ["OOPS"]}}, "NOT_FOUND"),
        ({"turns": [{"message": "hi", "expect": {"reply_matches": "(unclosed"}}]}, "regular expr"),
        ({"turns": [{"message": "hi", "expect": {"stats": "ok"}}]}, "Extra inputs"),
        ({"setup": {"chaos": "medium"}}, "moderate"),
    ],
)
def test_invalid_cases_are_rejected(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make_case(**overrides)


def test_a_text_decision_continues_the_previous_script() -> None:
    case = make_case(
        turns=[{"message": "My printer jams."}, {"message": "yes"}, {"message": "Thanks!"}],
        fake=[[{"content": "first"}], [{"content": "second"}]],
    )
    scripts = [None if script is None else script[0].content for _, script in case.scripts()]
    assert scripts == ["first", None, "second"]


def test_a_text_decision_cannot_open_a_session() -> None:
    with pytest.raises(ValueError, match="nothing was said"):
        make_case(turns=[{"message": "yes"}], fake=[])


def test_invalid_yaml_is_rejected() -> None:
    with pytest.raises(CaseFileError, match="invalid YAML"):
        _parse("id: [unclosed")


def test_the_file_must_be_named_after_the_case() -> None:
    text = (CASES_DIR / "status_question.yaml").read_text(encoding="utf-8")
    with pytest.raises(CaseFileError, match=r"status_question\.yaml"):
        _parse(text, name="renamed")


def test_load_cases_rejects_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(CaseFileError, match="no"):
        load_cases(tmp_path)


def test_users_include_the_seed_users_and_everyone_the_case_names() -> None:
    case = make_case(
        turns=[{"message": "hi"}, {"message": "hello", "user_id": "dave"}],
        fake=[[{"content": "a"}], [{"content": "b"}]],
        expect={"no_facts_for": ["erin"]},
    )
    assert case.users == {"alice", "bob", "carol", "dave", "erin"}


# --- selection ---------------------------------------------------------------------------


def test_select_cases_by_id_and_chaos() -> None:
    calm = make_case()
    stormy = make_case(id="stormy", setup={"chaos": "all_5xx"})
    assert select_cases([calm, stormy]) == [calm, stormy]
    assert select_cases([calm, stormy], chaos_only=True) == [stormy]
    assert select_cases([calm, stormy], ids=["unit_case"]) == [calm]
    with pytest.raises(ValueError, match="unknown case id"):
        select_cases([calm], ids=["nope"])


def test_real_chaos_selection_is_the_two_chaos_cases() -> None:
    ids = {case.id for case in select_cases(load_cases(), chaos_only=True)}
    assert ids == {"chaos_moderate_completes", "chaos_total_outage"}


def test_base_case_is_valid() -> None:
    assert Case.model_validate(BASE_CASE).id == "unit_case"
