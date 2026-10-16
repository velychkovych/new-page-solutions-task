"""`python -m evals.run --list`: the cases as a markdown table (pasted into the README)."""

from collections.abc import Sequence

from evals.cases import Case, MessageTurn
from evals.chaos_profiles import ChaosProfile, ChaosSetting, enables_chaos
from evals.report import md_cell


def _chaos_name(setting: ChaosSetting) -> str:
    return setting.value if isinstance(setting, ChaosProfile) else "custom"


def chaos_label(case: Case) -> str:
    settings = [case.setup.chaos, *(turn.chaos for turn in case.turns if turn.chaos is not None)]
    names = sorted({_chaos_name(s) for s in settings if enables_chaos(s)})
    return f" (chaos: {', '.join(names)})" if names else ""


def describe_turn(case: Case, turn_index: int) -> str:
    turn = case.turns[turn_index]
    text = f'"{turn.message}"' if isinstance(turn, MessageTurn) else f"[{turn.decision}]"
    who = []
    if turn.user_id is not None and turn.user_id != case.user_id:
        who.append(turn.user_id)
    if turn.session != case.turns[0].session or any(t.session != turn.session for t in case.turns):
        who.append(f"session {turn.session}")
    return f"{text} ({', '.join(who)})" if who else text


def render_case_table(cases: Sequence[Case]) -> str:
    lines = [
        "| Case | PLAN # | Title | User | Turns | Key expectations |",
        "|---|---|---|---|---|---|",
    ]
    for case in cases:
        turns = "<br>".join(
            f"{i + 1}. {md_cell(describe_turn(case, i))}" for i in range(len(case.turns))
        )
        keys = case.expect.set_keys()
        keys += [
            f"turn {i + 1}: {key}"
            for i, turn in enumerate(case.turns)
            for key in turn.expect.set_keys()
        ]
        chaos = chaos_label(case)
        lines.append(
            f"| `{case.id}` | {', '.join(map(str, case.plan_case))} | {md_cell(case.title)}{chaos}"
            f" | {case.user_id} | {turns} | {md_cell(', '.join(keys))} |"
        )
    return "\n".join(lines) + "\n"
