"""Eval cases: one YAML file per case under evals/cases/, validated strictly on load.

A case names the PLAN "Minimum cases" it covers (`plan_case`), the user and the stack
setup (chaos, stored facts), the turns to send, and what must hold afterwards. `fake` is
the deterministic tier's model script: one script per turn that reaches the model.

The case files are the source of truth; evals.dataset turns each into a DeepEval golden.
`expect` holds the deterministic checks (evals.checks, one DeepEval metric per key).
`judge` holds the LLM-as-judge metrics, which grade only what the reply means; whether a
write, an approval, a retry or a memory leak happened is never theirs to decide.
`fake_judge` scripts the judge's verdicts for the deterministic tier (default: pass).
"""

import re
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    JsonValue,
    StringConstraints,
    Tag,
    ValidationError,
    model_validator,
)

from evals.chaos_profiles import ChaosProfile, ChaosSetting, enables_chaos
from helpdesk.agent.fake_model import ScriptedMessage
from helpdesk.mock.seed_data import SEED_USERS
from helpdesk.shared.api import ChatStatus
from helpdesk.shared.approval import classify_decision_text
from helpdesk.shared.errors import ErrorCode
from helpdesk.shared.ids import UserId
from helpdesk.shared.memory import FactKey, FactValue
from helpdesk.shared.tools import ToolName

CASES_DIR = Path(__file__).resolve().with_name("cases")
DEFAULT_SESSION = "main"

CaseId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9_]{1,32}$")]
MetricName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9_]{1,48}$")]
SessionLabel = Annotated[str, StringConstraints(pattern=r"^[a-z0-9_]{1,12}$")]
PlanCase = Annotated[int, Field(ge=1, le=13)]


def _compiles(pattern: str) -> str:
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"not a valid regular expression: {exc}") from exc
    return pattern


Pattern = Annotated[str, AfterValidator(_compiles)]
NonEmptyText = Annotated[str, StringConstraints(min_length=1)]


def _as_list(value: Any) -> Any:
    """A single value where a list is expected (`plan_case: 3`, `reply_matches: "..."`)."""
    return value if isinstance(value, list) else [value]


def _yaml_off(value: Any) -> Any:
    # YAML 1.1 reads a bare `off` as false, and PLAN writes `chaos: off`.
    return ChaosProfile.OFF.value if value is False else value


ListOf = BeforeValidator(_as_list)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _Expect(_Model):
    def set_keys(self) -> list[str]:
        """The checks the case file sets, in declaration order."""
        return [name for name in type(self).model_fields if name in self.model_fields_set]


class TurnExpect(_Expect):
    """Checks on one turn: its response, and the LLM calls in its trace."""

    status: ChatStatus | None = None
    reply_matches: Annotated[list[Pattern], ListOf] = []
    reply_not_matches: Annotated[list[Pattern], ListOf] = []
    # Case-insensitive substrings of the input the LLM spans recorded (system prompt with
    # the user's facts, history, tool results): how memory loading and isolation are
    # proven from the trace rather than from the reply.
    llm_input_contains: Annotated[list[NonEmptyText], ListOf] = []
    llm_input_not_contains: Annotated[list[NonEmptyText], ListOf] = []
    # Tool calls listed in the response that ran without an error. remember_fact runs
    # inside the agent, so it never has an mcp.tool span.
    tool_calls_include: Annotated[list[NonEmptyText], ListOf] = []


class _TurnBase(_Model):
    session: SessionLabel = DEFAULT_SESSION
    # Another user in the same case (memory isolation); default: the case's user.
    user_id: UserId | None = None
    # Applied just before this turn is sent.
    chaos: Annotated[ChaosSetting | None, BeforeValidator(_yaml_off)] = None
    expect: TurnExpect = TurnExpect()


class MessageTurn(_TurnBase):
    # Quote decision words in YAML: a bare `yes` is a boolean there.
    message: NonEmptyText

    @property
    def is_text_decision(self) -> bool:
        """`yes`/`no` and friends approve or decline without reaching the model."""
        return classify_decision_text(self.message) is not None


class DecisionTurn(_TurnBase):
    """The structured decision on the action pending in this turn's session."""

    decision: Literal["approve", "decline"]


def _turn_kind(value: Any) -> str | None:
    if isinstance(value, dict):
        if "decision" in value:
            return "decision"
        if "message" in value:
            return "message"
        return None
    return "decision" if isinstance(value, DecisionTurn) else "message"


type Turn = Annotated[
    Annotated[MessageTurn, Tag("message")] | Annotated[DecisionTurn, Tag("decision")],
    Discriminator(
        _turn_kind,
        custom_error_type="turn_kind",
        custom_error_message="a turn needs either `message` or `decision`",
    ),
]


def reaches_model(turn: MessageTurn | DecisionTurn) -> bool:
    """Whether the turn adds a user message, so the fake tier needs a new script for it."""
    return isinstance(turn, MessageTurn) and not turn.is_text_decision


class CaseExpect(_Expect):
    """Checks on the whole case run: every turn's response and trace, the mock, the store."""

    # A subsequence of the mcp.tool spans of all turns, by start time.
    tools_called_in_order_contains: list[ToolName] = []
    # Neither in a response's tool_calls nor in an mcp.tool span.
    tools_not_called: list[NonEmptyText] = []
    no_tool_calls: bool = False
    # A search_tickets mcp.tool span that carried a cursor (PLAN case 4).
    cursor_followed: bool = False
    # Every proposal of the case (and there is at least one) must satisfy these.
    proposed_tool: ToolName | None = None
    proposed_args_match: dict[str, JsonValue] = {}
    proposed_args_contain: dict[str, NonEmptyText] = {}
    # Case-insensitive substrings of the proposal's text arguments taken together, for
    # facts a model may put in either the title or the description.
    proposed_text_contains: Annotated[list[NonEmptyText], ListOf] = []
    # Distinct pending action ids across the case.
    max_proposals: Annotated[int, Field(ge=0)] | None = None
    # /admin/writes at the end: this total, and no idempotency key counted twice.
    writes_executed: Annotated[int, Field(ge=0)] | None = None
    no_write_before_approval: bool = False
    retries_seen: bool = False
    tool_error_codes_include: list[ErrorCode] = []
    no_invented_ticket_ids: bool = False
    final_status: ChatStatus | None = None
    # Stored facts at the end; each expected value is a case-insensitive substring.
    facts: dict[UserId, dict[FactKey, FactValue]] = {}
    no_facts_for: list[UserId] = []


class JudgeSpec(_Model):
    """One LLM-as-judge metric (DeepEval ConversationalGEval, strict: pass or fail)."""

    name: MetricName
    # Explicit steps, so the judge makes one call per metric instead of first writing
    # its own steps from a criteria text.
    evaluation_steps: Annotated[list[NonEmptyText], Field(min_length=1)]
    # Show the judge the tool calls (arguments and results), not only the messages.
    tools: bool = False


type FakeVerdict = Literal["pass", "fail"]


class CaseSetup(_Model):
    chaos: Annotated[ChaosSetting, BeforeValidator(_yaml_off)] = ChaosProfile.OFF
    # {user_id: {fact_key: value}}; the PLAN shorthand {fact_key: value} means the case user
    # and is expanded by Case before this model sees it.
    facts: dict[UserId, dict[FactKey, FactValue]] = {}


class Case(_Model):
    id: CaseId
    title: NonEmptyText
    plan_case: Annotated[list[PlanCase], ListOf, Field(min_length=1)]
    user_id: UserId
    # The DeepEval golden's scenario and expected outcome (default: the title).
    scenario: NonEmptyText | None = None
    expected_outcome: NonEmptyText | None = None
    setup: CaseSetup = CaseSetup()
    turns: Annotated[list[Turn], Field(min_length=1)]
    expect: CaseExpect = CaseExpect()
    judge: list[JudgeSpec] = []
    # The fake tier's judge verdict per judge metric; a metric left out passes.
    fake_judge: dict[MetricName, FakeVerdict] = {}
    # One model script per turn that reaches the model (see `reaches_model`). A decision
    # adds no user message, so it continues the script of the message before it.
    fake: list[Annotated[list[ScriptedMessage], Field(min_length=1)]]

    @model_validator(mode="before")
    @classmethod
    def _expand_fact_shorthand(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        setup = data.get("setup")
        if not isinstance(setup, dict) or not isinstance(setup.get("facts"), dict):
            return data
        facts: dict[Any, Any] = setup["facts"]
        values = list(facts.values())
        if values and all(isinstance(v, str) for v in values):
            facts = {data.get("user_id"): facts}
        elif not all(isinstance(v, dict) for v in values):
            raise ValueError(
                "setup.facts is either {fact_key: value} for the case user or"
                " {user_id: {fact_key: value}}, not a mix"
            )
        return {**data, "setup": {**setup, "facts": facts}}

    @model_validator(mode="after")
    def _check_turns(self) -> Self:
        model_turns = sum(reaches_model(turn) for turn in self.turns)
        if len(self.fake) != model_turns:
            raise ValueError(
                f"`fake` has {len(self.fake)} scripts, but {model_turns} turns reach the model"
                " (one script per message that is not a decision word)"
            )
        started: set[tuple[str, str]] = set()
        for index, turn in enumerate(self.turns, start=1):
            thread = (self.turn_user(turn), turn.session)
            if thread not in started and not reaches_model(turn):
                raise ValueError(
                    f"turn {index} decides on an action, but nothing was said in session"
                    f" {turn.session!r} of {thread[0]} before it"
                )
            started.add(thread)
        names = [spec.name for spec in self.judge]
        if len(set(names)) != len(names):
            raise ValueError(f"judge metric names must be unique: {names}")
        if unknown := sorted(set(self.fake_judge) - set(names)):
            raise ValueError(f"fake_judge names no judge metric of this case: {unknown}")
        return self

    def fake_verdict(self, metric: str) -> FakeVerdict:
        return self.fake_judge.get(metric, "pass")

    def turn_user(self, turn: MessageTurn | DecisionTurn) -> str:
        return turn.user_id or self.user_id

    @property
    def users(self) -> set[str]:
        """Everyone whose facts this case may touch: its facts are cleared before it runs."""
        return (
            {self.user_id, *SEED_USERS}
            | {self.turn_user(turn) for turn in self.turns}
            | set(self.setup.facts)
            | set(self.expect.facts)
            | set(self.expect.no_facts_for)
        )

    @property
    def uses_chaos(self) -> bool:
        return enables_chaos(self.setup.chaos) or any(
            enables_chaos(turn.chaos) for turn in self.turns
        )

    def scripts(self) -> Iterator[tuple[MessageTurn | DecisionTurn, list[ScriptedMessage] | None]]:
        """Each turn with the fake script it starts (None: it continues the previous one)."""
        fake = iter(self.fake)
        for turn in self.turns:
            yield turn, next(fake) if reaches_model(turn) else None


class CaseFileError(ValueError):
    def __init__(self, path: Path, problem: str) -> None:
        super().__init__(f"{path.name}: {problem}")
        self.path = path


def parse_case(text: str, path: Path) -> Case:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise CaseFileError(path, f"invalid YAML: {exc}") from exc
    try:
        case = Case.model_validate(data)
    except ValidationError as exc:
        raise CaseFileError(path, str(exc)) from exc
    if case.id != path.stem:
        raise CaseFileError(path, f"the file must be named after the case id ({case.id}.yaml)")
    return case


def load_case(path: Path) -> Case:
    return parse_case(path.read_text(encoding="utf-8"), path)


def load_cases(directory: Path = CASES_DIR) -> list[Case]:
    """Every case file, in PLAN case order."""
    cases = [load_case(path) for path in sorted(directory.glob("*.yaml"))]
    if not cases:
        raise CaseFileError(directory, "no *.yaml case files")
    return sorted(cases, key=lambda case: (min(case.plan_case), case.id))


def select_cases(
    cases: Sequence[Case], *, ids: Iterable[str] = (), chaos_only: bool = False
) -> list[Case]:
    """The cases to run, in file order. Unknown ids raise, so a typo never runs nothing."""
    wanted = list(ids)
    known = {case.id for case in cases}
    if unknown := [case_id for case_id in wanted if case_id not in known]:
        raise ValueError(f"unknown case id(s): {', '.join(unknown)}")
    selected = [case for case in cases if not wanted or case.id in wanted]
    if chaos_only:
        selected = [case for case in selected if case.uses_chaos]
    return selected
