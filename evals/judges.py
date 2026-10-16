"""The judge models behind the LLM-as-judge metrics, as DeepEval custom models.

- `LangChainJudge` wraps the chat model from `helpdesk.agent.model.build_judge_model`
  (JUDGE_MODEL_PROVIDER / JUDGE_MODEL_NAME), so the judge gets the same limits as the
  agent: max_tokens, timeout and the SDK's own retries from Settings, no temperature.
  DeepEval's native AnthropicModel would turn the SDK retries off and retry with its own
  policy; a custom model gets no retry layer from DeepEval (it calls `a_generate` once).
- `ScriptedJudge` is the deterministic tier's judge: a fixed verdict from the case file.

DeepEval accepts `(result, cost)` from a custom model's `a_generate` and reads the token
counts off an `EvaluationCost`, which is how the judge's tokens reach the report.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from deepeval.models import DeepEvalBaseLLM
from deepeval.models.utils import EvaluationCost
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from evals.cases import Case, FakeVerdict, JudgeSpec
from evals.traces import TokenUsage
from helpdesk.agent.model import build_judge_model
from helpdesk.shared.settings import Settings

SCRIPTED_JUDGE_NAME = "scripted-judge"
# The G-Eval score scale: strict mode passes only the top score.
_TOP_SCORE = 10
_BOTTOM_SCORE = 0

type JudgeOutput = tuple[BaseModel | str, EvaluationCost]
type JudgeFor = Callable[[Case, JudgeSpec], DeepEvalBaseLLM]


class JudgeError(RuntimeError):
    """The judge's answer could not be used (no structured output)."""


def usage_cost(message: object) -> EvaluationCost:
    """The tokens of one judge call; the price is not known, so the cost value is None."""
    usage = message.usage_metadata if isinstance(message, AIMessage) else None
    if usage is None:
        return EvaluationCost(None, 0, 0)
    return EvaluationCost(None, usage["input_tokens"], usage["output_tokens"])


def cost_tokens(input_tokens: int | None, output_tokens: int | None) -> TokenUsage:
    return TokenUsage(input=input_tokens or 0, output=output_tokens or 0)


class LangChainJudge(DeepEvalBaseLLM):
    def __init__(self, chat_model: BaseChatModel, model_name: str) -> None:
        self._chat = chat_model
        # DeepEvalBaseLLM.__init__ is unannotated; it stores the name and calls load_model.
        super().__init__(model_name)  # type: ignore[no-untyped-call]

    # Annotated `-> DeepEvalBaseLLM` in the base class, but DeepEval's own models return
    # their client here, which is what this is.
    def load_model(self) -> BaseChatModel:  # type: ignore[override]
        return self._chat

    def get_model_name(self) -> str:
        return str(self.name)

    @staticmethod
    def _parsed(output: Any, schema: type[BaseModel]) -> JudgeOutput:
        parsed = output.get("parsed") if isinstance(output, dict) else None
        if not isinstance(parsed, schema):
            error = output.get("parsing_error") if isinstance(output, dict) else None
            raise JudgeError(f"no {schema.__name__} in the judge's answer: {error!r}")
        return parsed, usage_cost(output.get("raw"))

    # The base class declares `-> str`; DeepEval also takes (result, cost), which is the
    # only way a custom model reports its tokens.
    def generate(  # type: ignore[override]
        self, prompt: str, schema: type[BaseModel] | None = None
    ) -> JudgeOutput:
        if schema is None:
            message = self._chat.invoke(prompt)
            return message.text, usage_cost(message)
        runnable = self._chat.with_structured_output(schema, include_raw=True)
        return self._parsed(runnable.invoke(prompt), schema)

    async def a_generate(  # type: ignore[override]
        self, prompt: str, schema: type[BaseModel] | None = None
    ) -> JudgeOutput:
        if schema is None:
            message = await self._chat.ainvoke(prompt)
            return message.text, usage_cost(message)
        runnable = self._chat.with_structured_output(schema, include_raw=True)
        return self._parsed(await runnable.ainvoke(prompt), schema)


class ScriptedJudge(DeepEvalBaseLLM):
    """Answers every judge prompt with the case's scripted verdict, and keeps the prompts."""

    def __init__(self, verdict: FakeVerdict) -> None:
        self.verdict: FakeVerdict = verdict
        self.prompts: list[str] = []
        super().__init__(SCRIPTED_JUDGE_NAME)  # type: ignore[no-untyped-call]

    def load_model(self) -> ScriptedJudge:
        return self

    def get_model_name(self) -> str:
        return SCRIPTED_JUDGE_NAME

    def _answer(self, prompt: str, schema: type[BaseModel] | None) -> JudgeOutput:
        self.prompts.append(prompt)
        passed = self.verdict == "pass"
        reason = f"scripted verdict: {self.verdict}"
        if schema is None:
            return reason, EvaluationCost(None, 0, 0)
        score = _TOP_SCORE if passed else _BOTTOM_SCORE
        return schema.model_validate({"score": score, "reason": reason}), EvaluationCost(None, 0, 0)

    def generate(  # type: ignore[override]
        self, prompt: str, schema: type[BaseModel] | None = None
    ) -> JudgeOutput:
        return self._answer(prompt, schema)

    async def a_generate(  # type: ignore[override]
        self, prompt: str, schema: type[BaseModel] | None = None
    ) -> JudgeOutput:
        return self._answer(prompt, schema)


JUDGE_NOTE = (
    "The judge defaults to claude-haiku-5-5 to save budget; production would use"
    " claude-opus-5-5, a different model from the agent's."
)
FAKE_JUDGE_NOTE = "Fake tier: each judge verdict is scripted by the case's `fake_judge`."


@dataclass(frozen=True, slots=True)
class Judges:
    """Which judge model grades which case's judge metrics, and how to report it."""

    judge_for: JudgeFor
    model: str
    provider: str
    note: str


def scripted_judges() -> Judges:
    return Judges(
        judge_for=lambda case, spec: ScriptedJudge(case.fake_verdict(spec.name)),
        model=SCRIPTED_JUDGE_NAME,
        provider="fake",
        note=FAKE_JUDGE_NOTE,
    )


def model_judges(settings: Settings) -> Judges:
    """The live judge, from JUDGE_MODEL_*; raises MissingSecretError without its key."""
    judge = LangChainJudge(build_judge_model(settings), settings.judge_model_name)
    return Judges(
        judge_for=lambda case, spec: judge,
        model=settings.judge_model_name,
        provider=settings.judge_model_provider.value,
        note=JUDGE_NOTE,
    )


def _no_judge(case: Case, spec: JudgeSpec) -> DeepEvalBaseLLM:
    raise LookupError(f"{case.id} has judge metric {spec.name}, but no judge was set up")


def no_judges() -> Judges:
    """For a live run whose cases have no judge metric: no key needed, nothing to grade."""
    return Judges(
        judge_for=_no_judge, model="-", provider="-", note="No selected case has a judge metric."
    )
