"""The judge models behind the LLM-as-judge metrics, as DeepEval custom models.

- `LangChainJudge` wraps the chat model from `helpdesk.agent.model.build_judge_model`
  (JUDGE_MODEL_PROVIDER / JUDGE_MODEL_NAME), so the judge gets the same limits as the
  agent: max_tokens, timeout and the SDK's own retries from Settings, no temperature.
  DeepEval's native AnthropicModel would turn the SDK retries off and retry with its own
  policy; a custom model gets no retry layer from DeepEval (it calls `a_generate` once).
  DeepEval does call it a second time, without the schema, when it raises a TypeError,
  so every failure inside the call leaves as a JudgeError instead.
- A model that cannot be forced to call the schema tool (langchain-anthropic does not
  force it on claude-opus-5-5 or claude-sonnet-5-5) may answer the G-Eval prompt with
  JSON text. That text goes back to DeepEval, which parses JSON text from any model.
- `ScriptedJudge` is the deterministic tier's judge: a fixed verdict from the case file.

DeepEval accepts `(result, cost)` from a custom model's `a_generate` and reads the token
counts off an `EvaluationCost`, which is how the judge's tokens reach the report.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepeval.models import DeepEvalBaseLLM
from deepeval.models.utils import EvaluationCost
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from pydantic_settings import DotEnvSettingsSource

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
    """The judge's answer could not be used. `cost` is what the failed call still used."""

    def __init__(self, message: str, cost: EvaluationCost | None = None) -> None:
        super().__init__(message)
        self.cost = cost if cost is not None else EvaluationCost(None, 0, 0)


def usage_cost(message: object) -> EvaluationCost:
    """The tokens of one judge call; the price is not known, so the cost value is None."""
    usage = message.usage_metadata if isinstance(message, AIMessage) else None
    if usage is None:
        return EvaluationCost(None, 0, 0)
    return EvaluationCost(None, usage["input_tokens"], usage["output_tokens"])


def cost_tokens(input_tokens: int | None, output_tokens: int | None) -> TokenUsage:
    return TokenUsage(input=input_tokens or 0, output=output_tokens or 0)


class _LastMessage(BaseCallbackHandler):
    """Keeps the model's reply even when a parser after it raises, for its text and tokens."""

    run_inline = True

    def __init__(self) -> None:
        self.message: AIMessage | None = None

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                if isinstance(generation, ChatGeneration) and isinstance(
                    generation.message, AIMessage
                ):
                    self.message = generation.message


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
        raw = output.get("raw") if isinstance(output, dict) else None
        parsed = output.get("parsed") if isinstance(output, dict) else None
        if isinstance(parsed, schema):
            return parsed, usage_cost(raw)
        if isinstance(raw, AIMessage) and raw.text.strip():
            # No tool call, but an answer in text: DeepEval parses JSON out of it.
            return raw.text, usage_cost(raw)
        error = output.get("parsing_error") if isinstance(output, dict) else None
        raise JudgeError(f"no {schema.__name__} in the judge's answer: {error!r}", usage_cost(raw))

    @staticmethod
    def _failed(exc: Exception, last: _LastMessage) -> JudgeError:
        return JudgeError(
            f"judge call failed: {type(exc).__name__}: {exc}", usage_cost(last.message)
        )

    def _structured(self, schema: type[BaseModel]) -> Any:
        return self._chat.with_structured_output(schema, include_raw=True)

    # The base class declares `-> str`; DeepEval also takes (result, cost), which is the
    # only way a custom model reports its tokens.
    def generate(  # type: ignore[override]
        self, prompt: str, schema: type[BaseModel] | None = None
    ) -> JudgeOutput:
        last = _LastMessage()
        config = RunnableConfig(callbacks=[last])
        try:
            if schema is None:
                message = self._chat.invoke(prompt, config)
                return message.text, usage_cost(message)
            return self._parsed(self._structured(schema).invoke(prompt, config), schema)
        except OutputParserException as exc:
            return self._text_or_fail(exc, last)
        except TypeError as exc:
            raise self._failed(exc, last) from exc

    async def a_generate(  # type: ignore[override]
        self, prompt: str, schema: type[BaseModel] | None = None
    ) -> JudgeOutput:
        last = _LastMessage()
        config = RunnableConfig(callbacks=[last])
        try:
            if schema is None:
                message = await self._chat.ainvoke(prompt, config)
                return message.text, usage_cost(message)
            return self._parsed(await self._structured(schema).ainvoke(prompt, config), schema)
        except OutputParserException as exc:
            return self._text_or_fail(exc, last)
        except TypeError as exc:
            raise self._failed(exc, last) from exc

    @classmethod
    def _text_or_fail(cls, exc: OutputParserException, last: _LastMessage) -> JudgeOutput:
        """A model not forced to call the tool answered in text: hand that text on."""
        message = last.message
        if message is not None and message.text.strip():
            return message.text, usage_cost(message)
        raise cls._failed(exc, last) from exc


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


# What build_judge_model reads. The live runner takes only these from .env, so the mock
# tokens and every other value there stay out of this process (CONTRACTS §7) and nothing
# unrelated to the judge (CHAOS_*) can fail its validation.
JUDGE_FIELDS = frozenset(
    {
        "judge_model_provider",
        "judge_model_name",
        "model_max_tokens",
        "model_timeout_s",
        "model_max_retries",
        "anthropic_api_key",
        "openai_api_key",
        "aws_region",
    }
)


def judge_settings(env_file: Path) -> Settings:
    """Settings for the judge: the JUDGE_FIELDS of `env_file`, read by Settings' own parser.

    The shell's variables still win over the file, as they do in `Settings()`.
    """
    from_file = DotEnvSettingsSource(Settings, env_file=env_file)()
    shell = {name.lower() for name in os.environ}
    picked = {k: v for k, v in from_file.items() if k in JUDGE_FIELDS and k not in shell}
    return Settings(_env_file=None, **picked)


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
