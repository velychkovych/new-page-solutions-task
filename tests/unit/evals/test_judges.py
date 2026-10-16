"""The judge models: the LangChain judge from the model factory, and the scripted one.

No real model is called: the LangChain judge wraps a fake chat model here.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field, SecretStr
from tests.unit.evals.conftest import collected, evidence, make_case, response, score, turn_run

from evals.judges import (
    JUDGE_FIELDS,
    JUDGE_NOTE,
    JudgeError,
    Judges,
    LangChainJudge,
    ScriptedJudge,
    judge_settings,
    model_judges,
)
from evals.results import Outcome
from helpdesk.shared.settings import Settings


class Verdict(BaseModel):
    reason: str
    score: float


def _usage(input_tokens: int, output_tokens: int) -> dict[str, int]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


class StructuredFake(FakeMessagesListChatModel):
    """A fake chat model whose structured output is scripted, with token usage."""

    parsed: list[Any] = Field(default_factory=list)
    prompts: list[str] = Field(default_factory=list)

    def with_structured_output(  # type: ignore[override]
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable[Any, Any]:
        assert include_raw

        def answer(prompt: Any) -> dict[str, Any]:
            self.prompts.append(str(prompt))
            raw = AIMessage(content="", usage_metadata=_usage(120, 30))  # type: ignore[arg-type]
            parsed = self.parsed.pop(0)
            if parsed is None:
                return {"raw": raw, "parsed": None, "parsing_error": ValueError("no tool call")}
            return {"raw": raw, "parsed": schema(**parsed), "parsing_error": None}

        return RunnableLambda(answer)


def _fake(parsed: Sequence[Any], replies: Sequence[BaseMessage] = ()) -> StructuredFake:
    return StructuredFake(responses=list(replies), parsed=list(parsed), prompts=[])


async def test_structured_output_and_tokens_come_from_the_langchain_model() -> None:
    judge = LangChainJudge(_fake([{"reason": "fine", "score": 10}]), "claude-haiku-5-5")
    result, cost = await judge.a_generate("grade this", schema=Verdict)
    assert result == Verdict(reason="fine", score=10)
    assert (cost.input_tokens, cost.output_tokens) == (120, 30)
    assert judge.get_model_name() == "claude-haiku-5-5"


async def test_a_plain_text_answer_also_reports_tokens() -> None:
    reply = AIMessage(content="ok", usage_metadata=_usage(5, 1))  # type: ignore[arg-type]
    judge = LangChainJudge(_fake([], [reply]), "m")
    text, cost = await judge.a_generate("hello")
    assert text == "ok"
    assert (cost.input_tokens, cost.output_tokens) == (5, 1)


async def test_an_answer_without_the_schema_is_an_error() -> None:
    judge = LangChainJudge(_fake([None]), "m")
    with pytest.raises(JudgeError, match="no Verdict"):
        await judge.a_generate("grade this", schema=Verdict)


class TypeErrorFake(FakeMessagesListChatModel):
    """A chat model whose structured call raises TypeError somewhere inside the stack."""

    structured_calls: int = 0

    def with_structured_output(  # type: ignore[override]
        self, schema: Any, *, include_raw: bool = False, **kwargs: Any
    ) -> Runnable[Any, Any]:
        def answer(prompt: Any) -> Any:
            self.structured_calls += 1
            raise TypeError("unexpected response shape")

        return RunnableLambda(answer)


async def test_a_type_error_in_the_call_fails_once_instead_of_a_second_unstructured_call() -> None:
    free_text = AIMessage(content='{"score": 3, "reason": "free text"}')
    fake = TypeErrorFake(responses=[free_text])
    judge = LangChainJudge(fake, "m")
    # The method DeepEval calls: on a TypeError it would call a_generate again, schemaless.
    with pytest.raises(JudgeError, match="TypeError: unexpected response shape"):
        await judge.a_generate_with_schema("grade this", schema=Verdict)
    with pytest.raises(JudgeError):
        judge.generate_with_schema("grade this", schema=Verdict)
    assert fake.structured_calls == 2  # one per call above
    assert fake.i == 0  # the plain, schemaless model was never called


class TextAnswerAnthropic(ChatAnthropic):
    """The real ChatAnthropic structured-output path, with a canned reply instead of the API."""

    reply: AIMessage

    async def _agenerate(self, messages: Any, *args: Any, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.reply)])

    def _generate(self, messages: Any, *args: Any, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.reply)])


def _anthropic(model: str, reply: AIMessage) -> TextAnswerAnthropic:
    return TextAnswerAnthropic(model=model, api_key=SecretStr("sk-ant-dummy"), reply=reply)  # type: ignore[call-arg]


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5"])
@pytest.mark.filterwarnings("ignore:You are attempting to use structured output")
async def test_a_judge_that_answers_in_json_text_is_graded(model: str) -> None:
    # langchain-anthropic cannot force the tool call on these models, so the model may
    # answer the G-Eval prompt in text; DeepEval parses JSON text from any model.
    text = AIMessage(
        content='{"reason": "refuses politely", "score": 10}',
        usage_metadata=_usage(50, 7),  # type: ignore[arg-type]
    )
    judge = LangChainJudge(_anthropic(model, text), model)
    result, cost = await judge.a_generate("grade this", schema=Verdict)
    assert result == '{"reason": "refuses politely", "score": 10}'
    assert (cost.input_tokens, cost.output_tokens) == (50, 7)
    assert judge.generate("grade this", schema=Verdict)[0] == result
    setup = Judges(judge_for=lambda case, spec: judge, model=model, provider="a", note="")
    scored = await score(_judged_run(), setup)
    assert scored.outcome is Outcome.PASSED, scored.reasons
    assert (scored.judge_tokens.input, scored.judge_tokens.output) == (50, 7)


async def test_the_default_judge_still_answers_through_the_forced_tool_call() -> None:
    call = AIMessage(
        content="",
        tool_calls=[{"name": "Verdict", "args": {"reason": "ok", "score": 10}, "id": "t1"}],
        usage_metadata=_usage(30, 9),  # type: ignore[arg-type]
    )
    judge = LangChainJudge(_anthropic("claude-haiku-5-5", call), "claude-haiku-5-5")
    result, cost = await judge.a_generate("grade this", schema=Verdict)
    assert result == Verdict(reason="ok", score=10)
    assert (cost.input_tokens, cost.output_tokens) == (30, 9)


@pytest.mark.filterwarnings("ignore:You are attempting to use structured output")
async def test_an_empty_text_answer_is_a_judge_error_with_its_tokens() -> None:
    empty = AIMessage(content="", usage_metadata=_usage(40, 0))  # type: ignore[arg-type]
    judge = LangChainJudge(_anthropic("claude-opus-5-5", empty), "claude-opus-5-5")
    with pytest.raises(JudgeError) as raised:
        await judge.a_generate("grade this", schema=Verdict)
    assert (raised.value.cost.input_tokens, raised.value.cost.output_tokens) == (40, 0)


async def test_the_tokens_of_a_failed_judge_call_still_reach_the_report() -> None:
    fake = _fake([None])  # the call used 120/30 tokens but held no Verdict
    judge = LangChainJudge(fake, "m")
    setup = Judges(judge_for=lambda case, spec: judge, model="m", provider="a", note="")
    result = await score(_judged_run(), setup)
    assert result.outcome is Outcome.FAILED
    (metric,) = result.metrics
    assert metric.error is not None
    assert metric.error.startswith("judge error: JudgeError")
    assert (metric.tokens.input, metric.tokens.output) == (120, 30)
    assert (result.judge_tokens.input, result.judge_tokens.output) == (120, 30)


def _judged_run() -> Any:
    case = make_case(
        judge=[{"name": "refuses", "evaluation_steps": ["Check that it refuses."]}],
    )
    run = turn_run(response(reply="Sorry, I can only help with IT."), case=case)
    return collected(evidence([run], case=case))


async def test_the_judge_metric_records_the_judge_tokens_and_model() -> None:
    fake = _fake([{"reason": "refuses politely", "score": 10}])
    judge = LangChainJudge(fake, "claude-haiku-5-5")
    setup = Judges(
        judge_for=lambda case, spec: judge, model="claude-haiku-5-5", provider="a", note=""
    )
    result = await score(_judged_run(), setup)
    assert result.outcome is Outcome.PASSED, result.reasons
    (metric,) = result.metrics
    assert (metric.score, metric.reason) == (1.0, "refuses politely")
    assert metric.evaluation_model == "claude-haiku-5-5"
    assert (metric.tokens.input, metric.tokens.output) == (120, 30)
    assert (result.judge_tokens.input, result.judge_tokens.output) == (120, 30)
    # One call: the steps are given, so the judge never writes its own first.
    assert len(fake.prompts) == 1


async def test_strict_mode_fails_anything_below_the_top_score() -> None:
    judge = LangChainJudge(_fake([{"reason": "mostly", "score": 9}]), "m")
    setup = Judges(judge_for=lambda case, spec: judge, model="m", provider="a", note="")
    result = await score(_judged_run(), setup)
    assert result.outcome is Outcome.FAILED
    assert result.metrics[0].score == 0
    assert result.reasons == ["judge refuses [Conversational GEval]: mostly"]


async def test_deepeval_adds_no_retry_around_the_judge() -> None:
    calls: list[str] = []

    class Failing(ScriptedJudge):
        async def a_generate(self, prompt: str, schema: Any = None) -> Any:
            calls.append(prompt)
            raise ConnectionError("overloaded")

    setup = Judges(judge_for=lambda case, spec: Failing("pass"), model="m", provider="a", note="")
    result = await score(_judged_run(), setup)
    assert result.outcome is Outcome.FAILED
    # The Anthropic SDK's own retries (MODEL_MAX_RETRIES) are the only retry layer.
    assert len(calls) == 1


def test_the_live_judge_is_built_by_the_model_factory_from_settings() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        anthropic_api_key=SecretStr("sk-ant-dummy"),
        model_max_retries=4,
        model_timeout_s=12.5,
    )
    judges = model_judges(settings)
    judge = judges.judge_for(make_case(), None)  # type: ignore[arg-type]
    assert isinstance(judge, LangChainJudge)
    chat = judge.load_model()
    assert isinstance(chat, ChatAnthropic)
    assert chat.model == "claude-haiku-5-5"
    assert chat.max_retries == 4
    assert chat.temperature is None
    assert (judges.model, judges.provider, judges.note) == (
        "claude-haiku-5-5",
        "anthropic",
        JUDGE_NOTE,
    )
    assert "save budget" in JUDGE_NOTE
    assert "claude-opus-5-5" in JUDGE_NOTE


ENV_FILE = """\
MOCK_API_TOKEN=mock-token-from-dotenv
MOCK_ADMIN_TOKEN=admin-token-from-dotenv
CHAOS_RATE_LIMIT_RATE=0.7
CHAOS_ERROR_RATE=0.7
ANTHROPIC_API_KEY=sk-ant-dummy
JUDGE_MODEL_NAME=claude-opus-5-5
MODEL_MAX_RETRIES=4
"""


def test_the_judge_reads_only_its_own_settings_from_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in ENV_FILE.split():
        monkeypatch.delenv(name.split("=")[0], raising=False)
    monkeypatch.setenv("MODEL_TIMEOUT_S", "7")
    env_file = tmp_path / ".env"
    env_file.write_text(ENV_FILE + "MODEL_TIMEOUT_S=99\n", encoding="utf-8")
    settings = judge_settings(env_file)
    # The mock tokens stay in the file; CHAOS_* is not read, so it cannot fail validation.
    assert settings.mock_api_token is None
    assert settings.mock_admin_token is None
    assert settings.chaos_rate_limit_rate == 0.0
    assert settings.judge_model_name == "claude-opus-5-5"
    assert settings.model_max_retries == 4
    assert settings.anthropic_api_key == SecretStr("sk-ant-dummy")
    # The shell still wins over the file, as with Settings().
    assert settings.model_timeout_s == 7
    judge = model_judges(settings).judge_for(make_case(), None)  # type: ignore[arg-type]
    assert isinstance(judge, LangChainJudge)
    assert judge.get_model_name() == "claude-opus-5-5"


def test_the_judge_fields_are_settings_fields() -> None:
    assert JUDGE_FIELDS.issubset(Settings.model_fields)


async def test_the_scripted_judge_answers_with_the_scripted_verdict() -> None:
    passing, cost = await ScriptedJudge("pass").a_generate("p", schema=Verdict)
    failing, _ = await ScriptedJudge("fail").a_generate("p", schema=Verdict)
    assert passing == Verdict(reason="scripted verdict: pass", score=10)
    assert failing == Verdict(reason="scripted verdict: fail", score=0)
    assert (cost.input_tokens, cost.output_tokens) == (0, 0)
