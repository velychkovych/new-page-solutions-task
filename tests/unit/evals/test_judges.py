"""The judge models: the LangChain judge from the model factory, and the scripted one.

No real model is called: the LangChain judge wraps a fake chat model here.
"""

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field, SecretStr
from tests.unit.evals.conftest import collected, evidence, make_case, response, score, turn_run

from evals.judges import (
    JUDGE_NOTE,
    JudgeError,
    Judges,
    LangChainJudge,
    ScriptedJudge,
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


async def test_the_scripted_judge_answers_with_the_scripted_verdict() -> None:
    passing, cost = await ScriptedJudge("pass").a_generate("p", schema=Verdict)
    failing, _ = await ScriptedJudge("fail").a_generate("p", schema=Verdict)
    assert passing == Verdict(reason="scripted verdict: pass", score=10)
    assert failing == Verdict(reason="scripted verdict: fail", score=0)
    assert (cost.input_tokens, cost.output_tokens) == (0, 0)
