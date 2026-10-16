"""Run one case against a stack: set it up, send its turns over HTTP, collect the evidence,
check it.

Between cases: chaos off, mock reset, the facts of every user the case touches cleared,
the case's facts set, then its chaos. Every run gets fresh session ids. Chaos is turned
off again before the final reads, since GET /tickets is subject to chaos (/admin/* is not).
"""

import time
import uuid
from dataclasses import dataclass

import httpx
from pydantic import ValidationError

from evals.cases import Case, DecisionTurn, MessageTurn
from evals.chaos_profiles import CHAOS_OFF, resolve_chaos
from evals.checks import evaluate
from evals.evidence import CaseEvidence, TurnRun
from evals.facts_command import FactsCommand
from evals.results import CaseRunResult, TurnResult
from evals.stack import Stack, StackError
from evals.traces import TRACE_WAIT_S, TokenUsage, TraceIncompleteError, wait_for_trace
from helpdesk.shared.api import ChatRequest, ChatResponse, Decision, PendingActionView

# Longer than the agent's 90 s turn deadline, so the agent's own deadline reply arrives.
TURN_TIMEOUT_S = 150.0


class CaseFailedError(Exception):
    """The run cannot go on (an HTTP error, nothing to decide on); it fails, others go on."""


def new_session_id(case_id: str, label: str) -> str:
    return f"ev-{case_id}-{label}-{uuid.uuid4().hex[:8]}"


async def prepare(case: Case, stack: Stack) -> None:
    await stack.set_chaos(CHAOS_OFF)
    await stack.reset_mock()
    await stack.facts(FactsCommand(clear=sorted(case.users), set=case.setup.facts))
    await stack.set_chaos(resolve_chaos(case.setup.chaos))


def build_request(
    turn: MessageTurn | DecisionTurn,
    *,
    index: int,
    user_id: str,
    session_id: str,
    pending: PendingActionView | None,
) -> ChatRequest:
    if isinstance(turn, MessageTurn):
        return ChatRequest(user_id=user_id, session_id=session_id, message=turn.message)
    if pending is None:
        raise CaseFailedError(f"turn {index}: a {turn.decision} turn, but no action is pending")
    decision = Decision(action_id=pending.action_id, approve=turn.decision == "approve")
    return ChatRequest(user_id=user_id, session_id=session_id, decision=decision)


async def send(
    client: httpx.AsyncClient, agent_url: str, request: ChatRequest, index: int
) -> tuple[ChatResponse, float]:
    started = time.perf_counter()
    try:
        response = await client.post(
            f"{agent_url}/chat",
            content=request.model_dump_json(exclude_none=True),
            headers={"Content-Type": "application/json"},
            timeout=TURN_TIMEOUT_S,
        )
    except httpx.TimeoutException as exc:
        raise CaseFailedError(f"turn {index}: no response within {TURN_TIMEOUT_S:.0f} s") from exc
    except httpx.TransportError as exc:
        raise StackError(f"the agent at {agent_url} is unreachable: {type(exc).__name__}") from exc
    latency_ms = (time.perf_counter() - started) * 1000
    if response.status_code != 200:
        raise CaseFailedError(f"turn {index}: HTTP {response.status_code}: {response.text[:200]}")
    try:
        return ChatResponse.model_validate_json(response.text), latency_ms
    except ValidationError as exc:
        raise CaseFailedError(
            f"turn {index}: not a ChatResponse ({exc.error_count()} errors)"
        ) from exc


@dataclass
class _Progress:
    turns: list[TurnRun]
    results: list[TurnResult]


def _turn_result(
    turn: MessageTurn | DecisionTurn,
    request: ChatRequest,
    response: ChatResponse,
    *,
    index: int,
    latency_ms: float,
    tokens: TokenUsage,
) -> TurnResult:
    return TurnResult(
        index=index,
        user_id=request.user_id,
        session=turn.session,
        kind="message" if isinstance(turn, MessageTurn) else "decision",
        text=turn.message if isinstance(turn, MessageTurn) else turn.decision,
        status=response.status,
        latency_ms=round(latency_ms, 1),
        trace_id=response.trace_id,
        tokens=tokens,
    )


async def _run_turns(
    case: Case, stack: Stack, client: httpx.AsyncClient, progress: _Progress, trace_wait_s: float
) -> None:
    sessions: dict[str, str] = {}
    pending: dict[tuple[str, str], PendingActionView | None] = {}
    writes = (await stack.writes()).total
    for index, (turn, script) in enumerate(case.scripts(), start=1):
        if turn.chaos is not None:
            await stack.set_chaos(resolve_chaos(turn.chaos))
        if script is not None:
            await stack.use_script(script)
        user_id = case.turn_user(turn)
        session_id = sessions.setdefault(turn.session, new_session_id(case.id, turn.session))
        request = build_request(
            turn,
            index=index,
            user_id=user_id,
            session_id=session_id,
            pending=pending.get((user_id, session_id)),
        )
        response, latency_ms = await send(client, stack.agent_url, request, index)
        pending[(user_id, session_id)] = response.pending_action
        writes_after = (await stack.writes()).total
        try:
            trace = await wait_for_trace(stack.trace_dirs, response, timeout_s=trace_wait_s)
        except TraceIncompleteError:
            progress.results.append(
                _turn_result(
                    turn, request, response, index=index, latency_ms=latency_ms, tokens=TokenUsage()
                )
            )
            raise
        progress.results.append(
            _turn_result(
                turn, request, response, index=index, latency_ms=latency_ms, tokens=trace.tokens
            )
        )
        progress.turns.append(
            TurnRun(
                index=index,
                turn=turn,
                user_id=user_id,
                session_id=session_id,
                request=request,
                response=response,
                latency_ms=latency_ms,
                writes_before=writes,
                writes_after=writes_after,
                trace=trace,
            )
        )
        writes = writes_after


async def run_case(
    case: Case,
    stack: Stack,
    client: httpx.AsyncClient,
    *,
    run: int,
    trace_wait_s: float = TRACE_WAIT_S,
) -> CaseRunResult:
    """One run of `case`. Raises StackError when the stack itself fails."""
    started = time.perf_counter()
    progress = _Progress(turns=[], results=[])
    try:
        await prepare(case, stack)
        await _run_turns(case, stack, client, progress, trace_wait_s)
        await stack.set_chaos(CHAOS_OFF)
        writes = await stack.writes()
        readers = sorted(set(case.expect.facts) | set(case.expect.no_facts_for))
        facts = (await stack.facts(FactsCommand(read=readers))).facts if readers else {}
        reasons = evaluate(
            CaseEvidence(case=case, turns=progress.turns, writes=writes, facts=facts)
        )
    except (CaseFailedError, TraceIncompleteError) as exc:
        reasons = [str(exc)]
    traces = [turn.trace for turn in progress.turns]
    return CaseRunResult(
        case_id=case.id,
        plan_case=case.plan_case,
        run=run,
        passed=not reasons,
        reasons=reasons,
        turns=progress.results,
        duration_s=round(time.perf_counter() - started, 2),
        tokens=sum((result.tokens for result in progress.results), TokenUsage()),
        models=sorted({model for trace in traces for model in trace.models}),
        providers=sorted({provider for trace in traces for provider in trace.providers}),
    )
