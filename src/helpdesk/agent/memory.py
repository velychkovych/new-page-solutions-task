"""Cross-session user facts in the LangGraph Store (CONTRACTS §5.4)."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from pydantic import ValidationError

from helpdesk.agent.persistence import MINUTES_PER_DAY
from helpdesk.agent.run_config import user_id_from_config
from helpdesk.agent.tool_provider import ToolDefinition
from helpdesk.shared.memory import (
    REMEMBER_FACT_TOOL,
    FactKey,
    RememberFactArgs,
    UserFact,
    facts_namespace,
)

type Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


REMEMBER_FACT_DESCRIPTION = (
    "Save a stable fact the user has told you about themselves, so later sessions can use "
    "it: their team, office_location, device or preferred_contact. Call it when the user "
    "states such a fact. Never guess a value, and never store anything else."
)

REMEMBER_FACT_DEFINITION = ToolDefinition(
    name=REMEMBER_FACT_TOOL,
    description=REMEMBER_FACT_DESCRIPTION,
    input_schema=RememberFactArgs.model_json_schema(),
)


async def save_fact(
    store: BaseStore, user_id: str, args: RememberFactArgs, *, ttl_days: int, now: datetime
) -> UserFact:
    fact = UserFact(
        key=args.key,
        value=args.value,
        updated_at=now,
        expires_at=now + timedelta(days=ttl_days),
    )
    await store.aput(
        facts_namespace(user_id),
        fact.key.value,
        fact.model_dump(mode="json"),
        ttl=float(ttl_days * MINUTES_PER_DAY),
    )
    return fact


async def load_facts(store: BaseStore, user_id: str, *, now: datetime) -> list[UserFact]:
    """The user's unexpired facts.

    Exact gets per key instead of a prefix search, so no other namespace can ever
    match. Expiry is filtered here because the SQLite Store keeps returning expired
    rows until a sweep deletes them.
    """
    namespace = facts_namespace(user_id)
    items = await asyncio.gather(
        *(store.aget(namespace, key.value, refresh_ttl=False) for key in FactKey)
    )
    facts: list[UserFact] = []
    for item in items:
        if item is None or tuple(item.namespace) != namespace:
            continue
        try:
            fact = UserFact.model_validate(item.value)
        except ValidationError:
            continue
        if fact.expires_at > now:
            facts.append(fact)
    return facts


def build_remember_fact_tool(*, ttl_days: int, clock: Clock = utc_now) -> BaseTool:
    """The remember_fact tool. Its user id comes from the run config, never from the model."""

    async def remember_fact(
        key: FactKey,
        value: str,
        store: Annotated[BaseStore, InjectedStore()],
        config: RunnableConfig,
    ) -> str:
        args = RememberFactArgs(key=key, value=value)
        fact = await save_fact(
            store, user_id_from_config(config), args, ttl_days=ttl_days, now=clock()
        )
        return f"Saved {fact.key.value}."

    return StructuredTool.from_function(
        coroutine=remember_fact,
        name=REMEMBER_FACT_TOOL,
        description=REMEMBER_FACT_DESCRIPTION,
    )
