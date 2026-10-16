"""Clear, set and read user facts in the agent's Store: `python -c <this file> '<JSON>'`.

The live tier runs it inside the `agent-eval` container, the fake tier with the local
interpreter and the temporary STORE_DB_PATH, so both go through the agent's own store code
(helpdesk.agent.persistence and helpdesk.agent.memory) and the same Settings.
"""

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from helpdesk.agent.persistence import open_store
from helpdesk.shared.memory import FactKey, FactValue, RememberFactArgs, facts_namespace
from helpdesk.shared.settings import Settings


class FactsCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    clear: list[str] = []
    set: dict[str, dict[FactKey, FactValue]] = {}
    read: list[str] = []


class FactsResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    facts: dict[str, dict[str, str]] = {}


async def run(command: FactsCommand, store_path: Path, *, ttl_days: int) -> FactsResult:
    now = datetime.now(UTC)
    read: dict[str, dict[str, str]] = {}
    async with open_store(store_path, fact_ttl_days=ttl_days) as store:
        for user in command.clear:
            for key in FactKey:
                await store.adelete(facts_namespace(user), key.value)
        if command.set or command.read:
            # Imported only when needed: it pulls in LangChain, which costs ~0.4 s per call,
            # and most calls only clear.
            from helpdesk.agent.memory import load_facts, save_fact  # noqa: PLC0415

            for user, values in command.set.items():
                for key, value in values.items():
                    args = RememberFactArgs(key=key, value=value)
                    await save_fact(store, user, args, ttl_days=ttl_days, now=now)
            for user in command.read:
                facts = await load_facts(store, user, now=now)
                read[user] = {fact.key.value: fact.value for fact in facts}
    return FactsResult(facts=read)


def main(argv: list[str]) -> None:
    command = FactsCommand.model_validate_json(argv[0])
    settings = Settings()
    result = asyncio.run(run(command, settings.store_db_path, ttl_days=settings.fact_ttl_days))
    sys.stdout.write(result.model_dump_json() + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
