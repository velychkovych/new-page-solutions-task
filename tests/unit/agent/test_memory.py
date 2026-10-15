"""Cross-session facts: per-user namespace, isolation, TTL and expiry (CONTRACTS §5.4)."""

from datetime import UTC, datetime, timedelta

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from helpdesk.agent.memory import (
    REMEMBER_FACT_DEFINITION,
    build_remember_fact_tool,
    load_facts,
    save_fact,
)
from helpdesk.agent.persistence import fact_ttl_config, open_store
from helpdesk.agent.run_config import build_run_config
from helpdesk.shared.errors import ErrorCode, parse_tool_error
from helpdesk.shared.memory import FactKey, RememberFactArgs, facts_namespace


def _config(user: str, session: str = "s1"):
    return build_run_config(user, session, recursion_limit=25)


async def _say(graph, user: str, text: str, session: str = "s1"):
    return await graph.ainvoke(
        {"messages": [HumanMessage(text)], "user_id": user}, _config(user, session)
    )


async def test_remember_fact_stores_under_the_request_user(
    make_graph, provider, fake_model, make_ai, settings
):
    model = fake_model(
        responses=[
            make_ai("", ("remember_fact", {"key": "team", "value": "finance"}, "m1")),
            make_ai("Noted."),
        ]
    )
    graph = await make_graph(model, provider)
    result = await _say(graph, "alice", "I'm on the finance team")

    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_message.content == "Saved team."
    async with open_store(settings.store_db_path, fact_ttl_days=90) as store:
        alice = await load_facts(store, "alice", now=datetime.now(UTC))
        bob = await load_facts(store, "bob", now=datetime.now(UTC))
        item = await store.aget(facts_namespace("alice"), "team")
    assert [(f.key, f.value) for f in alice] == [(FactKey.TEAM, "finance")]
    assert bob == []
    assert item is not None
    assert item.namespace == ("users", "alice", "facts")


async def test_remember_fact_rejects_a_model_supplied_user_id(
    make_graph, provider, fake_model, make_ai, settings
):
    forged = {"key": "team", "value": "finance", "user_id": "bob"}
    model = fake_model(responses=[make_ai("", ("remember_fact", forged, "m1")), make_ai("Hmm.")])
    graph = await make_graph(model, provider)
    result = await _say(graph, "alice", "I'm on finance")

    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert parse_tool_error(str(tool_message.content)).code is ErrorCode.INVALID_ARGUMENT
    async with open_store(settings.store_db_path, fact_ttl_days=90) as store:
        assert await load_facts(store, "bob", now=datetime.now(UTC)) == []
        assert await load_facts(store, "alice", now=datetime.now(UTC)) == []


async def test_facts_reach_the_users_prompt_in_a_new_session_only_for_that_user(
    make_graph, provider, fake_model, make_ai
):
    model = fake_model(
        responses=[
            make_ai("", ("remember_fact", {"key": "team", "value": "finance"}, "m1")),
            make_ai("Noted."),
            make_ai("Hello alice."),
            make_ai("Hello bob."),
        ]
    )
    graph = await make_graph(model, provider)
    await _say(graph, "alice", "I'm on the finance team", session="s1")
    await _say(graph, "alice", "hi", session="s2")
    await _say(graph, "bob", "hi", session="s1")

    alice_prompt = model.calls[2][0]
    bob_prompt = model.calls[3][0]
    assert isinstance(alice_prompt, SystemMessage)
    assert "- team: finance" in str(alice_prompt.content)
    assert "finance" not in str(bob_prompt.content)


async def test_user_b_cannot_read_user_a_facts(settings):
    now = datetime.now(UTC)
    async with open_store(settings.store_db_path, fact_ttl_days=90) as store:
        await save_fact(
            store,
            "alice",
            RememberFactArgs(key="device", value="ThinkPad X1"),
            ttl_days=90,
            now=now,
        )
        # "_" is a wildcard in SQL LIKE; a lookalike id must not match either.
        await save_fact(
            store, "a_b", RememberFactArgs(key="team", value="ops"), ttl_days=90, now=now
        )
        assert await load_facts(store, "bob", now=now) == []
        assert await load_facts(store, "axb", now=now) == []
        assert [f.value for f in await load_facts(store, "alice", now=now)] == ["ThinkPad X1"]


async def test_expired_facts_are_never_returned(settings):
    now = datetime.now(UTC)
    async with open_store(settings.store_db_path, fact_ttl_days=90) as store:
        await save_fact(
            store,
            "alice",
            RememberFactArgs(key="team", value="old team"),
            ttl_days=90,
            now=now - timedelta(days=91),
        )
        await save_fact(
            store, "alice", RememberFactArgs(key="device", value="Mac"), ttl_days=90, now=now
        )
        # The Store's own TTL has not swept the row, so the read must filter it.
        assert await store.aget(facts_namespace("alice"), "team") is not None
        facts = await load_facts(store, "alice", now=now)
    assert [f.key for f in facts] == [FactKey.DEVICE]


async def test_fact_ttl_is_set_in_minutes(settings):
    assert fact_ttl_config(90) == {"default_ttl": 129_600.0, "refresh_on_read": False}
    async with open_store(settings.store_db_path, fact_ttl_days=90) as store:
        await save_fact(
            store,
            "alice",
            RememberFactArgs(key="team", value="x"),
            ttl_days=90,
            now=datetime.now(UTC),
        )
        cursor = await store.conn.execute(
            "SELECT ttl_minutes, expires_at FROM store WHERE key = 'team'"
        )
        ttl_minutes, expires_at = await cursor.fetchone()
    assert ttl_minutes == 129_600
    assert expires_at is not None


def test_remember_fact_model_schema_has_no_user_id():
    properties = REMEMBER_FACT_DEFINITION.input_schema["properties"]
    assert set(properties) == {"key", "value"}
    tool = build_remember_fact_tool(ttl_days=90)
    assert set(tool.tool_call_schema.model_json_schema()["properties"]) == {"key", "value"}
