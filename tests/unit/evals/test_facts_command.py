"""The facts command, run in process against a temporary Store file (no subprocess)."""

from pathlib import Path

from evals.facts_command import FactsCommand, run
from helpdesk.shared.memory import FactKey


async def test_set_read_and_clear_facts(tmp_path: Path) -> None:
    store = tmp_path / "store.sqlite"
    command = FactsCommand(
        set={"alice": {FactKey.TEAM: "finance", FactKey.DEVICE: "ThinkPad"}}, read=["alice", "bob"]
    )
    result = await run(command, store, ttl_days=90)
    assert result.facts == {"alice": {"team": "finance", "device": "ThinkPad"}, "bob": {}}

    cleared = await run(FactsCommand(clear=["alice"], read=["alice"]), store, ttl_days=90)
    assert cleared.facts == {"alice": {}}


async def test_clearing_one_user_keeps_the_others(tmp_path: Path) -> None:
    store = tmp_path / "store.sqlite"
    await run(
        FactsCommand(set={"alice": {FactKey.TEAM: "a"}, "bob": {FactKey.TEAM: "b"}}),
        store,
        ttl_days=90,
    )
    result = await run(FactsCommand(clear=["alice"], read=["alice", "bob"]), store, ttl_days=90)
    assert result.facts == {"alice": {}, "bob": {"team": "b"}}


async def test_a_clear_only_command_reads_nothing(tmp_path: Path) -> None:
    result = await run(FactsCommand(clear=["alice"]), tmp_path / "store.sqlite", ttl_days=90)
    assert result.facts == {}
