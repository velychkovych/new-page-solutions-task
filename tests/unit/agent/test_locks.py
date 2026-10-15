"""Per-thread turn serialisation (CONTRACTS §3.3)."""

import asyncio
import gc

from helpdesk.agent.locks import ThreadLocks


async def test_same_thread_is_serialised_other_threads_are_not():
    locks = ThreadLocks()
    order: list[str] = []

    async def turn(key: str, name: str, hold_s: float) -> None:
        async with locks.hold(key):
            order.append(f"{name}:start")
            await asyncio.sleep(hold_s)
            order.append(f"{name}:end")

    await asyncio.gather(turn("t1", "a", 0.05), turn("t1", "b", 0.0), turn("t2", "c", 0.0))

    assert order.index("a:end") < order.index("b:start")
    assert order.index("c:end") < order.index("a:end")


async def test_idle_locks_are_released():
    locks = ThreadLocks()
    async with locks.hold("t1"):
        assert len(locks) == 1
    gc.collect()
    assert len(locks) == 0
