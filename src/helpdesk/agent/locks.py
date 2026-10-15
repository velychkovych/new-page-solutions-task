"""Per-thread serialisation of turns (CONTRACTS §3.3)."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from weakref import WeakValueDictionary


class ThreadLocks:
    """One asyncio.Lock per thread id, held for a whole turn.

    Weak values: a lock disappears once no request holds or waits on it, so the
    registry does not grow with every session ever seen.
    """

    def __init__(self) -> None:
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

    def _lock_for(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    @asynccontextmanager
    async def hold(self, key: str) -> AsyncIterator[None]:
        lock = self._lock_for(key)
        async with lock:
            yield

    def __len__(self) -> int:
        return len(self._locks)
