from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass
class PrepareState:
    running: bool = False
    ready: bool = True


class PrepareManager:
    def __init__(self, sleep_seconds: int) -> None:
        self._state = PrepareState()
        self._task: asyncio.Task[None] | None = None
        self._sleep_seconds = sleep_seconds

    @property
    def ready(self) -> bool:
        return self._state.ready and not self._state.running

    async def start(self, callback: Callable[[], Awaitable[None]] | None = None) -> bool:
        if self._state.running:
            return False
        self._state.running = True
        self._state.ready = False
        self._task = asyncio.create_task(self._background_prepare(callback))
        return True

    async def _background_prepare(self, callback: Callable[[], Awaitable[None]] | None) -> None:
        try:
            tasks: list[Awaitable[object]] = []
            if self._sleep_seconds > 0:
                tasks.append(asyncio.sleep(self._sleep_seconds))
            if callback is not None:
                tasks.append(callback())
            if tasks:
                await asyncio.gather(*tasks)
            self._state.ready = True
        finally:
            self._state.running = False
