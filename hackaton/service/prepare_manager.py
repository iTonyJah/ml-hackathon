"""
PrepareManager — обучает ML модель и кэширует активных пользователей по локациям.
Кэш используется при predict для быстрого получения кандидатов без SQL JOIN.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import aiosqlite
import pandas as pd

from hackaton.service.ml_model import MLModel

LOGGER = logging.getLogger(__name__)


@dataclass
class PrepareState:
    running: bool = False
    ready: bool = True


class PrepareManager:
    def __init__(self, db_path: str, sleep_seconds: int = 0) -> None:
        self._db_path = db_path
        self._state = PrepareState()
        self._task: asyncio.Task[None] | None = None
        self._sleep_seconds = sleep_seconds
        self.model = MLModel()

        # Кэш данных пользователей: user_id -> dict
        self._users_cache: dict[str, dict] = {}
        # Все пользователи отсортированные по активности (для predict)
        self._global_top: list[str] = []

    @property
    def ready(self) -> bool:
        return self._state.ready and not self._state.running

    async def start(self) -> bool:
        if self._state.running:
            return False
        self._state.running = True
        self._state.ready = False
        self._task = asyncio.create_task(self._background_prepare())
        return True

    async def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute("SELECT * FROM users")
            users_rows = await cursor.fetchall()
            users = pd.DataFrame([dict(r) for r in users_rows]) if users_rows else pd.DataFrame()

            cursor = await db.execute("SELECT * FROM shifts")
            shifts_rows = await cursor.fetchall()
            shifts = pd.DataFrame([dict(r) for r in shifts_rows]) if shifts_rows else pd.DataFrame()

            cursor = await db.execute("SELECT * FROM events")
            events_rows = await cursor.fetchall()
            events = pd.DataFrame([dict(r) for r in events_rows]) if events_rows else pd.DataFrame()

        LOGGER.info(
            "Loaded from DB: users=%d, shifts=%d, events=%d", len(users), len(shifts), len(events)
        )
        return users, shifts, events

    def _build_cache(self, users: pd.DataFrame, events: pd.DataFrame) -> None:
        """Строим кэш пользователей по локациям (без ограничения TOP-N)."""
        LOGGER.info("Building user cache...")

        if not events.empty:
            activity = events.groupby("user_id").size().reset_index(name="n_events")
        else:
            activity = pd.DataFrame(columns=["user_id", "n_events"])

        users_with_activity = users.merge(activity, left_on="id", right_on="user_id", how="left")
        users_with_activity["n_events"] = users_with_activity["n_events"].fillna(0).astype(int)
        users_with_activity = users_with_activity.sort_values("n_events", ascending=False)

        self._users_cache = {
            str(row["id"]): {
                "id": str(row["id"]),
                "location_id": str(row["location_id"]),
                "is_strict_location": bool(row["is_strict_location"]),
                "has_mk": bool(row["has_mk"]),
            }
            for _, row in users_with_activity.iterrows()
        }

        # Кэш по локациям: ALL users (без ограничения — чтобы не пропустить аппликантов)
        from collections import defaultdict

        location_cache: dict[str, list[str]] = defaultdict(list)
        for _, row in users_with_activity.iterrows():
            location_cache[str(row["location_id"])].append(str(row["id"]))

        self._location_cache: dict[str, list[str]] = dict(location_cache)

        # Глобальный топ — для cross-location кандидатов (fallback)
        self._global_top = users_with_activity["id"].astype(str).tolist()

        LOGGER.info(
            "Cache built: %d locations, %d users total",
            len(self._location_cache),
            len(self._users_cache),
        )

    def get_candidates(self, location_id: str, need_mk: bool, limit: int) -> list[str]:
        """Возвращает кандидатов для предсказания.

        Для больших локаций (≥50 пользователей) — только пользователи в локации.
        Это критически важно: кросс-локационные пользователи из global_top вытесняют
        реальных кандидатов с более высоким location_match-скором.
        Для маленьких локаций (<50) добавляем global fallback чтобы набрать пул.
        """
        loc_candidates = self._location_cache.get(str(location_id), [])

        if len(loc_candidates) < 50:
            # Маленькая локация — добавляем глобальный топ как fallback
            existing = set(loc_candidates)
            extra = [u for u in self._global_top if u not in existing][:500]
            candidates = loc_candidates + extra
        else:
            # Большая локация — только свои пользователи (без global шума)
            candidates = loc_candidates

        return candidates[:limit]

    async def _background_prepare(self) -> None:
        try:
            LOGGER.info("PrepareManager: starting background prepare")
            users, shifts, events = await self._load_data()

            if users.empty:
                LOGGER.warning("No users loaded, skipping prepare")
                self._state.ready = True
                return

            if "ts" in events.columns:
                events["ts"] = pd.to_datetime(events["ts"], utc=True, errors="coerce")
            if "start_at" in shifts.columns:
                shifts["start_at"] = pd.to_datetime(shifts["start_at"], utc=True, errors="coerce")

            # Строим кэш активных пользователей (быстро, в основном потоке)
            self._build_cache(users, events)

            # Обучаем модель в executor чтобы не блокировать event loop
            if not events.empty and not shifts.empty:
                loop = asyncio.get_event_loop()

                def _train_and_cache() -> None:
                    self.model.train(events, shifts, users)
                    # После обучения строим векторизованный кэш для быстрого inference
                    self.model.build_inference_cache(users)

                await loop.run_in_executor(None, _train_and_cache)
            else:
                LOGGER.warning("Not enough data to train model")

            LOGGER.info("PrepareManager: prepare complete, model ready=%s", self.model.is_trained)
            self._state.ready = True

        except Exception as exc:
            LOGGER.exception("PrepareManager: prepare failed: %s", exc)
            self._state.ready = True
        finally:
            self._state.running = False
