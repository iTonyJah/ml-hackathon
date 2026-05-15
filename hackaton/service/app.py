"""
RPC сервис хакатона.
predict() использует кэш активных пользователей + ML скоринг.
Latency оптимизирована - нет тяжёлых SQL запросов при predict.
"""

from __future__ import annotations

import logging

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import ValidationError

from hackaton.service.dto import (
    BatchEventsRequest,
    BatchShiftsRequest,
    BatchUsersRequest,
    PredictRequest,
)
from hackaton.service.prepare_manager import PrepareManager
from hackaton.service.repositories import Repository

REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["endpoint"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Latency of API requests", ["endpoint"])
LOGGER = logging.getLogger(__name__)


class HackatonRpcService:
    def __init__(self, repository: Repository, prepare: PrepareManager) -> None:
        self.repository = repository
        self.prepare_manager = prepare

    async def user(self, payload: dict) -> dict:
        REQUEST_COUNT.labels("user").inc()
        with REQUEST_LATENCY.labels("user").time():
            request = BatchUsersRequest.model_validate(payload)
            LOGGER.info("RPC user: batch_size=%s", len(request.items))
            accepted = await self.repository.upsert_users(request.items)
            return {"accepted": accepted}

    async def user_stat(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("user_stat").inc()
        with REQUEST_LATENCY.labels("user_stat").time():
            return {"count": await self.repository.count_table("users")}

    async def event(self, payload: dict) -> dict:
        REQUEST_COUNT.labels("event").inc()
        with REQUEST_LATENCY.labels("event").time():
            request = BatchEventsRequest.model_validate(payload)
            LOGGER.info("RPC event: batch_size=%s", len(request.items))
            accepted = await self.repository.insert_events(request.items)
            return {"accepted": accepted}

    async def event_stat(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("event_stat").inc()
        with REQUEST_LATENCY.labels("event_stat").time():
            return {"count": await self.repository.count_table("events")}

    async def shift(self, payload: dict) -> dict:
        REQUEST_COUNT.labels("shift").inc()
        with REQUEST_LATENCY.labels("shift").time():
            request = BatchShiftsRequest.model_validate(payload)
            LOGGER.info("RPC shift: batch_size=%s", len(request.items))
            accepted = await self.repository.upsert_shifts(request.items)
            return {"accepted": accepted}

    async def shift_stat(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("shift_stat").inc()
        with REQUEST_LATENCY.labels("shift_stat").time():
            return {"count": await self.repository.count_table("shifts")}

    async def prepare(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("prepare").inc()
        with REQUEST_LATENCY.labels("prepare").time():
            LOGGER.info("RPC prepare: вызван")
            started = await self.prepare_manager.start()
            if not started:
                return {"status": "already_running", "status_code": 409}
            return {"status": "started", "status_code": 200}

    async def ready(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("ready").inc()
        with REQUEST_LATENCY.labels("ready").time():
            if not self.prepare_manager.ready:
                return {"ready": False, "status_code": 425}
            return {"ready": True, "status_code": 200}

    async def predict(self, payload: dict) -> dict:
        REQUEST_COUNT.labels("predict").inc()
        with REQUEST_LATENCY.labels("predict").time():
            if not self.prepare_manager.ready:
                return {"user_ids": [], "status_code": 503, "detail": "модель в состоянии prepare"}
            try:
                request = PredictRequest.model_validate(payload)
            except ValidationError as exc:
                return {"user_ids": [], "status_code": 422, "detail": str(exc)}

            shift = request.shift
            pm = self.prepare_manager

            # Шаг 1: получаем кандидатов из локации + глобальный fallback
            candidates = pm.get_candidates(
                location_id=shift.location_id,
                need_mk=shift.need_mk,
                limit=300,  # топ-300 по активности - достаточно для recall, меньше шума
            )

            # До первого prepare кэш пуст - fallback к DB (только при старте сервиса)
            if not candidates:
                candidates = await self.repository.find_top_candidates(
                    location_id=str(shift.location_id),
                    need_mk=bool(shift.need_mk),
                    limit=request.limit,
                )
            if not candidates:
                candidates = await self.repository.fallback_candidates(limit=request.limit)
            if not candidates:
                return {"user_ids": [], "status_code": 400, "detail": "пользователи не загружены"}

            # Шаг 2: скорим ML моделью используя кэш данных пользователей
            model = pm.model
            if model.is_trained:
                shift_dict = {
                    "id": shift.id,
                    "start_at": shift.start_at.isoformat(),
                    "location_id": shift.location_id,
                    "task_type": shift.task_type,
                    "employer_id": shift.employer_id,
                    "workplace_id": shift.workplace_id,
                    "need_mk": shift.need_mk,
                    "id_differential": shift.id_differential,
                    "hours": shift.hours,
                    "reward": shift.reward,
                    "capacity": shift.capacity,
                }
                # Используем кэш пользователей - нет обращений к БД
                scored = model.predict_scores(candidates, pm._users_cache, shift_dict)
                top_candidates = [uid for uid, _ in scored[: request.limit]]
                LOGGER.info(
                    "predict итог: shift=%s loc=%s pool=%d top=%d model=trained",
                    shift.id,
                    shift.location_id,
                    len(candidates),
                    len(top_candidates),
                )
            else:
                # Fallback - просто топ активных
                top_candidates = candidates[: request.limit]
                LOGGER.warning(
                    "predict: shift=%s модель не обучена, используем ранжирование по активности",
                    shift.id,
                )

            return {"user_ids": top_candidates, "status_code": 200}

    async def health(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("health").inc()
        return {"status": "ok", "status_code": 200}

    async def metrics(self, _: dict | None = None) -> dict:
        REQUEST_COUNT.labels("metrics").inc()
        return {
            "content_type": CONTENT_TYPE_LATEST,
            "payload": generate_latest().decode("utf-8"),
            "status_code": 200,
        }
