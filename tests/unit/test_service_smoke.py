from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from hackaton.service.app import HackatonRpcService
from hackaton.service.db import init_db_for
from hackaton.service.prepare_manager import PrepareManager
from hackaton.service.repositories import Repository


def build_service(tmp_path: Path) -> HackatonRpcService:
    db_path = str(tmp_path / "test.db")
    asyncio.run(init_db_for(db_path))
    repository = Repository(db_path=db_path)
    prepare = PrepareManager(sleep_seconds=0)
    return HackatonRpcService(repository=repository, prepare=prepare)


def test_health_rpc(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    response = asyncio.run(service.health(None))
    assert response["status_code"] == 200
    assert response["status"] == "ok"


def test_user_event_shift_and_predict_flow(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    users_payload = {
        "items": [
            {
                "id": "u1",
                "location_id": "loc-1",
                "is_strict_location": True,
                "has_mk": True,
            }
        ]
    }
    shifts_payload = {
        "items": [
            {
                "id": "s1",
                "start_at": datetime.now(tz=UTC).isoformat(),
                "location_id": "loc-1",
                "task_type": "loader",
                "employer_id": "emp-1",
                "workplace_id": "wp-1",
                "need_mk": True,
                "id_differential": False,
                "hours": 8,
                "reward": 1200.0,
                "capacity": 2,
            }
        ]
    }
    events_payload = {
        "items": [
            {
                "id": str(uuid4()),
                "shift_id": "s1",
                "user_id": "u1",
                "interaction": "VIEW",
                "ts": datetime.now(tz=UTC).isoformat(),
            }
        ]
    }

    assert asyncio.run(service.user(users_payload))["accepted"] == 1
    assert asyncio.run(service.shift(shifts_payload))["accepted"] == 1
    assert asyncio.run(service.event(events_payload))["accepted"] == 1

    predict_response = asyncio.run(
        service.predict(
            {
                "shift": shifts_payload["items"][0],
                "limit": 10,
            }
        )
    )
    assert predict_response["status_code"] == 200
    assert predict_response["user_ids"]


def test_prepare_builds_features_and_history_changes_ranking(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    now = datetime.now(tz=UTC).isoformat()
    users_payload = {
        "items": [
            {
                "id": "u_alpha",
                "location_id": "loc-1",
                "is_strict_location": True,
                "has_mk": True,
            },
            {
                "id": "u_best",
                "location_id": "loc-2",
                "is_strict_location": False,
                "has_mk": True,
            },
        ]
    }
    shifts_payload = {
        "items": [
            {
                "id": "history-shift",
                "start_at": now,
                "location_id": "loc-2",
                "task_type": "picker",
                "employer_id": "emp-1",
                "workplace_id": "wp-1",
                "need_mk": True,
                "id_differential": False,
                "hours": 8,
                "reward": 1600.0,
                "capacity": 2,
            }
        ]
    }
    events_payload = {
        "items": [
            {
                "id": str(uuid4()),
                "shift_id": "history-shift",
                "user_id": "u_best",
                "interaction": "FINISHED",
                "ts": now,
            },
            {
                "id": str(uuid4()),
                "shift_id": "history-shift",
                "user_id": "u_best",
                "interaction": "APPLY",
                "ts": now,
            },
        ]
    }
    target_shift = {
        "id": "target-shift",
        "start_at": now,
        "location_id": "loc-1",
        "task_type": "picker",
        "employer_id": "emp-1",
        "workplace_id": "wp-1",
        "need_mk": True,
        "id_differential": False,
        "hours": 8,
        "reward": 1600.0,
        "capacity": 2,
    }

    assert asyncio.run(service.user(users_payload))["accepted"] == 2
    assert asyncio.run(service.shift(shifts_payload))["accepted"] == 1
    assert asyncio.run(service.event(events_payload))["accepted"] == 2
    assert asyncio.run(service.prepare(None))["status_code"] == 200
    assert asyncio.run(service.ready(None))["ready"]
    assert asyncio.run(service.repository.count_table("user_features")) == 1
    assert asyncio.run(service.repository.count_table("user_task_features")) == 1

    predict_response = asyncio.run(service.predict({"shift": target_shift, "limit": 10}))

    assert predict_response["status_code"] == 200
    assert predict_response["user_ids"][:2] == ["u_best", "u_alpha"]
