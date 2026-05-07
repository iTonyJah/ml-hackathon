from __future__ import annotations

from collections.abc import Iterable

import aiosqlite

from hackaton.service.dto import EventDTO, ShiftDTO, UserDTO
from hackaton.service.ml_reranker import MlReranker


class Repository:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.reranker = MlReranker()

    async def upsert_users(self, users: Iterable[UserDTO]) -> int:
        payload = [(u.id, u.location_id, int(u.is_strict_location), int(u.has_mk)) for u in users]
        if not payload:
            return 0
        query = """
        INSERT INTO users(id, location_id, is_strict_location, has_mk)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          location_id=excluded.location_id,
          is_strict_location=excluded.is_strict_location,
          has_mk=excluded.has_mk
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(query, payload)
            await db.commit()
        return len(payload)

    async def upsert_shifts(self, shifts: Iterable[ShiftDTO]) -> int:
        payload = [
            (
                s.id,
                s.start_at.isoformat(),
                s.location_id,
                s.task_type,
                s.employer_id,
                s.workplace_id,
                int(s.need_mk),
                int(s.id_differential),
                s.hours,
                float(s.reward),
                s.capacity,
            )
            for s in shifts
        ]
        if not payload:
            return 0
        query = """
        INSERT INTO shifts(id, start_at, location_id, task_type, employer_id,
                           workplace_id, need_mk, id_differential, hours, reward, capacity)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          start_at=excluded.start_at,
          location_id=excluded.location_id,
          task_type=excluded.task_type,
          employer_id=excluded.employer_id,
          workplace_id=excluded.workplace_id,
          need_mk=excluded.need_mk,
          id_differential=excluded.id_differential,
          hours=excluded.hours,
          reward=excluded.reward,
          capacity=excluded.capacity
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(query, payload)
            await db.commit()
        return len(payload)

    async def insert_events(self, events: Iterable[EventDTO]) -> int:
        payload = [
            (str(e.id), e.shift_id, e.user_id, e.interaction.value, e.ts.isoformat())
            for e in events
        ]
        if not payload:
            return 0
        query = """
        INSERT OR REPLACE INTO events(id, shift_id, user_id, interaction, ts)
        VALUES(?, ?, ?, ?, ?)
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany(query, payload)
            await db.commit()
        return len(payload)

    async def count_table(self, table_name: str) -> int:
        if table_name not in {
            "users",
            "events",
            "shifts",
            "user_features",
            "user_task_features",
            "user_employer_features",
            "user_workplace_features",
        }:
            raise ValueError(f"unsupported table: {table_name}")
        query = f"SELECT COUNT(1) FROM {table_name}"  # nosec - table is validated above
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query)
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def rebuild_features(self) -> None:
        query = """
        DELETE FROM user_features;
        DELETE FROM user_task_features;
        DELETE FROM user_employer_features;
        DELETE FROM user_workplace_features;

        INSERT INTO user_features (
            user_id,
            view_cnt,
            apply_cnt,
            finished_cnt,
            user_cancel_cnt,
            system_cancel_cnt,
            avg_reward_per_hour
        )
        SELECT
            e.user_id,
            SUM(CASE WHEN e.interaction = 'VIEW' THEN 1 ELSE 0 END) AS view_cnt,
            SUM(CASE WHEN e.interaction = 'APPLY' THEN 1 ELSE 0 END) AS apply_cnt,
            SUM(CASE WHEN e.interaction = 'FINISHED' THEN 1 ELSE 0 END) AS finished_cnt,
            SUM(CASE WHEN e.interaction = 'USER_CANCEL' THEN 1 ELSE 0 END) AS user_cancel_cnt,
            SUM(CASE WHEN e.interaction = 'SYSTEM_CANCEL' THEN 1 ELSE 0 END) AS system_cancel_cnt,
            AVG(
                CASE
                    WHEN e.interaction IN ('APPLY', 'FINISHED') AND s.hours > 0
                    THEN s.reward / s.hours
                    ELSE NULL
                END
            ) AS avg_reward_per_hour
        FROM events e
        LEFT JOIN shifts s ON s.id = e.shift_id
        GROUP BY e.user_id;

        INSERT INTO user_task_features (
            user_id, task_type, view_cnt, apply_cnt, finished_cnt
        )
        SELECT
            e.user_id,
            s.task_type,
            SUM(CASE WHEN e.interaction = 'VIEW' THEN 1 ELSE 0 END) AS view_cnt,
            SUM(CASE WHEN e.interaction = 'APPLY' THEN 1 ELSE 0 END) AS apply_cnt,
            SUM(CASE WHEN e.interaction = 'FINISHED' THEN 1 ELSE 0 END) AS finished_cnt
        FROM events e
        JOIN shifts s ON s.id = e.shift_id
        GROUP BY e.user_id, s.task_type;

        INSERT INTO user_employer_features (
            user_id, employer_id, view_cnt, apply_cnt, finished_cnt
        )
        SELECT
            e.user_id,
            s.employer_id,
            SUM(CASE WHEN e.interaction = 'VIEW' THEN 1 ELSE 0 END) AS view_cnt,
            SUM(CASE WHEN e.interaction = 'APPLY' THEN 1 ELSE 0 END) AS apply_cnt,
            SUM(CASE WHEN e.interaction = 'FINISHED' THEN 1 ELSE 0 END) AS finished_cnt
        FROM events e
        JOIN shifts s ON s.id = e.shift_id
        GROUP BY e.user_id, s.employer_id;

        INSERT INTO user_workplace_features (
            user_id, workplace_id, view_cnt, apply_cnt, finished_cnt
        )
        SELECT
            e.user_id,
            s.workplace_id,
            SUM(CASE WHEN e.interaction = 'VIEW' THEN 1 ELSE 0 END) AS view_cnt,
            SUM(CASE WHEN e.interaction = 'APPLY' THEN 1 ELSE 0 END) AS apply_cnt,
            SUM(CASE WHEN e.interaction = 'FINISHED' THEN 1 ELSE 0 END) AS finished_cnt
        FROM events e
        JOIN shifts s ON s.id = e.shift_id
        GROUP BY e.user_id, s.workplace_id;
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(query)
            await db.commit()
        await self._fit_reranker()

    async def _fit_reranker(self) -> None:
        rows = await self._fetch_reranker_training_rows()
        self.reranker.fit(rows)

    async def _fetch_reranker_training_rows(self) -> list[dict[str, object]]:
        query = """
        WITH pair_events AS (
            SELECT
                e.user_id,
                e.shift_id,
                SUM(CASE WHEN e.interaction = 'VIEW' THEN 1 ELSE 0 END) AS pair_view_cnt,
                SUM(CASE WHEN e.interaction = 'APPLY' THEN 1 ELSE 0 END) AS pair_apply_cnt,
                SUM(CASE WHEN e.interaction = 'FINISHED' THEN 1 ELSE 0 END) AS pair_finished_cnt,
                SUM(
                    CASE WHEN e.interaction = 'USER_CANCEL' THEN 1 ELSE 0 END
                ) AS pair_user_cancel_cnt,
                SUM(
                    CASE WHEN e.interaction = 'SYSTEM_CANCEL' THEN 1 ELSE 0 END
                ) AS pair_system_cancel_cnt
            FROM events e
            GROUP BY e.user_id, e.shift_id
        )
        SELECT
            u.id AS user_id,
            CASE WHEN pair_events.pair_apply_cnt > 0 THEN 1 ELSE 0 END AS target,
            CASE WHEN u.location_id = s.location_id THEN 1.0 ELSE 0.0 END AS location_match,
            CASE
                WHEN u.is_strict_location = 1 AND u.location_id = s.location_id THEN 1.0
                ELSE 0.0
            END AS strict_location_match,
            CAST(u.has_mk AS REAL) AS has_mk,
            CAST(s.need_mk AS REAL) AS need_mk,
            CASE WHEN s.need_mk = 0 OR u.has_mk = 1 THEN 1.0 ELSE 0.0 END AS mk_match,
            CAST(u.is_strict_location AS REAL) AS is_strict_location,
            CAST(s.hours AS REAL) AS hours,
            CAST(s.reward AS REAL) AS reward,
            CAST(s.capacity AS REAL) AS capacity,
            CASE WHEN s.hours > 0 THEN s.reward / s.hours ELSE 0.0 END AS reward_per_hour,
            COALESCE(uf.avg_reward_per_hour, 0.0) AS avg_reward_per_hour,
            CASE
                WHEN uf.avg_reward_per_hour IS NULL THEN 0.0
                WHEN s.hours > 0 THEN ABS(uf.avg_reward_per_hour - (s.reward / s.hours))
                ELSE 0.0
            END AS reward_per_hour_diff,
            COALESCE(uf.view_cnt, 0) AS view_cnt,
            COALESCE(uf.apply_cnt, 0) AS apply_cnt,
            COALESCE(uf.finished_cnt, 0) AS finished_cnt,
            COALESCE(uf.user_cancel_cnt, 0) AS user_cancel_cnt,
            COALESCE(uf.system_cancel_cnt, 0) AS system_cancel_cnt,
            COALESCE(utf.view_cnt, 0) AS task_view_cnt,
            COALESCE(utf.apply_cnt, 0) AS task_apply_cnt,
            COALESCE(utf.finished_cnt, 0) AS task_finished_cnt,
            COALESCE(uef.view_cnt, 0) AS employer_view_cnt,
            COALESCE(uef.apply_cnt, 0) AS employer_apply_cnt,
            COALESCE(uef.finished_cnt, 0) AS employer_finished_cnt,
            COALESCE(uwf.view_cnt, 0) AS workplace_view_cnt,
            COALESCE(uwf.apply_cnt, 0) AS workplace_apply_cnt,
            COALESCE(uwf.finished_cnt, 0) AS workplace_finished_cnt,
            (
                CASE WHEN u.location_id = s.location_id THEN 30.0 ELSE 0.0 END
                + CASE WHEN u.has_mk = 1 THEN 8.0 ELSE 0.0 END
                + CASE
                    WHEN u.is_strict_location = 1 AND u.location_id = s.location_id THEN 5.0
                    ELSE 0.0
                  END
                + COALESCE(uf.finished_cnt, 0) * 3.0
                + COALESCE(uf.apply_cnt, 0) * 2.0
                + COALESCE(uf.view_cnt, 0) * 0.15
                - COALESCE(uf.user_cancel_cnt, 0) * 1.5
                - COALESCE(uf.system_cancel_cnt, 0) * 0.5
                + COALESCE(utf.finished_cnt, 0) * 12.0
                + COALESCE(utf.apply_cnt, 0) * 7.0
                + COALESCE(utf.view_cnt, 0) * 0.5
                + COALESCE(uef.finished_cnt, 0) * 18.0
                + COALESCE(uef.apply_cnt, 0) * 9.0
                + COALESCE(uef.view_cnt, 0) * 0.5
                + COALESCE(uwf.finished_cnt, 0) * 24.0
                + COALESCE(uwf.apply_cnt, 0) * 12.0
                + COALESCE(uwf.view_cnt, 0) * 0.5
                - MIN(20.0, CASE
                    WHEN uf.avg_reward_per_hour IS NULL THEN 0.0
                    WHEN s.hours > 0 THEN ABS(uf.avg_reward_per_hour - (s.reward / s.hours))
                    ELSE 0.0
                  END) * 0.05
            ) AS rule_score
        FROM pair_events
        JOIN users u ON u.id = pair_events.user_id
        JOIN shifts s ON s.id = pair_events.shift_id
        LEFT JOIN user_features uf ON uf.user_id = u.id
        LEFT JOIN user_task_features utf
            ON utf.user_id = u.id AND utf.task_type = s.task_type
        LEFT JOIN user_employer_features uef
            ON uef.user_id = u.id AND uef.employer_id = s.employer_id
        LEFT JOIN user_workplace_features uwf
            ON uwf.user_id = u.id AND uwf.workplace_id = s.workplace_id
        WHERE pair_events.pair_system_cancel_cnt = 0
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query)
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def find_top_candidates(
        self,
        location_id: str,
        need_mk: bool,
        limit: int,
    ) -> list[str]:
        query = """
        SELECT id
        FROM users
        WHERE location_id = ?
          AND (? = 0 OR has_mk = 1)
        ORDER BY is_strict_location DESC, has_mk DESC, id ASC
        LIMIT ?
        """
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, (location_id, int(need_mk), limit))
            rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def find_scored_candidates(self, shift: ShiftDTO, limit: int) -> list[str]:
        rows = await self._find_candidate_feature_rows(shift=shift, limit=limit)
        model_scores = self.reranker.predict_proba(rows)
        if model_scores is not None:
            rule_scores = [float(row["rule_score"]) for row in rows]
            min_rule_score = min(rule_scores)
            max_rule_score = max(rule_scores)
            rule_score_range = max(max_rule_score - min_rule_score, 1.0)
            for row, model_score in zip(rows, model_scores, strict=True):
                row["model_score"] = model_score
                normalized_rule_score = (
                    float(row["rule_score"]) - min_rule_score
                ) / rule_score_range
                row["ml_score"] = (0.75 * normalized_rule_score) + (0.25 * model_score)
            rows = sorted(
                rows,
                key=lambda row: (
                    -float(row["ml_score"]),
                    -float(row["rule_score"]),
                    -int(row["location_match"]),
                    -int(row["has_mk"]),
                    str(row["user_id"]),
                ),
            )
        return [str(row["user_id"]) for row in rows[:limit]]

    async def _find_candidate_feature_rows(
        self, shift: ShiftDTO, limit: int
    ) -> list[dict[str, object]]:
        reward_per_hour = float(shift.reward) / shift.hours if shift.hours else 0.0
        query = """
        SELECT
            u.id AS user_id,
            CASE WHEN u.location_id = ? THEN 1.0 ELSE 0.0 END AS location_match,
            CASE
                WHEN u.is_strict_location = 1 AND u.location_id = ? THEN 1.0
                ELSE 0.0
            END AS strict_location_match,
            CAST(u.has_mk AS REAL) AS has_mk,
            CAST(? AS REAL) AS need_mk,
            CASE WHEN ? = 0 OR u.has_mk = 1 THEN 1.0 ELSE 0.0 END AS mk_match,
            CAST(u.is_strict_location AS REAL) AS is_strict_location,
            CAST(? AS REAL) AS hours,
            CAST(? AS REAL) AS reward,
            CAST(? AS REAL) AS capacity,
            CAST(? AS REAL) AS reward_per_hour,
            COALESCE(uf.avg_reward_per_hour, 0.0) AS avg_reward_per_hour,
            CASE
                WHEN uf.avg_reward_per_hour IS NULL THEN 0.0
                ELSE ABS(uf.avg_reward_per_hour - ?)
            END AS reward_per_hour_diff,
            COALESCE(uf.view_cnt, 0) AS view_cnt,
            COALESCE(uf.apply_cnt, 0) AS apply_cnt,
            COALESCE(uf.finished_cnt, 0) AS finished_cnt,
            COALESCE(uf.user_cancel_cnt, 0) AS user_cancel_cnt,
            COALESCE(uf.system_cancel_cnt, 0) AS system_cancel_cnt,
            COALESCE(utf.view_cnt, 0) AS task_view_cnt,
            COALESCE(utf.apply_cnt, 0) AS task_apply_cnt,
            COALESCE(utf.finished_cnt, 0) AS task_finished_cnt,
            COALESCE(uef.view_cnt, 0) AS employer_view_cnt,
            COALESCE(uef.apply_cnt, 0) AS employer_apply_cnt,
            COALESCE(uef.finished_cnt, 0) AS employer_finished_cnt,
            COALESCE(uwf.view_cnt, 0) AS workplace_view_cnt,
            COALESCE(uwf.apply_cnt, 0) AS workplace_apply_cnt,
            COALESCE(uwf.finished_cnt, 0) AS workplace_finished_cnt,
            (
                CASE WHEN u.location_id = ? THEN 30.0 ELSE 0.0 END
                + CASE WHEN u.has_mk = 1 THEN 8.0 ELSE 0.0 END
                + CASE WHEN u.is_strict_location = 1 AND u.location_id = ? THEN 5.0 ELSE 0.0 END
                + COALESCE(uf.finished_cnt, 0) * 3.0
                + COALESCE(uf.apply_cnt, 0) * 2.0
                + COALESCE(uf.view_cnt, 0) * 0.15
                - COALESCE(uf.user_cancel_cnt, 0) * 1.5
                - COALESCE(uf.system_cancel_cnt, 0) * 0.5
                + COALESCE(utf.finished_cnt, 0) * 12.0
                + COALESCE(utf.apply_cnt, 0) * 7.0
                + COALESCE(utf.view_cnt, 0) * 0.5
                + COALESCE(uef.finished_cnt, 0) * 18.0
                + COALESCE(uef.apply_cnt, 0) * 9.0
                + COALESCE(uef.view_cnt, 0) * 0.5
                + COALESCE(uwf.finished_cnt, 0) * 24.0
                + COALESCE(uwf.apply_cnt, 0) * 12.0
                + COALESCE(uwf.view_cnt, 0) * 0.5
                - CASE
                    WHEN uf.avg_reward_per_hour IS NULL THEN 0.0
                    ELSE MIN(20.0, ABS(uf.avg_reward_per_hour - ?)) * 0.05
                  END
            ) AS rule_score
        FROM users u
        LEFT JOIN user_features uf ON uf.user_id = u.id
        LEFT JOIN user_task_features utf
            ON utf.user_id = u.id AND utf.task_type = ?
        LEFT JOIN user_employer_features uef
            ON uef.user_id = u.id AND uef.employer_id = ?
        LEFT JOIN user_workplace_features uwf
            ON uwf.user_id = u.id AND uwf.workplace_id = ?
        WHERE (? = 0 OR u.has_mk = 1)
          AND (
              u.location_id = ?
              OR u.is_strict_location = 0
              OR utf.user_id IS NOT NULL
              OR uef.user_id IS NOT NULL
              OR uwf.user_id IS NOT NULL
          )
        ORDER BY rule_score DESC, u.location_id = ? DESC, u.has_mk DESC, u.id ASC
        LIMIT ?
        """
        params = (
            shift.location_id,
            shift.location_id,
            int(shift.need_mk),
            int(shift.need_mk),
            shift.hours,
            shift.reward,
            shift.capacity,
            reward_per_hour,
            reward_per_hour,
            shift.location_id,
            shift.location_id,
            reward_per_hour,
            shift.task_type,
            shift.employer_id,
            shift.workplace_id,
            int(shift.need_mk),
            shift.location_id,
            shift.location_id,
            limit,
        )
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def fallback_candidates(self, limit: int) -> list[str]:
        query = "SELECT id FROM users ORDER BY id ASC LIMIT ?"
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, (limit,))
            rows = await cursor.fetchall()
        return [row[0] for row in rows]
