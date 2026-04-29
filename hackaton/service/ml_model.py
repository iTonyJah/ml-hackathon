"""
ML модель для предсказания выхода пользователя на смену.
Используется LightGBM с feature engineering на основе истории взаимодействий.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class MLModel:
    FEATURE_NAMES = [
        "user_apply_rate", "user_finish_rate", "user_cancel_rate",
        "user_total_applies", "user_total_views", "user_active_days",
        "user_has_mk", "user_is_strict_location",
        "shift_hour", "shift_dayofweek", "shift_hours", "shift_reward",
        "shift_capacity", "shift_need_mk", "shift_id_differential",
        "location_match", "mk_compatible",
        "user_worked_with_employer", "user_worked_at_workplace",
        "employer_avg_fill_rate", "user_reward_vs_avg",
    ]

    def __init__(self) -> None:
        self.model = None
        self.is_trained = False
        self._user_stats: dict[str, dict] = {}
        self._employer_stats: dict[str, float] = {}
        self._scaler = None
        self._use_scaler = False

    def _build_user_stats(self, events: pd.DataFrame, shifts: pd.DataFrame) -> None:
        LOGGER.info("Building user stats from %d events", len(events))

        ev = events.merge(
            shifts[["id", "employer_id", "workplace_id", "reward"]].rename(
                columns={"id": "shift_id"}
            ),
            on="shift_id",
            how="left",
        )

        for user_id, group in ev.groupby("user_id"):
            applies = group[group["interaction"] == "APPLY"]
            views = group[group["interaction"] == "VIEW"]
            finishes = group[group["interaction"] == "FINISHED"]
            cancels = group[group["interaction"] == "USER_CANCEL"]

            n_applies = len(applies)
            n_views = len(views)
            n_total = len(group)

            self._user_stats[str(user_id)] = {
                "apply_rate": n_applies / max(1, n_total),
                "finish_rate": len(finishes) / max(1, n_applies),
                "cancel_rate": len(cancels) / max(1, n_applies),
                "total_applies": n_applies,
                "total_views": n_views,
                "active_days": group["ts"].nunique() if "ts" in group else 1,
                "employers": set(applies["employer_id"].dropna().astype(str)),
                "workplaces": set(applies["workplace_id"].dropna().astype(str)),
                "avg_reward": float(applies["reward"].mean()) if n_applies > 0 else 0.0,
            }

        if "employer_id" in ev.columns:
            employer_applies = ev[ev["interaction"] == "APPLY"].groupby("employer_id").size()
            employer_total = ev.groupby("employer_id").size()
            self._employer_stats = (employer_applies / employer_total.clip(lower=1)).to_dict()

        LOGGER.info("Built stats for %d users, %d employers",
                    len(self._user_stats), len(self._employer_stats))

    def _make_features(self, user_id: str, user_row: dict, shift_row: dict) -> list[float]:
        stats = self._user_stats.get(str(user_id), {})

        user_apply_rate = float(stats.get("apply_rate", 0.0))
        user_finish_rate = float(stats.get("finish_rate", 0.0))
        user_cancel_rate = float(stats.get("cancel_rate", 0.0))
        user_total_applies = int(stats.get("total_applies", 0))
        user_total_views = int(stats.get("total_views", 0))
        user_active_days = int(stats.get("active_days", 0))
        user_has_mk = int(bool(user_row.get("has_mk", 0)))
        user_is_strict_location = int(bool(user_row.get("is_strict_location", 0)))

        start_at = pd.to_datetime(shift_row.get("start_at"), utc=True, errors="coerce")
        shift_hour = int(start_at.hour) if start_at is not pd.NaT else 12
        shift_dayofweek = int(start_at.dayofweek) if start_at is not pd.NaT else 0
        shift_hours = int(shift_row.get("hours", 8))
        shift_reward = float(shift_row.get("reward", 0.0))
        shift_capacity = int(shift_row.get("capacity", 1))
        shift_need_mk = int(bool(shift_row.get("need_mk", 0)))
        shift_id_differential = int(bool(shift_row.get("id_differential", 0)))

        location_match = int(
            str(user_row.get("location_id", "")) == str(shift_row.get("location_id", ""))
        )
        mk_compatible = int(not shift_need_mk or user_has_mk)

        employer_id = str(shift_row.get("employer_id", ""))
        workplace_id = str(shift_row.get("workplace_id", ""))
        user_worked_with_employer = int(employer_id in stats.get("employers", set()))
        user_worked_at_workplace = int(workplace_id in stats.get("workplaces", set()))
        employer_avg_fill_rate = float(self._employer_stats.get(employer_id, 0.0))

        user_avg_reward = float(stats.get("avg_reward", shift_reward))
        user_reward_vs_avg = shift_reward / max(1.0, user_avg_reward)

        return [
            user_apply_rate, user_finish_rate, user_cancel_rate,
            user_total_applies, user_total_views, user_active_days,
            user_has_mk, user_is_strict_location,
            shift_hour, shift_dayofweek, shift_hours, shift_reward,
            shift_capacity, shift_need_mk, shift_id_differential,
            location_match, mk_compatible,
            user_worked_with_employer, user_worked_at_workplace,
            employer_avg_fill_rate, user_reward_vs_avg,
        ]

    def _build_training_data(
        self,
        events: pd.DataFrame,
        shifts: pd.DataFrame,
        users: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        sys_cancel_ids = set(
            events[events["interaction"] == "SYSTEM_CANCEL"]["shift_id"].astype(str)
        )
        applies = events[
            (events["interaction"] == "APPLY") &
            (~events["shift_id"].astype(str).isin(sys_cancel_ids))
        ][["user_id", "shift_id"]].drop_duplicates().copy()
        applies["label"] = 1

        views = events[
            events["interaction"] == "VIEW"
        ][["user_id", "shift_id"]].drop_duplicates().copy()

        apply_pairs = set(
            zip(applies["user_id"].astype(str), applies["shift_id"].astype(str))
        )
        views["has_apply"] = views.apply(
            lambda r: (str(r["user_id"]), str(r["shift_id"])) in apply_pairs, axis=1
        )
        negatives = views[~views["has_apply"]][["user_id", "shift_id"]].copy()
        negatives["label"] = 0
        negatives = negatives.sample(
            n=min(len(negatives), len(applies) * 3), random_state=42
        )

        LOGGER.info("Training data: pos=%d, neg=%d", len(applies), len(negatives))
        data = pd.concat([applies, negatives], ignore_index=True)

        users_dict = {str(r["id"]): r.to_dict() for _, r in users.iterrows()}
        shifts_dict = {str(r["id"]): r.to_dict() for _, r in shifts.iterrows()}

        X_rows, y_vals = [], []
        for _, row in data.iterrows():
            uid = str(row["user_id"])
            sid = str(row["shift_id"])
            feats = self._make_features(uid, users_dict.get(uid, {}), shifts_dict.get(sid, {}))
            X_rows.append(feats)
            y_vals.append(int(row["label"]))

        return np.array(X_rows, dtype=np.float32), np.array(y_vals, dtype=np.int32)

    def train(self, events: pd.DataFrame, shifts: pd.DataFrame, users: pd.DataFrame) -> None:
        self._build_user_stats(events, shifts)
        X, y = self._build_training_data(events, shifts, users)

        if len(X) == 0:
            LOGGER.warning("No training data, skipping")
            return

        try:
            import lightgbm as lgb
            pos_weight = float((y == 0).sum()) / max(1, (y == 1).sum())
            self.model = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=63,
                max_depth=6,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=pos_weight,
                n_jobs=2,
                random_state=42,
                verbose=-1,
            )
            self.model.fit(X, y)
            self._use_scaler = False
            LOGGER.info("LightGBM model trained, samples=%d", len(X))

        except ImportError:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
            self.model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight="balanced")
            self.model.fit(X_scaled, y)
            self._use_scaler = True
            LOGGER.info("LogisticRegression trained, samples=%d", len(X))

        self.is_trained = True

    def predict_scores(
        self,
        user_ids: list[str],
        users_dict: dict[str, dict],
        shift_row: dict,
    ) -> list[tuple[str, float]]:
        if not self.is_trained or not user_ids:
            return [(uid, float(i) / max(1, len(user_ids)))
                    for i, uid in enumerate(reversed(user_ids))]

        X_rows = [self._make_features(str(uid), users_dict.get(str(uid), {}), shift_row)
                  for uid in user_ids]
        X = np.array(X_rows, dtype=np.float32)

        if self._use_scaler and self._scaler is not None:
            X = self._scaler.transform(X)

        scores = self.model.predict_proba(X)[:, 1]
        return sorted(zip(user_ids, scores.tolist()), key=lambda x: x[1], reverse=True)