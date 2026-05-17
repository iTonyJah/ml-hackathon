"""
ML модель для предсказания выхода пользователя на смену.
LightGBM + векторизованный inference по всем пользователям.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

# Suppress sklearn feature names warning for LGBMClassifier when using numpy arrays
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")

LOGGER = logging.getLogger(__name__)


class MLModel:
    FEATURE_NAMES = [
        # User history features
        "user_apply_rate",
        "user_finish_rate",
        "user_cancel_rate",
        "user_total_applies",
        "user_total_views",
        "user_active_days",
        "user_has_mk",
        # New: replaces useless user_is_strict_location (always False in dataset)
        "user_location_applies",  # how many times user applied in this location
        "user_task_type_applies",  # how many times user applied to this task_type
        # Shift features
        "shift_hour",
        "shift_dayofweek",
        "shift_hours",
        "shift_reward",
        "shift_capacity",
        "shift_need_mk",
        "shift_id_differential",
        # Cross features
        "location_match",
        "mk_compatible",
        "user_worked_with_employer",
        "user_worked_at_workplace",
        "employer_avg_fill_rate",
        "user_reward_vs_avg",
        # Recurring shift: how many times user applied to this exact shift before
        "user_shift_apply_count",
        # Recurring shift outcome signals
        "user_shift_finish_count",
        "user_shift_cancel_count",
        # Recency: days from user's last apply to this shift to shift start (9999 if never applied)
        "user_shift_apply_recency_days",
    ]

    def __init__(self) -> None:
        self.model = None
        self.is_trained = False
        self._user_stats: dict[str, dict] = {}
        self._employer_stats: dict[str, float] = {}
        self._scaler = None
        self._use_scaler = False

        # Maps (user_id, shift_id) -> count of prior interactions (for recurring shifts)
        self._apply_map: dict[tuple[str, str], int] = {}
        self._finish_map: dict[tuple[str, str], int] = {}
        self._cancel_map: dict[tuple[str, str], int] = {}
        # (user_id, shift_id) -> unix timestamp (seconds) of most recent APPLY to this shift
        self._apply_ts_map: dict[tuple[str, str], float] = {}
        # Users with 0 applies but ≥1 finish: workers whose applies predate training window
        self._sleeper_set: set[str] = set()

        # Vectorized inference cache — populated in build_inference_cache()
        self._inf_user_ids: list[str] = []
        self._inf_uid_to_idx: dict[str, int] = {}
        self._inf_apply_rates: np.ndarray = np.array([], dtype=np.float32)
        self._inf_finish_rates: np.ndarray = np.array([], dtype=np.float32)
        self._inf_cancel_rates: np.ndarray = np.array([], dtype=np.float32)
        self._inf_total_applies: np.ndarray = np.array([], dtype=np.float32)
        self._inf_total_views: np.ndarray = np.array([], dtype=np.float32)
        self._inf_active_days: np.ndarray = np.array([], dtype=np.float32)
        self._inf_has_mk: np.ndarray = np.array([], dtype=np.float32)
        self._inf_avg_rewards: np.ndarray = np.array([], dtype=np.float32)
        self._inf_locations: np.ndarray = np.array([], dtype=object)
        self._inf_employer_sets: list[set] = []
        self._inf_workplace_sets: list[set] = []
        self._inf_location_apply_counts: list[dict] = []
        self._inf_task_type_apply_counts: list[dict] = []

    def _build_user_stats(self, events: pd.DataFrame, shifts: pd.DataFrame) -> None:
        LOGGER.info("Building user stats from %d events", len(events))

        # Include location_id and task_type for location/task_type apply tracking
        shift_cols = ["id", "employer_id", "workplace_id", "reward", "location_id", "task_type"]
        available = [c for c in shift_cols if c in shifts.columns]
        ev = events.merge(
            shifts[available].rename(columns={"id": "shift_id"}),
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

            # Fix: count unique calendar days, not unique timestamps
            if "ts" in group.columns and not group["ts"].isna().all():
                ts_series = pd.to_datetime(group["ts"], utc=True, errors="coerce")
                active_days = int(ts_series.dt.date.nunique())
            else:
                active_days = 1

            # Location-level apply counts: how often user applied in each location
            location_apply_counts: dict[str, int] = {}
            if "location_id" in applies.columns:
                for loc in applies["location_id"].dropna().astype(str):
                    location_apply_counts[loc] = location_apply_counts.get(loc, 0) + 1

            # Task-type apply counts
            task_type_apply_counts: dict[str, int] = {}
            if "task_type" in applies.columns:
                for tt in applies["task_type"].dropna().astype(str):
                    task_type_apply_counts[tt] = task_type_apply_counts.get(tt, 0) + 1

            self._user_stats[str(user_id)] = {
                "apply_rate": n_applies / max(1, n_total),
                "finish_rate": len(finishes) / max(1, n_applies),
                "cancel_rate": len(cancels) / max(1, n_applies),
                "total_applies": n_applies,
                "total_views": n_views,
                "active_days": active_days,
                "employers": set(applies["employer_id"].dropna().astype(str)),
                "workplaces": set(applies["workplace_id"].dropna().astype(str)),
                "avg_reward": float(applies["reward"].mean()) if n_applies > 0 else 0.0,
                "location_apply_counts": location_apply_counts,
                "task_type_apply_counts": task_type_apply_counts,
            }

        if "employer_id" in ev.columns:
            employer_applies = ev[ev["interaction"] == "APPLY"].groupby("employer_id").size()
            employer_total = ev.groupby("employer_id").size()
            self._employer_stats = (employer_applies / employer_total.clip(lower=1)).to_dict()

        # Build per-(user, shift) interaction counts for recurring shift features
        def _build_interaction_map(interaction: str) -> dict[tuple[str, str], int]:
            sub = events[events["interaction"] == interaction].copy()
            if sub.empty:
                return {}
            counts = sub.groupby([sub["user_id"].astype(str), sub["shift_id"].astype(str)]).size()
            return {(str(uid), str(sid)): int(cnt) for (uid, sid), cnt in counts.items()}

        self._apply_map = _build_interaction_map("APPLY")
        self._finish_map = _build_interaction_map("FINISHED")
        self._cancel_map = _build_interaction_map("USER_CANCEL")

        # Most recent apply timestamp per (user, shift) for recency feature
        applies_sub = events[events["interaction"] == "APPLY"].copy()
        if not applies_sub.empty and "ts" in applies_sub.columns:
            ts_parsed = pd.to_datetime(applies_sub["ts"], utc=True, errors="coerce")
            applies_sub = applies_sub.assign(ts_secs=ts_parsed.astype(np.int64) / 1e9)
            uid_str = applies_sub["user_id"].astype(str)
            sid_str = applies_sub["shift_id"].astype(str)
            max_ts = applies_sub.groupby([uid_str, sid_str])["ts_secs"].max()
            self._apply_ts_map = {
                (str(uid), str(sid)): float(ts) for (uid, sid), ts in max_ts.items()
            }
        else:
            self._apply_ts_map = {}

        self._sleeper_set = {
            uid
            for uid, stats in self._user_stats.items()
            if stats.get("total_applies", 0) == 0 and stats.get("finish_rate", 0.0) > 0.0
        }
        LOGGER.info(
            "Built stats for %d users, %d employers, %d shift-apply pairs, %d sleepers",
            len(self._user_stats),
            len(self._employer_stats),
            len(self._apply_map),
            len(self._sleeper_set),
        )

    def build_inference_cache(self, users: pd.DataFrame) -> None:
        """Precompute numpy arrays for vectorized batch inference (all users)."""
        LOGGER.info("Building vectorized inference cache for %d users", len(users))
        user_ids = users["id"].astype(str).tolist()
        n = len(user_ids)

        self._inf_user_ids = user_ids
        self._inf_uid_to_idx = {uid: i for i, uid in enumerate(user_ids)}

        apply_rates = np.zeros(n, dtype=np.float32)
        finish_rates = np.zeros(n, dtype=np.float32)
        cancel_rates = np.zeros(n, dtype=np.float32)
        total_applies = np.zeros(n, dtype=np.float32)
        total_views = np.zeros(n, dtype=np.float32)
        active_days = np.zeros(n, dtype=np.float32)
        has_mk = np.zeros(n, dtype=np.float32)
        avg_rewards = np.zeros(n, dtype=np.float32)
        locations: list[str] = [""] * n
        employer_sets: list[set] = [set() for _ in range(n)]
        workplace_sets: list[set] = [set() for _ in range(n)]
        loc_apply_counts: list[dict] = [{} for _ in range(n)]
        tt_apply_counts: list[dict] = [{} for _ in range(n)]

        user_map = {str(r["id"]): r for _, r in users.iterrows()}

        for i, uid in enumerate(user_ids):
            stats = self._user_stats.get(uid, {})
            apply_rates[i] = float(stats.get("apply_rate", 0.0))
            finish_rates[i] = float(stats.get("finish_rate", 0.0))
            cancel_rates[i] = float(stats.get("cancel_rate", 0.0))
            total_applies[i] = float(stats.get("total_applies", 0))
            total_views[i] = float(stats.get("total_views", 0))
            active_days[i] = float(stats.get("active_days", 0))
            avg_rewards[i] = float(stats.get("avg_reward", 0.0))
            employer_sets[i] = stats.get("employers", set())
            workplace_sets[i] = stats.get("workplaces", set())
            loc_apply_counts[i] = stats.get("location_apply_counts", {})
            tt_apply_counts[i] = stats.get("task_type_apply_counts", {})

            row = user_map.get(uid)
            if row is not None:
                has_mk[i] = float(bool(row["has_mk"]))
                locations[i] = str(row["location_id"])

        self._inf_apply_rates = apply_rates
        self._inf_finish_rates = finish_rates
        self._inf_cancel_rates = cancel_rates
        self._inf_total_applies = total_applies
        self._inf_total_views = total_views
        self._inf_active_days = active_days
        self._inf_has_mk = has_mk
        self._inf_avg_rewards = avg_rewards
        self._inf_locations = np.array(locations, dtype=object)
        self._inf_employer_sets = employer_sets
        self._inf_workplace_sets = workplace_sets
        self._inf_location_apply_counts = loc_apply_counts
        self._inf_task_type_apply_counts = tt_apply_counts

        LOGGER.info("Inference cache ready: %d users", n)

    def _make_features(self, user_id: str, user_row: dict, shift_row: dict) -> list[float]:
        stats = self._user_stats.get(str(user_id), {})

        start_at = pd.to_datetime(shift_row.get("start_at"), utc=True, errors="coerce")
        shift_hour = int(start_at.hour) if start_at is not pd.NaT else 12
        shift_dayofweek = int(start_at.dayofweek) if start_at is not pd.NaT else 0
        shift_hours = int(shift_row.get("hours", 8))
        shift_reward = float(shift_row.get("reward", 0.0))
        shift_capacity = int(shift_row.get("capacity", 1))
        shift_need_mk = int(bool(shift_row.get("need_mk", 0)))
        shift_id_differential = int(bool(shift_row.get("id_differential", 0)))
        shift_location = str(shift_row.get("location_id", ""))
        task_type = str(shift_row.get("task_type", ""))
        employer_id = str(shift_row.get("employer_id", ""))
        workplace_id = str(shift_row.get("workplace_id", ""))

        user_has_mk = int(bool(user_row.get("has_mk", 0)))
        location_match = int(str(user_row.get("location_id", "")) == shift_location)
        mk_compatible = int(not shift_need_mk or user_has_mk)
        user_worked_with_employer = int(employer_id in stats.get("employers", set()))
        user_worked_at_workplace = int(workplace_id in stats.get("workplaces", set()))
        employer_fill_rate = float(self._employer_stats.get(employer_id, 0.0))
        user_avg_reward = float(stats.get("avg_reward", shift_reward))
        user_reward_vs_avg = shift_reward / max(1.0, user_avg_reward)
        user_location_applies = float(stats.get("location_apply_counts", {}).get(shift_location, 0))
        user_task_type_applies = float(stats.get("task_type_apply_counts", {}).get(task_type, 0))
        shift_id = str(shift_row.get("id", ""))
        uid_str = str(user_id)
        user_shift_apply_count = float(self._apply_map.get((uid_str, shift_id), 0))
        user_shift_finish_count = float(self._finish_map.get((uid_str, shift_id), 0))
        user_shift_cancel_count = float(self._cancel_map.get((uid_str, shift_id), 0))

        last_apply_ts = self._apply_ts_map.get((uid_str, shift_id))
        if last_apply_ts is not None and start_at is not pd.NaT:
            user_shift_apply_recency_days = max(
                0.0, (start_at.timestamp() - last_apply_ts) / 86400.0
            )
        else:
            user_shift_apply_recency_days = 9999.0

        return [
            float(stats.get("apply_rate", 0.0)),
            float(stats.get("finish_rate", 0.0)),
            float(stats.get("cancel_rate", 0.0)),
            float(stats.get("total_applies", 0)),
            float(stats.get("total_views", 0)),
            float(stats.get("active_days", 0)),
            float(user_has_mk),
            user_location_applies,
            user_task_type_applies,
            float(shift_hour),
            float(shift_dayofweek),
            float(shift_hours),
            shift_reward,
            float(shift_capacity),
            float(shift_need_mk),
            float(shift_id_differential),
            float(location_match),
            float(mk_compatible),
            float(user_worked_with_employer),
            float(user_worked_at_workplace),
            employer_fill_rate,
            user_reward_vs_avg,
            user_shift_apply_count,
            user_shift_finish_count,
            user_shift_cancel_count,
            user_shift_apply_recency_days,
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

        # shift_id -> start_at for filtering post-start APPLY events
        shift_start_map: dict[str, pd.Timestamp] = {}
        if "start_at" in shifts.columns:
            starts = pd.to_datetime(shifts["start_at"], utc=True, errors="coerce")
            shift_start_map = dict(zip(shifts["id"].astype(str), starts))

        # Positives: APPLY before shift.start_at, excluding SYSTEM_CANCEL shifts
        apply_events = events[
            (events["interaction"] == "APPLY")
            & (~events["shift_id"].astype(str).isin(sys_cancel_ids))
        ].copy()
        if "ts" in apply_events.columns and shift_start_map:
            ts_parsed = pd.to_datetime(apply_events["ts"], utc=True, errors="coerce")
            shift_start = apply_events["shift_id"].astype(str).map(shift_start_map)
            apply_events = apply_events[
                shift_start.isna() | ts_parsed.isna() | (ts_parsed <= shift_start)
            ]

        applies = apply_events[["user_id", "shift_id"]].drop_duplicates().copy()
        applies["label"] = 1

        apply_pairs = set(zip(applies["user_id"].astype(str), applies["shift_id"].astype(str)))

        # Negatives: VIEW and USER_CANCEL without subsequent APPLY (per requirements)
        neg_events = (
            events[events["interaction"].isin(["VIEW", "USER_CANCEL"])][["user_id", "shift_id"]]
            .drop_duplicates()
            .copy()
        )
        neg_events["has_apply"] = neg_events.apply(
            lambda r: (str(r["user_id"]), str(r["shift_id"])) in apply_pairs, axis=1
        )
        negatives = neg_events[~neg_events["has_apply"]][["user_id", "shift_id"]].copy()
        negatives["label"] = 0
        negatives = negatives.sample(n=min(len(negatives), len(applies) * 5), random_state=42)

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

        try:
            import lightgbm as lgb

            X, y = self._build_training_data(events, shifts, users)
            if len(X) == 0:
                LOGGER.warning("No training data, skipping")
                return

            from sklearn.model_selection import train_test_split

            pos_weight = float((y == 0).sum()) / max(1, (y == 1).sum())
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.15, random_state=42, stratify=y
            )
            self.model = lgb.LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=63,
                max_depth=6,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=pos_weight,
                n_jobs=2,
                random_state=42,
                verbose=-1,
            )
            self.model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
            self._use_scaler = False
            LOGGER.info(
                "LGBMClassifier trained: samples=%d best_iter=%d val_auc=%.4f",
                len(X),
                self.model.best_iteration_,
                list(self.model.best_score_["valid_0"].values())[0],
            )

        except ImportError:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            X_fb, y_fb = self._build_training_data(events, shifts, users)
            if len(X_fb) == 0:
                LOGGER.warning("No training data, skipping")
                return
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X_fb)
            self.model = LogisticRegression(
                C=1.0, max_iter=1000, random_state=42, class_weight="balanced"
            )
            self.model.fit(X_scaled, y_fb)
            self._use_scaler = True
            LOGGER.info("LogisticRegression trained, samples=%d", len(X_fb))

        self.is_trained = True

    def _rerank_by_apply_recency(
        self,
        scored: list[tuple[str, float]],
        shift_id: str,
        shift_start_secs: float,
    ) -> list[tuple[str, float]]:
        """Re-rank by recency of last apply to this specific shift.

        Sort order: (no_prior_apply=0<1, days_since_apply ASC, ml_score DESC).
        Users who applied most recently to this exact shift rank first.
        Users with no prior apply rank after all prior-appliers.
        """

        def sort_key(item: tuple[str, float]) -> tuple[int, float, float]:
            uid, score = item
            last_ts = self._apply_ts_map.get((uid, shift_id))
            if last_ts is not None:
                days_ago = max(0.0, (shift_start_secs - last_ts) / 86400.0)
                return (0, days_ago, -score)
            if uid in self._sleeper_set:
                # Sleeping workers (0 applies, ≥1 finish): applies predate training window
                finish_rate = self._user_stats.get(uid, {}).get("finish_rate", 0.0)
                return (1, -finish_rate, -score)
            return (2, 0.0, -score)

        return sorted(scored, key=sort_key)

    def predict_scores(
        self,
        user_ids: list[str],
        users_dict: dict[str, dict],
        shift_row: dict,
    ) -> list[tuple[str, float]]:
        if not self.is_trained or not user_ids:
            return [
                (uid, float(i) / max(1, len(user_ids))) for i, uid in enumerate(reversed(user_ids))
            ]

        # Use vectorized inference cache if available
        if self._inf_uid_to_idx:
            scored = self._predict_scores_vectorized(user_ids, shift_row)
        else:
            # Fallback: per-user feature construction
            X_rows = [
                self._make_features(str(uid), users_dict.get(str(uid), {}), shift_row)
                for uid in user_ids
            ]
            X = np.array(X_rows, dtype=np.float32)
            if self._use_scaler and self._scaler is not None:
                X = self._scaler.transform(X)
            scores = self.model.predict_proba(X)[:, 1]
            scored = sorted(zip(user_ids, scores.tolist()), key=lambda x: x[1], reverse=True)

        # Re-rank: users who applied most recently to this exact shift rank first
        shift_id = str(shift_row.get("id", ""))
        start_at = pd.to_datetime(shift_row.get("start_at"), utc=True, errors="coerce")
        if shift_id and start_at is not pd.NaT and self._apply_ts_map:
            scored = self._rerank_by_apply_recency(scored, shift_id, float(start_at.timestamp()))

        return scored

    def _predict_scores_vectorized(
        self, user_ids: list[str], shift_row: dict
    ) -> list[tuple[str, float]]:
        """Vectorized scoring using precomputed numpy arrays. Called for all users at once."""
        # Precompute shift constants once
        start_at = pd.to_datetime(shift_row.get("start_at"), utc=True, errors="coerce")
        shift_hour = float(start_at.hour if start_at is not pd.NaT else 12)
        shift_dayofweek = float(start_at.dayofweek if start_at is not pd.NaT else 0)
        shift_hours = float(shift_row.get("hours", 8))
        shift_reward = float(shift_row.get("reward", 0.0))
        shift_capacity = float(shift_row.get("capacity", 1))
        shift_need_mk_val = bool(shift_row.get("need_mk", False))
        shift_need_mk = float(int(shift_need_mk_val))
        shift_id_diff = float(int(bool(shift_row.get("id_differential", 0))))
        employer_id = str(shift_row.get("employer_id", ""))
        workplace_id = str(shift_row.get("workplace_id", ""))
        shift_location = str(shift_row.get("location_id", ""))
        task_type = str(shift_row.get("task_type", ""))
        employer_fill_rate = float(self._employer_stats.get(employer_id, 0.0))
        shift_id = str(shift_row.get("id", ""))

        uid_strs = [str(uid) for uid in user_ids]
        indices = [self._inf_uid_to_idx.get(uid, -1) for uid in uid_strs]

        # Separate known from unknown users
        known = [(j, idx) for j, idx in enumerate(indices) if idx >= 0]
        unknown = [j for j, idx in enumerate(indices) if idx < 0]

        if not known:
            return [(uid_strs[j], 0.0) for j in range(len(uid_strs))]

        known_j = [k[0] for k in known]
        vi = np.array([k[1] for k in known], dtype=np.int64)
        n_valid = len(vi)

        # Vectorized user features
        col_apply_rate = self._inf_apply_rates[vi]
        col_finish_rate = self._inf_finish_rates[vi]
        col_cancel_rate = self._inf_cancel_rates[vi]
        col_total_applies = self._inf_total_applies[vi]
        col_total_views = self._inf_total_views[vi]
        col_active_days = self._inf_active_days[vi]
        col_has_mk = self._inf_has_mk[vi]

        # Location-based apply counts (still needs loop but over dicts)
        col_location_applies = np.array(
            [float(self._inf_location_apply_counts[i].get(shift_location, 0)) for i in vi],
            dtype=np.float32,
        )
        col_task_type_applies = np.array(
            [float(self._inf_task_type_apply_counts[i].get(task_type, 0)) for i in vi],
            dtype=np.float32,
        )

        # Shift features (broadcast)
        ones = np.ones(n_valid, dtype=np.float32)
        col_shift_hour = ones * shift_hour
        col_shift_dow = ones * shift_dayofweek
        col_shift_hours = ones * shift_hours
        col_shift_reward = ones * shift_reward
        col_shift_capacity = ones * shift_capacity
        col_shift_need_mk = ones * shift_need_mk
        col_shift_id_diff = ones * shift_id_diff

        # Cross features (vectorized)
        col_location_match = (self._inf_locations[vi] == shift_location).astype(np.float32)
        col_mk_compatible = np.logical_or(
            not shift_need_mk_val, self._inf_has_mk[vi].astype(bool)
        ).astype(np.float32)

        # Set-based features (fast Python loops)
        col_worked_employer = np.array(
            [float(employer_id in self._inf_employer_sets[i]) for i in vi],
            dtype=np.float32,
        )
        col_worked_workplace = np.array(
            [float(workplace_id in self._inf_workplace_sets[i]) for i in vi],
            dtype=np.float32,
        )

        col_employer_fill_rate = ones * employer_fill_rate
        # Users with no reward history get neutral ratio (1.0) not inflated (shift_reward/1.0)
        avg_rewards = self._inf_avg_rewards[vi]
        col_reward_vs_avg = np.where(
            avg_rewards > 0,
            shift_reward / np.maximum(1.0, avg_rewards),
            1.0,
        )

        shift_start_secs = float(start_at.timestamp()) if start_at is not pd.NaT else 0.0

        col_shift_apply_count = np.array(
            [float(self._apply_map.get((uid_strs[j], shift_id), 0)) for j in known_j],
            dtype=np.float32,
        )
        col_shift_finish_count = np.array(
            [float(self._finish_map.get((uid_strs[j], shift_id), 0)) for j in known_j],
            dtype=np.float32,
        )
        col_shift_cancel_count = np.array(
            [float(self._cancel_map.get((uid_strs[j], shift_id), 0)) for j in known_j],
            dtype=np.float32,
        )
        col_apply_recency = np.array(
            [
                max(0.0, (shift_start_secs - self._apply_ts_map[(uid_strs[j], shift_id)]) / 86400.0)
                if (uid_strs[j], shift_id) in self._apply_ts_map
                else 9999.0
                for j in known_j
            ],
            dtype=np.float32,
        )

        X = np.column_stack(
            [
                col_apply_rate,
                col_finish_rate,
                col_cancel_rate,
                col_total_applies,
                col_total_views,
                col_active_days,
                col_has_mk,
                col_location_applies,
                col_task_type_applies,
                col_shift_hour,
                col_shift_dow,
                col_shift_hours,
                col_shift_reward,
                col_shift_capacity,
                col_shift_need_mk,
                col_shift_id_diff,
                col_location_match,
                col_mk_compatible,
                col_worked_employer,
                col_worked_workplace,
                col_employer_fill_rate,
                col_reward_vs_avg,
                col_shift_apply_count,
                col_shift_finish_count,
                col_shift_cancel_count,
                col_apply_recency,
            ]
        )

        if self._use_scaler and self._scaler is not None:
            X = self._scaler.transform(X)

        scores = self.model.predict_proba(X)[:, 1]

        result: list[tuple[str, float]] = [
            (uid_strs[j], float(scores[k])) for k, j in enumerate(known_j)
        ]
        for j in unknown:
            result.append((uid_strs[j], 0.0))

        return sorted(result, key=lambda x: x[1], reverse=True)
