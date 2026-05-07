from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

LOGGER = logging.getLogger(__name__)

ML_FEATURE_COLUMNS = [
    "location_match",
    "strict_location_match",
    "has_mk",
    "need_mk",
    "mk_match",
    "is_strict_location",
    "hours",
    "reward",
    "capacity",
    "reward_per_hour",
    "avg_reward_per_hour",
    "reward_per_hour_diff",
    "view_cnt",
    "apply_cnt",
    "finished_cnt",
    "user_cancel_cnt",
    "system_cancel_cnt",
    "task_view_cnt",
    "task_apply_cnt",
    "task_finished_cnt",
    "employer_view_cnt",
    "employer_apply_cnt",
    "employer_finished_cnt",
    "workplace_view_cnt",
    "workplace_apply_cnt",
    "workplace_finished_cnt",
    "rule_score",
]


class MlReranker:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model: HistGradientBoostingClassifier | None = None
        self.train_rows = 0
        self.positive_rows = 0

    @property
    def ready(self) -> bool:
        return self.model is not None

    def reset(self) -> None:
        self.model = None
        self.train_rows = 0
        self.positive_rows = 0

    def fit(self, rows: list[dict[str, object]]) -> bool:
        self.reset()
        self.train_rows = len(rows)
        self.positive_rows = int(sum(int(row["target"]) for row in rows))
        if len(rows) < 4:
            LOGGER.info("ML reranker skipped: not enough rows=%s", len(rows))
            return False
        if self.positive_rows == 0 or self.positive_rows == len(rows):
            LOGGER.info(
                "ML reranker skipped: target has one class, rows=%s positives=%s",
                len(rows),
                self.positive_rows,
            )
            return False

        x = self._to_matrix(rows)
        y = np.array([int(row["target"]) for row in rows], dtype=np.int8)
        model = HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=self.random_state,
        )
        model.fit(x, y)
        self.model = model
        LOGGER.info(
            "ML reranker fitted: rows=%s positives=%s features=%s",
            self.train_rows,
            self.positive_rows,
            len(ML_FEATURE_COLUMNS),
        )
        return True

    def predict_proba(self, rows: list[dict[str, object]]) -> list[float] | None:
        if self.model is None or not rows:
            return None
        x = self._to_matrix(rows)
        return [float(v) for v in self.model.predict_proba(x)[:, 1]]

    @staticmethod
    def _to_matrix(rows: list[dict[str, object]]) -> np.ndarray:
        return np.array(
            [[float(row.get(column) or 0.0) for column in ML_FEATURE_COLUMNS] for row in rows],
            dtype=np.float32,
        )
