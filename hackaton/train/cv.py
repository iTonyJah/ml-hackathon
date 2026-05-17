"""
Time-based cross-validation for the ranking model.

Split: events/shifts before cutoff = train, after cutoff = validation.
Replicates the production scoring pipeline without starting the RPC service.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hackaton.eval.metric import MetricResult, calculate_target_metric
from hackaton.service.ml_model import MLModel

LOGGER = logging.getLogger(__name__)

CANDIDATE_LIMIT = 300
GLOBAL_FALLBACK_THRESHOLD = 50  # mirrors PrepareManager.get_candidates logic


@dataclass
class CVResult:
    overall_metric: float
    evaluated_days: int
    evaluated_shifts: int
    day_metrics: dict[str, float]
    train_events: int
    val_events: int
    val_apply_shifts: int
    cutoff_date: str


def _build_location_cache(
    users: pd.DataFrame, events: pd.DataFrame
) -> tuple[dict[str, list[str]], list[str]]:
    """Replicate PrepareManager._build_cache: sorted by activity, no top-N cap."""
    if not events.empty:
        activity = events.groupby("user_id").size().reset_index(name="n_events")
    else:
        activity = pd.DataFrame(columns=["user_id", "n_events"])

    merged = users.merge(activity, left_on="id", right_on="user_id", how="left")
    merged["n_events"] = merged["n_events"].fillna(0).astype(int)
    merged = merged.sort_values("n_events", ascending=False)

    location_cache: dict[str, list[str]] = defaultdict(list)
    for _, row in merged.iterrows():
        location_cache[str(row["location_id"])].append(str(row["id"]))

    global_top = merged["id"].astype(str).tolist()
    return dict(location_cache), global_top


def _get_candidates(
    location_id: str,
    location_cache: dict[str, list[str]],
    global_top: list[str],
    limit: int,
) -> list[str]:
    """Mirrors PrepareManager.get_candidates."""
    loc = location_cache.get(str(location_id), [])
    if len(loc) < GLOBAL_FALLBACK_THRESHOLD:
        existing = set(loc)
        extra = [u for u in global_top if u not in existing][:500]
        candidates = loc + extra
    else:
        candidates = loc
    return candidates[:limit]


def run_cv(
    user_path: str,
    shift_path: str,
    event_path: str,
    val_days: int = 30,
    candidate_limit: int = CANDIDATE_LIMIT,
    output_dir: str | None = None,
) -> CVResult:
    LOGGER.info("Loading data...")
    users = pd.read_csv(user_path)
    shifts = pd.read_csv(shift_path)
    events = pd.read_csv(event_path)

    events["ts"] = pd.to_datetime(events["ts"], utc=True, errors="coerce")
    shifts["start_at"] = pd.to_datetime(shifts["start_at"], utc=True, errors="coerce")

    max_ts = events["ts"].max()
    cutoff = max_ts - pd.Timedelta(days=val_days)
    LOGGER.info("Cutoff: %s  (train < cutoff, val >= cutoff, val_days=%d)", cutoff.date(), val_days)

    train_events = events[events["ts"] < cutoff].copy()
    val_events = events[events["ts"] >= cutoff].copy()

    LOGGER.info("Split: train_events=%d  val_events=%d", len(train_events), len(val_events))

    # Ground truth for val: APPLY events in val window
    val_applies = (
        val_events[val_events["interaction"] == "APPLY"][["user_id", "shift_id"]]
        .drop_duplicates()
        .copy()
    )
    val_applies["user_id"] = val_applies["user_id"].astype(str)
    val_applies["shift_id"] = val_applies["shift_id"].astype(str)

    # Val shifts: only those that have at least one APPLY in val window
    val_shift_ids = set(val_applies["shift_id"])
    val_shifts = shifts[shifts["id"].astype(str).isin(val_shift_ids)].copy()
    LOGGER.info("Val: apply_events=%d  unique_shifts=%d", len(val_applies), len(val_shift_ids))

    LOGGER.info("Training model on pre-cutoff data...")
    model = MLModel()
    model.train(train_events, shifts, users)
    if not model.is_trained:
        raise RuntimeError("Model failed to train — not enough pre-cutoff data")
    model.build_inference_cache(users)
    LOGGER.info("Model trained. Building location cache...")

    location_cache, global_top = _build_location_cache(users, train_events)

    # Build users_cache for model.predict_scores
    users_cache = {
        str(r["id"]): {
            "id": str(r["id"]),
            "location_id": str(r["location_id"]),
            "is_strict_location": bool(r["is_strict_location"]),
            "has_mk": bool(r["has_mk"]),
        }
        for _, r in users.iterrows()
    }

    # Score val shifts
    LOGGER.info("Scoring %d val shifts...", len(val_shifts))
    positive_pairs = {(r["user_id"], r["shift_id"]) for _, r in val_applies.iterrows()}

    rows: list[dict] = []
    skipped = 0
    for _, shift_row in val_shifts.iterrows():
        sid = str(shift_row["id"])
        candidates = _get_candidates(
            shift_row["location_id"], location_cache, global_top, candidate_limit
        )
        if not candidates:
            skipped += 1
            continue

        shift_dict = {
            "id": sid,
            "start_at": shift_row["start_at"].isoformat(),
            "location_id": str(shift_row["location_id"]),
            "task_type": str(shift_row["task_type"]),
            "employer_id": str(shift_row["employer_id"]),
            "workplace_id": str(shift_row["workplace_id"]),
            "need_mk": bool(shift_row["need_mk"]),
            "id_differential": bool(shift_row["id_differential"]),
            "hours": int(shift_row["hours"]),
            "reward": float(shift_row["reward"]),
            "capacity": int(shift_row["capacity"]),
        }

        scored = model.predict_scores(candidates, users_cache, shift_dict)
        top10 = scored[:10]

        for rank, (uid, _) in enumerate(top10, start=1):
            score = 1.0 - (rank - 1) / max(1, len(top10))
            rows.append(
                {
                    "shift_id": sid,
                    "start_at": shift_row["start_at"],
                    "capacity": int(shift_row["capacity"]),
                    "target": int((uid, sid) in positive_pairs),
                    "score": float(score),
                }
            )

    if skipped:
        LOGGER.warning("Skipped %d shifts (no candidates)", skipped)

    if not rows:
        LOGGER.error("No prediction rows built — check data")
        return CVResult(
            overall_metric=0.0,
            evaluated_days=0,
            evaluated_shifts=0,
            day_metrics={},
            train_events=len(train_events),
            val_events=len(val_events),
            val_apply_shifts=len(val_shift_ids),
            cutoff_date=str(cutoff.date()),
        )

    frame = pd.DataFrame(rows)
    metric: MetricResult = calculate_target_metric(frame)

    LOGGER.info(
        "CV result: overall=%.4f  days=%d  shifts=%d",
        metric.target_metric,
        metric.evaluated_days,
        metric.evaluated_shifts,
    )

    result = CVResult(
        overall_metric=metric.target_metric,
        evaluated_days=metric.evaluated_days,
        evaluated_shifts=metric.evaluated_shifts,
        day_metrics=metric.day_metrics,
        train_events=len(train_events),
        val_events=len(val_events),
        val_apply_shifts=len(val_shift_ids),
        cutoff_date=str(cutoff.date()),
    )

    if output_dir:
        _write_report(Path(output_dir), result, val_days)

    return result


def _write_report(output_dir: Path, result: CVResult, val_days: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Time-based CV Report",
        "",
        f"- cutoff_date: {result.cutoff_date}",
        f"- val_days: {val_days}",
        "- model: LGBMClassifier+AUC",
        f"- train_events: {result.train_events:,}",
        f"- val_events: {result.val_events:,}",
        f"- val_apply_shifts: {result.val_apply_shifts:,}",
        "",
        "## Overall metric",
        "",
        f"- overall_target_metric: {result.overall_metric:.4f}",
        f"- evaluated_days: {result.evaluated_days}",
        f"- evaluated_shifts: {result.evaluated_shifts}",
        "",
        "## Daily metrics",
        "",
    ]
    for day, m in sorted(result.day_metrics.items()):
        lines.append(f"- {day}: {m:.4f}")
    (output_dir / "cv_report.md").write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("CV report saved to %s", output_dir / "cv_report.md")
