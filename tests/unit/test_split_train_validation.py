from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from scripts.split_train_validation import split_train_validation


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_split_train_validation_writes_eval_contract_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_train_dir = tmp_path / "train_split"
    output_validation_dir = tmp_path / "validation"

    _write_csv(
        input_dir / "user.csv",
        ["location_id", "is_strict_location", "id", "has_mk"],
        [{"location_id": "loc-1", "is_strict_location": "False", "id": "u1", "has_mk": "True"}],
    )
    _write_csv(
        input_dir / "shift.csv",
        [
            "id",
            "start_at",
            "location_id",
            "task_type",
            "employer_id",
            "workplace_id",
            "need_mk",
            "id_differential",
            "hours",
            "reward",
            "capacity",
        ],
        [
            {
                "id": "s1",
                "start_at": "2026-01-01T09:00:00+00:00",
                "location_id": "loc-1",
                "task_type": "loader",
                "employer_id": "e1",
                "workplace_id": "w1",
                "need_mk": "False",
                "id_differential": "False",
                "hours": 8,
                "reward": 1000,
                "capacity": 1,
            },
            {
                "id": "s2",
                "start_at": "2026-01-02T09:00:00+00:00",
                "location_id": "loc-1",
                "task_type": "loader",
                "employer_id": "e1",
                "workplace_id": "w1",
                "need_mk": "False",
                "id_differential": "False",
                "hours": 8,
                "reward": 1000,
                "capacity": 1,
            },
        ],
    )
    _write_csv(
        input_dir / "event.csv",
        ["id", "shift_id", "user_id", "interaction", "ts"],
        [
            {
                "id": "event-1",
                "shift_id": "s1",
                "user_id": "u1",
                "interaction": "VIEW",
                "ts": "2026-01-01",
            },
            {
                "id": "event-early-val-shift",
                "shift_id": "s2",
                "user_id": "u1",
                "interaction": "VIEW",
                "ts": "2026-01-01",
            },
            {
                "id": "event-2",
                "shift_id": "s2",
                "user_id": "u1",
                "interaction": "APPLY",
                "ts": "2026-01-02",
            },
        ],
    )

    summary = split_train_validation(
        input_dir=input_dir,
        output_train_dir=output_train_dir,
        output_validation_dir=output_validation_dir,
        validation_days=1,
        cutoff_date=None,
        force=False,
    )

    assert summary.cutoff_date == date(2026, 1, 2)
    assert summary.train_shifts == 1
    assert summary.validation_shifts == 1
    assert summary.validation_apply == 1
    assert _read_csv(output_validation_dir / "apply.csv") == [
        {"user_id": "u1", "shift_id": "s2", "date": "2026-01-02"}
    ]
    assert _read_csv(output_validation_dir / "shift.csv")[0]["id"] == "s2"
    assert _read_csv(output_train_dir / "shift.csv")[0]["id"] == "s1"
    assert _read_csv(output_train_dir / "event.csv") == [
        {
            "id": "event-1",
            "shift_id": "s1",
            "user_id": "u1",
            "interaction": "VIEW",
            "ts": "2026-01-01",
        }
    ]


def test_split_train_validation_apply_labels_follow_official_methodology(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_train_dir = tmp_path / "train_split"
    output_validation_dir = tmp_path / "validation"

    _write_csv(
        input_dir / "user.csv",
        ["location_id", "is_strict_location", "id", "has_mk"],
        [
            {"location_id": "loc-1", "is_strict_location": "False", "id": "u1", "has_mk": "True"},
            {"location_id": "loc-1", "is_strict_location": "False", "id": "u2", "has_mk": "True"},
            {"location_id": "loc-1", "is_strict_location": "False", "id": "u3", "has_mk": "True"},
        ],
    )
    _write_csv(
        input_dir / "shift.csv",
        [
            "id",
            "start_at",
            "location_id",
            "task_type",
            "employer_id",
            "workplace_id",
            "need_mk",
            "id_differential",
            "hours",
            "reward",
            "capacity",
        ],
        [
            {
                "id": "s-train",
                "start_at": "2026-01-01T09:00:00+00:00",
                "location_id": "loc-1",
                "task_type": "loader",
                "employer_id": "e1",
                "workplace_id": "w1",
                "need_mk": "False",
                "id_differential": "False",
                "hours": 8,
                "reward": 1000,
                "capacity": 1,
            },
            {
                "id": "s-val",
                "start_at": "2026-01-02T09:00:00+00:00",
                "location_id": "loc-1",
                "task_type": "loader",
                "employer_id": "e1",
                "workplace_id": "w1",
                "need_mk": "False",
                "id_differential": "False",
                "hours": 8,
                "reward": 1000,
                "capacity": 1,
            },
        ],
    )
    _write_csv(
        input_dir / "event.csv",
        ["id", "shift_id", "user_id", "interaction", "ts"],
        [
            {
                "id": "apply-only",
                "shift_id": "s-val",
                "user_id": "u1",
                "interaction": "APPLY",
                "ts": "2026-01-02",
            },
            {
                "id": "finished-only",
                "shift_id": "s-val",
                "user_id": "u2",
                "interaction": "FINISHED",
                "ts": "2026-01-02",
            },
            {
                "id": "apply-before-system-cancel",
                "shift_id": "s-val",
                "user_id": "u3",
                "interaction": "APPLY",
                "ts": "2026-01-02",
            },
            {
                "id": "system-cancel",
                "shift_id": "s-val",
                "user_id": "u3",
                "interaction": "SYSTEM_CANCEL",
                "ts": "2026-01-02",
            },
        ],
    )

    summary = split_train_validation(
        input_dir=input_dir,
        output_train_dir=output_train_dir,
        output_validation_dir=output_validation_dir,
        validation_days=1,
        cutoff_date=None,
        force=False,
    )

    assert summary.validation_apply == 1
    assert _read_csv(output_validation_dir / "apply.csv") == [
        {"user_id": "u1", "shift_id": "s-val", "date": "2026-01-02"}
    ]
