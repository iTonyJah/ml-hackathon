from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


POSITIVE_INTERACTIONS = {"APPLY", "FINISHED"}


@dataclass(frozen=True, slots=True)
class SplitSummary:
    cutoff_date: date
    train_users: int
    train_shifts: int
    train_events: int
    validation_shifts: int
    validation_events: int
    validation_apply: int


def _parse_date(raw: str) -> date:
    text = str(raw).strip()
    if not text:
        raise ValueError("empty datetime value")
    return datetime.fromisoformat(text.replace("Z", "+00:00")).date()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        return list(reader.fieldnames), list(reader)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _ensure_can_write(paths: list[Path], force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output files already exist: {rendered}. Use --force to overwrite.")


def _choose_cutoff_date(shifts: list[dict[str, str]], validation_days: int) -> date:
    if validation_days < 1:
        raise ValueError("--validation-days must be >= 1")
    shift_dates = sorted({_parse_date(row["start_at"]) for row in shifts})
    if len(shift_dates) < 2:
        raise ValueError("Need at least two unique shift dates to build validation split.")
    if validation_days >= len(shift_dates):
        raise ValueError(
            f"--validation-days={validation_days} leaves no train days "
            f"(available shift days: {len(shift_dates)})"
        )
    return shift_dates[-validation_days]


def split_train_validation(
    input_dir: Path,
    output_train_dir: Path,
    output_validation_dir: Path,
    validation_days: int,
    cutoff_date: date | None,
    force: bool,
) -> SplitSummary:
    user_header, users = _read_csv(input_dir / "user.csv")
    shift_header, shifts = _read_csv(input_dir / "shift.csv")
    event_header, events = _read_csv(input_dir / "event.csv")

    cutoff = cutoff_date or _choose_cutoff_date(shifts, validation_days)

    train_shifts: list[dict[str, str]] = []
    validation_shifts: list[dict[str, str]] = []
    validation_shift_ids: set[str] = set()
    validation_shift_dates: dict[str, date] = {}

    for row in shifts:
        shift_date = _parse_date(row["start_at"])
        if shift_date >= cutoff:
            validation_shifts.append(row)
            shift_id = str(row["id"])
            validation_shift_ids.add(shift_id)
            validation_shift_dates[shift_id] = shift_date
        else:
            train_shifts.append(row)

    train_events: list[dict[str, str]] = []
    validation_events: list[dict[str, str]] = []
    apply_keys: set[tuple[str, str, str]] = set()

    for row in events:
        event_date = _parse_date(row["ts"])
        shift_id = str(row["shift_id"])
        interaction = str(row["interaction"]).upper()
        if event_date < cutoff and shift_id not in validation_shift_ids:
            train_events.append(row)
        elif event_date >= cutoff:
            validation_events.append(row)
        if shift_id in validation_shift_ids and interaction in POSITIVE_INTERACTIONS:
            # The eval contract groups labels by the shift day.
            shift_day = validation_shift_dates[shift_id]
            apply_keys.add((str(row["user_id"]), shift_id, shift_day.isoformat()))

    validation_apply = [
        {"user_id": user_id, "shift_id": shift_id, "date": apply_date}
        for user_id, shift_id, apply_date in sorted(apply_keys, key=lambda x: (x[2], x[1], x[0]))
    ]

    output_paths = [
        output_train_dir / "user.csv",
        output_train_dir / "shift.csv",
        output_train_dir / "event.csv",
        output_validation_dir / "apply.csv",
        output_validation_dir / "shift.csv",
        output_validation_dir / "event.csv",
    ]
    _ensure_can_write(output_paths, force)

    _write_csv(output_train_dir / "user.csv", user_header, users)
    _write_csv(output_train_dir / "shift.csv", shift_header, train_shifts)
    _write_csv(output_train_dir / "event.csv", event_header, train_events)
    _write_csv(output_validation_dir / "shift.csv", shift_header, validation_shifts)
    _write_csv(output_validation_dir / "event.csv", event_header, validation_events)
    _write_csv(
        output_validation_dir / "apply.csv",
        ["user_id", "shift_id", "date"],
        validation_apply,
    )

    return SplitSummary(
        cutoff_date=cutoff,
        train_users=len(users),
        train_shifts=len(train_shifts),
        train_events=len(train_events),
        validation_shifts=len(validation_shifts),
        validation_events=len(validation_events),
        validation_apply=len(validation_apply),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split hackathon train CSV files into local train and validation datasets."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--output-train-dir", type=Path, default=Path("data/train_split"))
    parser.add_argument("--output-validation-dir", type=Path, default=Path("data/validation"))
    parser.add_argument(
        "--validation-days",
        type=int,
        default=14,
        help="Use the last N unique shift dates as validation when --cutoff-date is not set.",
    )
    parser.add_argument(
        "--cutoff-date",
        type=date.fromisoformat,
        default=None,
        help="First validation shift date in YYYY-MM-DD format.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = split_train_validation(
        input_dir=args.input_dir,
        output_train_dir=args.output_train_dir,
        output_validation_dir=args.output_validation_dir,
        validation_days=args.validation_days,
        cutoff_date=args.cutoff_date,
        force=args.force,
    )
    print(f"cutoff_date: {summary.cutoff_date.isoformat()}")
    print(f"train_users: {summary.train_users}")
    print(f"train_shifts: {summary.train_shifts}")
    print(f"train_events: {summary.train_events}")
    print(f"validation_shifts: {summary.validation_shifts}")
    print(f"validation_events: {summary.validation_events}")
    print(f"validation_apply: {summary.validation_apply}")


if __name__ == "__main__":
    main()
