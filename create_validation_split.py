"""
Скрипт создания валидационного сплита из тренировочных данных.
Разбивает данные по времени: train до 15 февраля, val — последние 2 недели февраля.

Запуск:
    poetry run python create_validation_split.py
"""

import datetime
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"

SPLIT_DATE = pd.Timestamp("2026-02-15", tz="UTC")
SPLIT_DATE_PLAIN = datetime.date(2026, 2, 15)  # для сравнения с date() объектами

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Загружаем данные...")
    shifts = pd.read_csv(TRAIN_DIR / "shift.csv")
    events = pd.read_csv(TRAIN_DIR / "event.csv")
    users = pd.read_csv(TRAIN_DIR / "user.csv")

    shifts["start_at"] = pd.to_datetime(shifts["start_at"], utc=True)
    events["ts"] = pd.to_datetime(events["ts"], utc=True)

    print(f"  Shifts:  {len(shifts):>7,} | {shifts['start_at'].min().date()} -> {shifts['start_at'].max().date()}")
    print(f"  Events:  {len(events):>7,} | {events['ts'].min().date()} -> {events['ts'].max().date()}")
    print(f"  Users:   {len(users):>7,}")
    print(f"  Event types: {events['interaction'].value_counts().to_dict()}")
    return shifts, events, users


def split_shifts(shifts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_shifts = shifts[shifts["start_at"] < SPLIT_DATE].copy()
    val_shifts = shifts[shifts["start_at"] >= SPLIT_DATE].copy()
    print(f"\nДата разбиения: {SPLIT_DATE.date()}")
    print(f"  Train shifts: {len(train_shifts):,}")
    print(f"  Val shifts:   {len(val_shifts):,}")
    return train_shifts, val_shifts


def split_events(
    events: pd.DataFrame,
    train_shifts: pd.DataFrame,
    val_shifts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_shift_ids = set(val_shifts["id"].astype(str))

    # Train events — все события строго до даты разбиения (без утечки)
    train_events = events[events["ts"] < SPLIT_DATE].copy()

    # Val events — события по валидационным сменам
    val_events = events[events["shift_id"].astype(str).isin(val_shift_ids)].copy()

    print(f"События:")
    print(f"  Train events: {len(train_events):,}")
    print(f"  Val events:   {len(val_events):,}")
    return train_events, val_events

def build_apply(events: pd.DataFrame, val_shifts: pd.DataFrame) -> pd.DataFrame:
    """
    apply.csv — факт записи пользователя на смену (y=1).
    Правила:
    - Только смены из валидационного периода (start_at >= SPLIT_DATE)
    - Только события APPLY после даты разбиения (ts >= SPLIT_DATE)
    - Исключаем смены с SYSTEM_CANCEL
    - Колонки: user_id, shift_id, date  <-- evaluator ожидает именно 'date'
    """
    val_shift_ids = set(val_shifts["id"].astype(str))

    sys_cancel_shift_ids = set(
        events[events["interaction"] == "SYSTEM_CANCEL"]["shift_id"].astype(str)
    )

    apply = (
        events[
            (events["interaction"] == "APPLY")
            & (events["shift_id"].astype(str).isin(val_shift_ids))
            & (~events["shift_id"].astype(str).isin(sys_cancel_shift_ids))
            & (events["ts"] >= SPLIT_DATE)
        ][["user_id", "shift_id", "ts"]]
        .drop_duplicates(subset=["user_id", "shift_id"])
        .copy()
    )

    # Evaluator ожидает колонку 'date', не 'ts'
    apply["date"] = apply["ts"].dt.date
    apply = apply[["user_id", "shift_id", "date"]]

    print(f"apply.csv:")
    print(f"  Записей APPLY (val, без SYSTEM_CANCEL): {len(apply):,}")
    print(f"  Уникальных пользователей: {apply['user_id'].nunique():,}")
    print(f"  Уникальных смен:          {apply['shift_id'].nunique():,}")
    print(f"  Диапазон дат: {apply['date'].min()} -> {apply['date'].max()}")
    print(f"  Колонки: {apply.columns.tolist()}")
    print(f"  Пример:{apply.head(3).to_string(index=False)}")
    return apply


def save_files(
    train_shifts: pd.DataFrame,
    train_events: pd.DataFrame,
    val_shifts: pd.DataFrame,
    val_events: pd.DataFrame,
    apply: pd.DataFrame,
    users: pd.DataFrame,
) -> None:
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    train_shifts.to_csv(TRAIN_DIR / "shift_train.csv", index=False)
    train_events.to_csv(TRAIN_DIR / "event_train.csv", index=False)

    val_shifts.to_csv(VAL_DIR / "shift.csv", index=False)
    val_events.to_csv(VAL_DIR / "event.csv", index=False)
    apply.to_csv(VAL_DIR / "apply.csv", index=False)
    users.to_csv(VAL_DIR / "users.csv", index=False)

    print(f"Файлы сохранены:")
    for path in [
        TRAIN_DIR / "shift_train.csv",
        TRAIN_DIR / "event_train.csv",
        VAL_DIR / "shift.csv",
        VAL_DIR / "event.csv",
        VAL_DIR / "apply.csv",
        VAL_DIR / "users.csv",
    ]:
        size_kb = path.stat().st_size / 1024
        print(f"  {path.relative_to(BASE_DIR)}  ({size_kb:.1f} KB)")


def validate_split(
    apply: pd.DataFrame,
    val_shifts: pd.DataFrame,
    train_events: pd.DataFrame,
) -> None:
    print(f"\nПроверка корректности сплита:")

    # apply['date'] — python date объекты, сравниваем с datetime.date
    bad_dates = [d for d in apply["date"] if d < SPLIT_DATE_PLAIN]
    if bad_dates:
        print(f"  [ОШИБКА] apply.csv содержит {len(bad_dates)} записей до split_date!")
    else:
        print(f"  [OK] Все даты в apply.csv >= {SPLIT_DATE_PLAIN}")

    # Пересечение shift_id между apply и val_shifts
    common = set(apply["shift_id"].astype(str)) & set(val_shifts["id"].astype(str))
    print(f"  [OK] Совпадающих shift_id в apply и val_shifts: {len(common)}")

    # Утечки из будущего в train_events
    future_events = train_events[train_events["ts"] >= SPLIT_DATE]
    if len(future_events) > 0:
        print(f"  [ОШИБКА] train_events содержит {len(future_events)} событий после split_date!")
    else:
        print(f"  [OK] Train events не содержат утечек из будущего")


def main() -> None:
    print("=" * 55)
    print("  Создание валидационного сплита")
    print("=" * 55)

    shifts, events, users = load_data()
    train_shifts, val_shifts = split_shifts(shifts)
    train_events, val_events = split_events(events, train_shifts, val_shifts)
    apply = build_apply(events, val_shifts)
    validate_split(apply, val_shifts, train_events)
    save_files(train_shifts, train_events, val_shifts, val_events, apply, users)

if __name__ == "__main__":
    main()