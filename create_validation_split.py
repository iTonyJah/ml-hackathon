"""
Скрипт создания валидационного сплита из тренировочных данных.

Разбиение по времени (требование хакатона):
  train: смены с start_at <= T_split
  test:  смены с start_at в интервале (T_split, T_train]

Запуск:
    poetry run python create_validation_split.py
"""

import datetime
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "new_train"
VAL_DIR = DATA_DIR / "new_validation"

# Финальная оценка — на последних 1-2 неделях.
# Новые данные заканчиваются ~2026-03-22, берём последние 2 недели как val.
SPLIT_DATE = pd.Timestamp("2026-03-08", tz="UTC")
SPLIT_DATE_PLAIN = datetime.date(2026, 3, 8)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Загружаем данные...")
    shifts = pd.read_csv(TRAIN_DIR / "shift.csv")
    events = pd.read_csv(TRAIN_DIR / "event.csv")
    users = pd.read_csv(TRAIN_DIR / "user.csv")

    shifts["start_at"] = pd.to_datetime(shifts["start_at"], utc=True)
    events["ts"] = pd.to_datetime(events["ts"], utc=True)

    s_min, s_max = shifts["start_at"].min().date(), shifts["start_at"].max().date()
    print(f"  Shifts:  {len(shifts):>7,} | {s_min} -> {s_max}")
    print(
        f"  Events:  {len(events):>7,} | {events['ts'].min().date()} -> {events['ts'].max().date()}"
    )
    print(f"  Users:   {len(users):>7,}")
    print(f"  Event types: {events['interaction'].value_counts().to_dict()}")
    return shifts, events, users


def split_shifts(shifts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # train: start_at <= T_split
    # val:   start_at >  T_split (т.е. в интервале (T_split, T_train])
    train_shifts = shifts[shifts["start_at"] <= SPLIT_DATE].copy()
    val_shifts = shifts[shifts["start_at"] > SPLIT_DATE].copy()
    print(f"\nДата разбиения: {SPLIT_DATE.date()}")
    print(f"  Train shifts: {len(train_shifts):,}  (до {SPLIT_DATE.date()} включительно)")
    print(
        f"  Val shifts:   {len(val_shifts):,}  "
        f"({val_shifts['start_at'].min().date()} -> {val_shifts['start_at'].max().date()})"
    )
    return train_shifts, val_shifts


def split_events(
    events: pd.DataFrame,
    val_shifts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Train events — все события строго до даты разбиения (нет утечки из будущего)
    train_events = events[events["ts"] <= SPLIT_DATE].copy()

    # Val events — все события в валидационном периоде (по времени, не по shift_id).
    # Evaluator загружает их день за днём в incremental prepare.
    val_shift_ids = set(val_shifts["id"].astype(str))
    val_events = events[
        (events["ts"] > SPLIT_DATE) & (events["shift_id"].astype(str).isin(val_shift_ids))
    ].copy()

    print("\nСобытия:")
    print(f"  Train events: {len(train_events):,}")
    print(f"  Val events:   {len(val_events):,}")
    return train_events, val_events


def build_apply(events: pd.DataFrame, val_shifts: pd.DataFrame) -> pd.DataFrame:
    """
    apply.csv — ground truth: пользователь реально подал заявку на смену.

    Правила (требования хакатона):
    - Только смены из val-периода (start_at > T_split)
    - Только события APPLY, произошедшие ДО start_at смены (нет утечки из будущего)
    - Исключаем смены с SYSTEM_CANCEL
    - Колонка date = shift.start_at.date() (не ts события!)
    """
    val_shift_ids = set(val_shifts["id"].astype(str))

    sys_cancel_shift_ids = set(
        events[events["interaction"] == "SYSTEM_CANCEL"]["shift_id"].astype(str)
    )

    # Словарь shift_id -> start_at для быстрого lookup
    shift_start = val_shifts.set_index(val_shifts["id"].astype(str))["start_at"]

    apply_raw = events[
        (events["interaction"] == "APPLY")
        & (events["shift_id"].astype(str).isin(val_shift_ids))
        & (~events["shift_id"].astype(str).isin(sys_cancel_shift_ids))
    ][["user_id", "shift_id", "ts"]].copy()

    apply_raw["shift_id_str"] = apply_raw["shift_id"].astype(str)
    apply_raw["shift_start_at"] = apply_raw["shift_id_str"].map(shift_start)

    # Исключаем APPLY, произошедшие ПОСЛЕ начала смены (утечка из будущего)
    apply_raw = apply_raw[apply_raw["ts"] <= apply_raw["shift_start_at"]].copy()

    apply_raw = apply_raw.drop_duplicates(subset=["user_id", "shift_id"])

    # date = shift.start_at.date() — требование evaluator и хакатона
    apply_raw["date"] = apply_raw["shift_start_at"].dt.date
    apply = apply_raw[["user_id", "shift_id", "date"]].copy()

    print("\napply.csv:")
    print(f"  Записей APPLY (val, без SYSTEM_CANCEL, без утечек): {len(apply):,}")
    print(f"  Уникальных пользователей: {apply['user_id'].nunique():,}")
    print(f"  Уникальных смен:          {apply['shift_id'].nunique():,}")
    print(f"  Диапазон дат: {apply['date'].min()} -> {apply['date'].max()}")
    print(f"  Пример:\n{apply.head(3).to_string(index=False)}")
    return apply


def validate_split(
    apply: pd.DataFrame,
    val_shifts: pd.DataFrame,
    train_events: pd.DataFrame,
    events: pd.DataFrame,
) -> None:
    print("\nПроверка корректности сплита:")

    # Все даты в apply >= минимальной даты val_shifts (граница — по времени, не по дате)
    val_min_date = val_shifts["start_at"].min().date()
    val_max_date = val_shifts["start_at"].max().date()
    bad_dates = [d for d in apply["date"] if d < val_min_date]
    if bad_dates:
        print(f"  [ОШИБКА] apply.csv содержит {len(bad_dates)} записей до {val_min_date}!")
    else:
        print(f"  [OK] Все даты в apply.csv в диапазоне [{val_min_date}, {val_max_date}]")

    # date совпадает с shift.start_at.date()
    shift_start_map = {
        str(r["id"]): r["start_at"].date() for _, r in val_shifts.iterrows()
    }
    wrong_dates = [
        (r["shift_id"], r["date"], shift_start_map.get(str(r["shift_id"])))
        for _, r in apply.iterrows()
        if r["date"] != shift_start_map.get(str(r["shift_id"]))
    ]
    if wrong_dates:
        print(f"  [ОШИБКА] {len(wrong_dates)} записей с date != shift.start_at.date():")
        for sid, got, expected in wrong_dates[:3]:
            print(f"    shift_id={sid}: date={got}, start_at.date()={expected}")
    else:
        print("  [OK] Все date в apply.csv == shift.start_at.date()")

    # Нет apply-событий после start_at смены
    apply_events = events[events["interaction"] == "APPLY"].copy()
    apply_events["shift_id_str"] = apply_events["shift_id"].astype(str)
    shift_start_ts_map = val_shifts.set_index(val_shifts["id"].astype(str))["start_at"]
    apply_events["shift_start"] = apply_events["shift_id_str"].map(shift_start_ts_map)
    future_applies = apply_events.dropna(subset=["shift_start"])
    future_applies = future_applies[future_applies["ts"] > future_applies["shift_start"]]
    val_future = future_applies[
        future_applies["shift_id_str"].isin(set(val_shifts["id"].astype(str)))
    ]
    print(
        f"  [OK] APPLY-событий после start_at смены (исключены из apply.csv): {len(val_future):,}"
    )

    # Нет утечек из будущего в train_events
    future_in_train = train_events[train_events["ts"] > SPLIT_DATE]
    if len(future_in_train) > 0:
        print(f"  [ОШИБКА] train_events содержит {len(future_in_train)} событий после split_date!")
    else:
        print("  [OK] Train events не содержат утечек из будущего")

    # Пересечение shift_id
    common = set(apply["shift_id"].astype(str)) & set(val_shifts["id"].astype(str))
    print(f"  [OK] Совпадающих shift_id в apply и val_shifts: {len(common)}")


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

    print("\nФайлы сохранены:")
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


def main() -> None:
    print("=" * 60)
    print("  Создание валидационного сплита (time-based)")
    print("=" * 60)

    shifts, events, users = load_data()
    train_shifts, val_shifts = split_shifts(shifts)
    train_events, val_events = split_events(events, val_shifts)
    apply = build_apply(events, val_shifts)
    validate_split(apply, val_shifts, train_events, events)
    save_files(train_shifts, train_events, val_shifts, val_events, apply, users)


if __name__ == "__main__":
    main()
