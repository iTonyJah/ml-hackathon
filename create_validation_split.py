"""
Скрипт создания валидационного сплита из тренировочных данных.
Разбивает данные по времени: train до указанной даты, val — после.

Важно: этот скрипт реализует временное разбиение (time-based split),
что критически важно для ML-задач с временными рядами, чтобы избежать
утечки данных из будущего в тренировочную выборку.

Запуск:
    poetry run python create_validation_split.py
    
Или с кастомной датой сплита:
    poetry run python create_validation_split.py --split-date 2026-02-15
"""

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Tuple, Set

import pandas as pd

# Настраиваем логирование для удобного отслеживания выполнения скрипта
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Базовые директории для данных
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"

# Глобальные переменные будут перезаписаны в main() на основе аргументов
SPLIT_DATE: pd.Timestamp = pd.Timestamp("2026-02-15", tz="UTC")
SPLIT_DATE_PLAIN: date = date(2026, 2, 15)


def parse_args() -> argparse.Namespace:
    """
    Парсит аргументы командной строки.
    
    Returns:
        Пространство имён с аргументами
    """
    parser = argparse.ArgumentParser(
        description="Создание валидационного сплита с временным разделением",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split-date",
        type=str,
        default="2026-02-15",
        help="Дата разбиения в формате YYYY-MM-DD. Train будет до этой даты, val — начиная с неё.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Базовая директория с данными",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Только валидация существующих файлов, без создания новых",
    )
    return parser.parse_args()


def load_data(train_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Загружает исходные данные из CSV файлов.
    
    Проверяет наличие всех необходимых файлов и колонок.
    Конвертирует временные колонки в datetime с UTC timezone.
    
    Args:
        train_dir: Директория с тренировочными данными
        
    Returns:
        Кортеж из трёх DataFrame: (shifts, events, users)
        
    Raises:
        FileNotFoundError: Если какой-то из файлов не найден
        KeyError: Если отсутствуют обязательные колонки
    """
    logger.info("Загружаем данные...")
    
    # Определяем пути к файлам
    shifts_path = train_dir / "shift.csv"
    events_path = train_dir / "event.csv"
    users_path = train_dir / "user.csv"
    
    # Проверяем существование файлов
    for file_path in [shifts_path, events_path, users_path]:
        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    # Загружаем данные
    shifts = pd.read_csv(shifts_path)
    events = pd.read_csv(events_path)
    users = pd.read_csv(users_path)
    
    # Проверяем обязательные колонки
    required_shift_cols = {"id", "start_at"}
    required_event_cols = {"shift_id", "ts", "interaction", "user_id"}
    required_user_cols = {"id"}
    
    if not required_shift_cols.issubset(shifts.columns):
        missing = required_shift_cols - set(shifts.columns)
        raise KeyError(f"В shift.csv отсутствуют колонки: {missing}")
    
    if not required_event_cols.issubset(events.columns):
        missing = required_event_cols - set(events.columns)
        raise KeyError(f"В event.csv отсутствуют колонки: {missing}")
    
    if not required_user_cols.issubset(users.columns):
        missing = required_user_cols - set(users.columns)
        raise KeyError(f"В user.csv отсутствуют колонки: {missing}")
    
    # Конвертируем временные колонки в datetime с указанием UTC timezone
    shifts["start_at"] = pd.to_datetime(shifts["start_at"], utc=True)
    events["ts"] = pd.to_datetime(events["ts"], utc=True)
    
    # Выводим статистику по загруженным данным
    s_min, s_max = shifts["start_at"].min().date(), shifts["start_at"].max().date()
    e_min, e_max = events["ts"].min().date(), events["ts"].max().date()
    
    logger.info(f"  Shifts:  {len(shifts):>7,} записей | {s_min} -> {s_max}")
    logger.info(f"  Events:  {len(events):>7,} записей | {e_min} -> {e_max}")
    logger.info(f"  Users:   {len(users):>7,} записей")
    
    # Показываем распределение типов событий
    event_types = events["interaction"].value_counts().to_dict()
    logger.info(f"  Типы событий: {event_types}")
    
    return shifts, events, users


def split_shifts(
    shifts: pd.DataFrame, 
    split_date: pd.Timestamp
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разбивает смены на тренировочные и валидационные по дате.
    
    Важно: используем строгое неравенство (<) для train и (>=) для val,
    чтобы избежать перекрытия и утечки данных.
    
    Args:
        shifts: DataFrame со всеми сменами
        split_date: Дата и время разбиения
        
    Returns:
        Кортеж (train_shifts, val_shifts)
    """
    # Train: смены, которые начались ДО даты разбиения
    train_shifts = shifts[shifts["start_at"] < split_date].copy()
    
    # Val: смены, которые начались ПОСЛЕ или в дату разбиения
    val_shifts = shifts[shifts["start_at"] >= split_date].copy()
    
    logger.info(f"\nДата разбиения: {split_date.date()}")
    logger.info(f"  Train смены: {len(train_shifts):,}")
    logger.info(f"  Val смены:   {len(val_shifts):,}")
    
    # Проверяем, что нет пустых датасетов
    if len(train_shifts) == 0:
        logger.warning("⚠️  Train набор пуст! Проверьте дату разбиения.")
    if len(val_shifts) == 0:
        logger.warning("⚠️  Validation набор пуст! Проверьте дату разбиения.")
    
    return train_shifts, val_shifts


def get_shift_ids_as_set(shifts: pd.DataFrame, id_column: str = "id") -> Set[str]:
    """
    Создаёт множество ID смен в строковом представлении.
    
    Используется для быстрой фильтрации событий по принадлежности к смене.
    Преобразование к строке необходимо для корректного сравнения,
    т.к. в событиях shift_id может быть сохранён как строка.
    
    Args:
        shifts: DataFrame со сменами
        id_column: Название колонки с ID
        
    Returns:
        Множество строк с ID смен
    """
    return set(shifts[id_column].astype(str))


def split_events(
    events: pd.DataFrame,
    train_shifts: pd.DataFrame,
    val_shifts: pd.DataFrame,
    split_date: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разбивает события на тренировочные и валидационные.
    
    Критически важно для предотвращения утечки данных:
    - Train события: ВСЕ события строго до split_date (независимо от смены)
    - Val события: Только события из валидационных смен И после split_date
    
    Исправление ошибки оригинальной версии: в оригинале val_events могли содержать
    события до split_date, если они принадлежали валидационным сменам.
    
    Args:
        events: DataFrame со всеми событиями
        train_shifts: Тренировочные смены (не используются напрямую, но нужны для логики)
        val_shifts: Валидационные смены
        split_date: Дата и время разбиения
        
    Returns:
        Кортеж (train_events, val_events)
    """
    # Получаем множество ID валидационных смен для быстрой фильтрации
    val_shift_ids: Set[str] = get_shift_ids_as_set(val_shifts)
    
    # Train events: ВСЕ события строго до даты разбиения
    # Это гарантирует отсутствие утечки из будущего
    train_events = events[events["ts"] < split_date].copy()
    
    # Val events: события из валидационных смен И после даты разбиения
    # Двойная фильтрация важна: и по смене, и по времени
    val_events_mask = (
        events["shift_id"].astype(str).isin(val_shift_ids) & 
        (events["ts"] >= split_date)
    )
    val_events = events[val_events_mask].copy()
    
    logger.info("События:")
    logger.info(f"  Train события: {len(train_events):,}")
    logger.info(f"  Val события:   {len(val_events):,}")
    
    # Предупреждение, если val_events пуст
    if len(val_events) == 0:
        logger.warning("⚠️  Validation события пусты! Проверьте логику разбиения.")
    
    return train_events, val_events


def build_apply(
    events: pd.DataFrame, 
    val_shifts: pd.DataFrame,
    split_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Создает файл apply.csv — факты записи пользователя на смену (положительные примеры).
    
    Бизнес-правила формирования:
    1. Только смены из валидационного периода (start_at >= split_date)
    2. Только события APPLY после даты разбиения (ts >= split_date)
    3. Исключаем смены, которые были отменены системой (SYSTEM_CANCEL)
    4. Убираем дубликаты (один пользователь — одна запись на смену)
    5. Формат: user_id, shift_id, date (evaluator ожидает именно 'date')
    
    Args:
        events: DataFrame со всеми событиями
        val_shifts: Валидационные смены
        split_date: Дата и время разбиения
        
    Returns:
        DataFrame с колонками [user_id, shift_id, date]
    """
    # Получаем множества ID для фильтрации
    val_shift_ids: Set[str] = get_shift_ids_as_set(val_shifts)
    
    # Находим смены, которые были отменены системой
    # Эти смены исключаем из apply, т.к. они не состоялись
    sys_cancel_mask = events["interaction"] == "SYSTEM_CANCEL"
    sys_cancel_shift_ids: Set[str] = get_shift_ids_as_set(
        events[sys_cancel_mask], 
        id_column="shift_id"
    )
    
    # Фильтруем события по всем критериям
    apply_mask = (
        (events["interaction"] == "APPLY") &                    # Только APPLY
        (events["shift_id"].astype(str).isin(val_shift_ids)) &  # Только val смены
        (~events["shift_id"].astype(str).isin(sys_cancel_shift_ids)) &  # Не отменены
        (events["ts"] >= split_date)                             # После split_date
    )
    
    apply = events[apply_mask][["user_id", "shift_id", "ts"]].copy()
    
    # Убираем дубликаты: один пользователь может много раз нажать APPLY на одну смену
    apply = apply.drop_duplicates(subset=["user_id", "shift_id"], keep="first")
    
    # Конвертируем timestamp в date (evaluator ожидает колонку 'date')
    apply["date"] = apply["ts"].dt.date
    
    # Оставляем только нужные колонки в правильном порядке
    apply = apply[["user_id", "shift_id", "date"]]
    
    # Логгируем статистику
    logger.info("apply.csv:")
    logger.info(f"  Записей APPLY (val, без SYSTEM_CANCEL): {len(apply):,}")
    logger.info(f"  Уникальных пользователей: {apply['user_id'].nunique():,}")
    logger.info(f"  Уникальных смен:          {apply['shift_id'].nunique():,}")
    
    if len(apply) > 0:
        logger.info(f"  Диапазон дат: {apply['date'].min()} -> {apply['date'].max()}")
        logger.info(f"  Пример данных:\n{apply.head(3).to_string(index=False)}")
    else:
        logger.warning("⚠️  apply.csv пуст! Проверьте критерии фильтрации.")
    
    return apply


def save_files(
    train_shifts: pd.DataFrame,
    train_events: pd.DataFrame,
    val_shifts: pd.DataFrame,
    val_events: pd.DataFrame,
    apply: pd.DataFrame,
    users: pd.DataFrame,
    train_dir: Path,
    val_dir: Path,
) -> None:
    """
    Сохраняет все датасеты в соответствующие директории.
    
    Создаёт директорию validation, если она не существует.
    Выводит размеры сохранённых файлов для контроля.
    
    Args:
        train_shifts: Тренировочные смены
        train_events: Тренировочные события
        val_shifts: Валидационные смены
        val_events: Валидационные события
        apply: Факты записи на смены
        users: Пользователи (копируются полностью)
        train_dir: Директория для тренировочных файлов
        val_dir: Директория для валидационных файлов
    """
    # Создаём директорию для валидации, если не существует
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем тренировочные файлы
    train_shifts.to_csv(train_dir / "shift_train.csv", index=False)
    train_events.to_csv(train_dir / "event_train.csv", index=False)
    
    # Сохраняем валидационные файлы
    val_shifts.to_csv(val_dir / "shift.csv", index=False)
    val_events.to_csv(val_dir / "event.csv", index=False)
    apply.to_csv(val_dir / "apply.csv", index=False)
    users.to_csv(val_dir / "users.csv", index=False)
    
    logger.info("\nФайлы сохранены:")
    
    # Список всех сохранённых файлов для отчёта
    saved_files = [
        train_dir / "shift_train.csv",
        train_dir / "event_train.csv",
        val_dir / "shift.csv",
        val_dir / "event.csv",
        val_dir / "apply.csv",
        val_dir / "users.csv",
    ]
    
    # Выводим размер каждого файла в КБ
    for file_path in saved_files:
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            rel_path = file_path.relative_to(BASE_DIR)
            logger.info(f"  {rel_path:<40s} ({size_kb:>8.1f} KB)")
        else:
            logger.error(f"  ⚠️  Файл не был сохранён: {file_path}")


def validate_split(
    apply: pd.DataFrame,
    val_shifts: pd.DataFrame,
    train_events: pd.DataFrame,
    split_date_plain: date,
) -> bool:
    """
    Проверяет корректность выполненного разбиения.
    
    Выполняет три ключевые проверки:
    1. В apply нет дат раньше split_date (утечка из прошлого)
    2. Все shift_id из apply присутствуют в val_shifts (целостность)
    3. В train_events нет событий после split_date (утечка из будущего)
    
    Args:
        apply: DataFrame с фактами записи
        val_shifts: Валидационные смены
        train_events: Тренировочные события
        split_date_plain: Дата разбиения как date объект
        
    Returns:
        True если все проверки пройдены, иначе False
    """
    logger.info("\n" + "="*50)
    logger.info("Проверка корректности сплита:")
    logger.info("="*50)
    
    all_checks_passed = True
    
    # Проверка 1: Даты в apply.csv не раньше split_date
    if len(apply) > 0:
        bad_dates_mask = apply["date"] < split_date_plain
        bad_dates_count = bad_dates_mask.sum()
        
        if bad_dates_count > 0:
            logger.error(f"  ❌ [ОШИБКА] apply.csv содержит {bad_dates_count} записей до split_date!")
            all_checks_passed = False
        else:
            logger.info(f"  ✅ [OK] Все даты в apply.csv >= {split_date_plain}")
    else:
        logger.warning("  ⚠️  Пропуск проверки дат: apply.csv пуст")
    
    # Проверка 2: Все shift_id из apply присутствуют в val_shifts
    if len(apply) > 0 and len(val_shifts) > 0:
        apply_shift_ids = set(apply["shift_id"].astype(str))
        val_shift_ids = set(val_shifts["id"].astype(str))
        common_shifts = apply_shift_ids & val_shift_ids
        
        # Все shift_id из apply должны быть в val_shifts
        if apply_shift_ids == common_shifts:
            logger.info(f"  ✅ [OK] Все shift_id из apply присутствуют в val_shifts: {len(common_shifts)}")
        else:
            missing = apply_shift_ids - common_shifts
            logger.error(f"  ❌ [ОШИБКА] {len(missing)} shift_id из apply отсутствуют в val_shifts!")
            all_checks_passed = False
    else:
        logger.warning("  ⚠️  Пропуск проверки shift_id: один из датасетов пуст")
    
    # Проверка 3: В train_events нет утечек из будущего
    if len(train_events) > 0:
        future_events_mask = train_events["ts"] >= pd.Timestamp(split_date_plain, tz="UTC")
        future_events_count = future_events_mask.sum()
        
        if future_events_count > 0:
            logger.error(f"  ❌ [ОШИБКА] train_events содержит {future_events_count} событий после split_date!")
            logger.error("     Это критическая утечка данных! Модель будет обучаться на будущем.")
            all_checks_passed = False
        else:
            logger.info("  ✅ [OK] Train events не содержат утечек из будущего")
    else:
        logger.warning("  ⚠️  Пропуск проверки утечек: train_events пуст")
    
    # Итоговый вывод
    if all_checks_passed:
        logger.info("\n✅ Все проверки пройдены успешно!")
    else:
        logger.error("\n❌ Обнаружены ошибки в разбиении данных!")
    
    return all_checks_passed


def main() -> int:
    """
    Главная функция скрипта.
    
    Последовательно выполняет:
    1. Парсинг аргументов
    2. Загрузку данных
    3. Разбиение на train/val
    4. Формирование apply.csv
    5. Валидацию результатов
    6. Сохранение файлов
    
    Returns:
        0 при успешном выполнении, 1 при ошибке
    """
    print("=" * 60)
    print("  Создание валидационного сплита (Time-Based Split)")
    print("=" * 60)
    
    try:
        # Парсим аргументы командной строки
        args = parse_args()
        
        # Обновляем глобальные переменные даты разбиения
        global SPLIT_DATE, SPLIT_DATE_PLAIN
        SPLIT_DATE = pd.Timestamp(args.split_date, tz="UTC")
        SPLIT_DATE_PLAIN = SPLIT_DATE.date()
        
        logger.info(f"Дата разбиения: {SPLIT_DATE_PLAIN}")
        logger.info(f"Директория данных: {args.data_dir}")
        
        # Обновляем пути к директориям
        train_dir = args.data_dir / "train"
        val_dir = args.data_dir / "validation"
        
        # Загружаем данные
        shifts, events, users = load_data(train_dir)
        
        # Разбиваем смены
        train_shifts, val_shifts = split_shifts(shifts, SPLIT_DATE)
        
        # Разбиваем события (с исправленной логикой)
        train_events, val_events = split_events(
            events, 
            train_shifts, 
            val_shifts,
            SPLIT_DATE,
        )
        
        # Формируем apply.csv
        apply = build_apply(events, val_shifts, SPLIT_DATE)
        
        # Валидируем результаты
        is_valid = validate_split(apply, val_shifts, train_events, SPLIT_DATE_PLAIN)
        
        if not is_valid:
            logger.error("\n⚠️  Валидация не пройдена! Файлы не будут сохранены.")
            return 1
        
        # Сохраняем файлы
        save_files(
            train_shifts,
            train_events,
            val_shifts,
            val_events,
            apply,
            users,
            train_dir,
            val_dir,
        )
        
        logger.info("\n✅ Скрипт завершён успешно!")
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"❌ Файл не найден: {e}")
        return 1
    except KeyError as e:
        logger.error(f"❌ Отсутствует обязательная колонка: {e}")
        return 1
    except Exception as e:
        logger.error(f"❌ Неожиданная ошибка: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
