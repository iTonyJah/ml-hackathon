#!/usr/bin/env python

"""
Interactive training pipeline for VS Code Interactive Window.

Запуск по шагам через Jupyter-ячейки (# %%):
  - В VS Code откройте этот файл
  - Выберите Python-интерпретатор с активным poetry-окружением
  - Нажимайте Shift+Enter для выполнения ячеек последовательно

Прямой запуск (выполнит все ячейки):
  poetry run python scripts/train_interactive.py
"""

# %% [0] импорт

from __future__ import annotations

import json
import logging
import pickle
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from hackaton.eval.metric import calculate_target_metric

# Импортируем компоненты из hackaton.train
from hackaton.train.training import (
    TrainConfig,
    _build_pipeline,
    _build_training_frame,
    _load_and_validate_data,
    _time_split,
)

print("Python:", sys.version)
print("Executable:", sys.executable)

# =============================================================================
# %% [0] Настройка окружения
# =============================================================================
"""
🔧 EXTENSION POINT: Если нужно изменить версию sklearn или добавить
дополнительные библиотеки (lightgbm, catboost), обновите pyproject.toml
и перезапустите интерпретатор.
"""

# Добавляем корень проекта в sys.path для импорта hackaton.train
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Подавляем предупреждения для чистого вывода
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

print(f"✅ Корень проекта: {ROOT_DIR}")
print(
    "✅ Импортированы: \
        TrainConfig, _build_pipeline, _build_training_frame, _load_and_validate_data, _time_split"
)


# =============================================================================
# %% [1] Конфигурация
# =============================================================================
"""
Настройка параметров обучения через TrainConfig.

🔧 EXTENSION POINT: Измените пути к данным или параметры модели здесь.
Для продакшена используйте CLI: poetry run python -m hackaton.train.cli train ...
"""

# Пути к данным (относительно корня проекта)
# Предполагается структура: data/train/{user,shift,event}.csv
USER_PATH = ROOT_DIR / "data" / "train" / "user.csv"
SHIFT_PATH = ROOT_DIR / "data" / "train" / "shift.csv"
EVENT_PATH = ROOT_DIR / "data" / "train" / "event.csv"

# Выходная директория для артефактов
OUTPUT_DIR = ROOT_DIR / "artifacts" / "train_debug"

# Создаём конфиг
cfg = TrainConfig(
    user_path=str(USER_PATH),
    shift_path=str(SHIFT_PATH),
    event_path=str(EVENT_PATH),
    output_dir=str(OUTPUT_DIR),
    random_state=42,
    max_iter=1000,
    test_ratio=0.2,
    skip_shap=True,  # Пропускаем SHAP для интерактивного режима
    shap_sample_size=1000,
)

print("✅ Конфигурация создана:")
print(f"   user_path: {cfg.user_path}")
print(f"   shift_path: {cfg.shift_path}")
print(f"   event_path: {cfg.event_path}")
print(f"   output_dir: {cfg.output_dir}")
print(f"   skip_shap: {cfg.skip_shap}")


# =============================================================================
# %% [2] Загрузка и валидация данных
# =============================================================================
"""
Загружает CSV-файлы и выполняет валидацию схем.

Возвращает:
  - users, shifts, events: DataFrame с очищенными данными
  - checks: словарь с результатами валидации
"""

LOGGER.info("Stage 1: Loading and validating train CSV contracts")

users, shifts, events, checks = _load_and_validate_data(cfg)

print("\n✅ Данные загружены:")
print(f"   users.shape: {users.shape}")
print(f"   shifts.shape: {shifts.shape}")
print(f"   events.shape: {events.shape}")

print("\n--- users.info() ---")
users.info()

print("\n--- shifts.info() ---")
shifts.info()

print("\n--- events.info() ---")
events.info()

print(f"\n✅ Validation checks: {json.dumps(checks, indent=2, default=str)}")


# %% [diagnose] 🔍 Диагностика фильтрации событий ← НОВАЯ ЯЧЕЙКА

print("=" * 60)
print("📊 ДИАГНОСТИКА ФИЛЬТРАЦИИ ДАННЫХ")
print("=" * 60)

# Загрузим сырые данные для сравнения
raw_events = pd.read_csv(cfg.event_path)
print(f"Сырых событий из CSV: {len(raw_events):,}")
print(f"После фильтрации:    {len(events):,}")
print(
    f"Удалено:             \
        {len(raw_events) - len(events):,} ({(1 - len(events) / len(raw_events)) * 100:.1f}%)"
)

# Проверим распределение по interaction
print("\n📈 Распределение взаимодействий:")
print(events["interaction"].value_counts())
print("\nБаланс классов:")
print(
    f"  APPLY / VIEW = {
        len(events[events['interaction'] == 'APPLY'])
        / len(events[events['interaction'] == 'VIEW'])
        * 100:.2f}%"
)

# Визуализация (опционально)
try:
    events["interaction"].value_counts().plot(kind="bar", title="Distribution of Interactions")
    plt.tight_layout()
    plt.show()
except ImportError:
    print("\n💡 Установите matplotlib для визуализации: poetry add matplotlib")

print("=" * 60)


# =============================================================================
# %% [3] Формирование признаков и таргета
# =============================================================================
"""
Строит training frame с признаками и целевой переменной.

Логика:
  - merging events с shifts (inner join по shift_id)
  - фильтрация событий после start_at (предотвращение leakage)
  - агрегация взаимодействий по паре (user_id, shift_id)
  - target=1 если есть APPLY или FINISHED
  - добавление пользовательских и парных признаков
"""

LOGGER.info("Stage 2: Building training frame and target")

frame = _build_training_frame(users, shifts, events)

print(f"✅ Training frame построен: {frame.shape}")

# Проверка распределения классов
target_counts = frame["target"].value_counts()
target_dist = (target_counts / len(frame) * 100).round(2)

print("\n📊 Распределение таргета:")
print(f"   Class 0 (no apply/finish): {target_counts.get(0, 0):,} ({target_dist.get(0, 0):.2f}%)")
print(f"   Class 1 (apply/finish):    {target_counts.get(1, 0):,} ({target_dist.get(1, 0):.2f}%)")

print("\n📋 Первые 5 строк training frame:")
display_cols = [
    "user_id",
    "shift_id",
    "target",
    "view_cnt",
    "apply_cnt",
    "finished_cnt",
    "location_match",
    "need_mk_match",
    "hours",
    "reward",
]
available_cols = [c for c in display_cols if c in frame.columns]
print(frame[available_cols].head().to_string(index=False))


# =============================================================================
# %% [4] Временной split
# =============================================================================
"""
Делит данные по времени ~80/20 без утечки.

Граница определяется по уникальным timestamp start_at.
Train: start_at < split_border
Test:  start_at >= split_border
"""

LOGGER.info("Stage 3: Time split (~80/20) without leakage")

train_frame, test_frame = _time_split(frame, cfg.test_ratio)

print("✅ Split выполнен:")
print(f"   Train rows: {len(train_frame):,}")
print(f"   Test rows:  {len(test_frame):,}")
print(f"   Ratio:      {len(test_frame) / len(frame) * 100:.1f}%")

# Проверка границ временных интервалов
train_min = train_frame["start_at"].min()
train_max = train_frame["start_at"].max()
test_min = test_frame["start_at"].min()
test_max = test_frame["start_at"].max()

print("\n🕐 Временные границы:")
print(f"   Train: [{train_min}, {train_max}]")
print(f"   Test:  [{test_min}, {test_max}]")

# Проверка отсутствия leakage (train_max < test_min)
assert train_max < test_min, "⚠️ LEAKAGE DETECTED: train_max >= test_min!"
print(f"\n✅ Leakage check passed: train_max ({train_max}) < test_min ({test_min})")


# =============================================================================
# %% [5] Инициализация и обучение модели
# =============================================================================
"""
Подготовка признаков и обучение baseline-модели (LogisticRegression).

Feature columns определены в training.py:
  - has_mk, is_strict_location (bool → int)
  - need_mk, id_differential (bool → int)
  - hours, reward, capacity (numeric)
  - location_match, need_mk_match (numeric)
  - view_cnt, user_cancel_cnt, system_cancel_cnt (counts)
  - user_hist_views, user_hist_applies, user_hist_finished (history)
  - user_finished_employer, user_finished_workplace (history)
  - task_type (categorical)

🔧 EXTENSION POINT: Замена модели на LightGBM/CatBoost
  Для использования другой модели:
  1. Импортируйте нужную модель (например, from lightgbm import LGBMClassifier)
  2. Замените _build_pipeline на свою функцию, возвращающую Pipeline
  3. Убедитесь, что интерфейс .fit(X, y) и .predict_proba(X) сохранён
"""


# Определяем список признаков (должен совпадать с training.py)
feature_columns = [
    "has_mk",
    "is_strict_location",
    "need_mk",
    "id_differential",
    "hours",
    "reward",
    "capacity",
    "location_match",
    "need_mk_match",
    "view_cnt",
    "user_cancel_cnt",
    "system_cancel_cnt",
    "user_hist_views",
    "user_hist_applies",
    "user_hist_finished",
    "user_finished_employer",
    "user_finished_workplace",
    "task_type",
]

# Проверяем наличие всех колонок
missing = [c for c in feature_columns if c not in train_frame.columns]
if missing:
    raise ValueError(f"Missing feature columns: {missing}")

numeric_features = [
    "hours",
    "reward",
    "capacity",
    "location_match",
    "need_mk_match",
    "view_cnt",
    "user_cancel_cnt",
    "system_cancel_cnt",
    "user_hist_views",
    "user_hist_applies",
    "user_hist_finished",
    "user_finished_employer",
    "user_finished_workplace",
    "has_mk",
    "is_strict_location",
    "need_mk",
    "id_differential",
]
categorical_features = ["task_type"]

# Подготовка X и y
x_train = train_frame[feature_columns].copy()
x_test = test_frame[feature_columns].copy()

# Конвертируем булевы колонки в int
for col in ["has_mk", "is_strict_location", "need_mk", "id_differential"]:
    x_train[col] = x_train[col].astype(int)
    x_test[col] = x_test[col].astype(int)

y_train = train_frame["target"].astype(int)
y_test = test_frame["target"].astype(int)

print("✅ Feature matrix prepared:")
print(f"   x_train.shape: {x_train.shape}")
print(f"   x_test.shape:  {x_test.shape}")
print(
    f"   Features: {len(feature_columns)} total ({len(numeric_features)} numeric, \
        {len(categorical_features)} categorical)"
)

# 🔧 EXTENSION POINT: Здесь можно заменить LogisticRegression на другую модель
# Пример для LightGBM:
#   from lightgbm import LGBMClassifier
#   model = LGBMClassifier(random_state=cfg.random_state, n_estimators=100)
#   pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

LOGGER.info("Stage 4: Fitting LogisticRegression baseline")

pipeline = _build_pipeline(numeric_features, categorical_features, cfg.random_state, cfg.max_iter)

print("\n📦 Pipeline structure:")
print(pipeline)

print("\n🔄 Обучение модели...")
pipeline.fit(x_train, y_train)
print("✅ Модель обучена!")


# =============================================================================
# %% [6] Оценка и сохранение артефактов
# =============================================================================
"""
Вычисление метрик на тестовой выборке и сохранение артефактов.

Артефакты:
  - model.pkl: сериализованная модель
  - metrics.json: метрики качества
  - feature_schema.json: описание признаков
  - train_config.json: конфигурация обучения
  - data_contract_check.json: результаты валидации данных
"""

LOGGER.info("Stage 5: Running inference and calculating target metric")

# Предсказания
proba = pipeline.predict_proba(x_test)[:, 1]

# Формирование DataFrame для метрики
metric_df = test_frame[["shift_id", "start_at", "capacity", "target"]].copy()
metric_df["score"] = proba

# Вычисление целевой метрики
metric_result = calculate_target_metric(metric_df)

metrics = {
    "target_metric": metric_result.target_metric,
    "evaluated_days": metric_result.evaluated_days,
    "evaluated_groups": metric_result.evaluated_groups,
    "evaluated_shifts": metric_result.evaluated_shifts,
    "day_metrics": metric_result.day_metrics,
    "test_rows": int(len(test_frame)),
    "train_rows": int(len(train_frame)),
}

print("✅ Метрики на тестовой выборке:")
print(f"   Target metric: {metrics['target_metric']:.4f}")
print(f"   Evaluated days: {metrics['evaluated_days']}")
print(f"   Evaluated shifts: {metrics['evaluated_shifts']}")

# Сохранение артефактов
output_dir = Path(cfg.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

LOGGER.info("Stage 6: Saving model and artifacts to %s", output_dir)

# model.pkl
with (output_dir / "model.pkl").open("wb") as f:
    pickle.dump(pipeline, f)
print(f"✅ Saved: {output_dir / 'model.pkl'}")

# metrics.json
(output_dir / "metrics.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"✅ Saved: {output_dir / 'metrics.json'}")

# feature_schema.json
(output_dir / "feature_schema.json").write_text(
    json.dumps(
        {
            "feature_columns": feature_columns,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "examples": x_train.head(5).to_dict(orient="records"),
        },
        ensure_ascii=False,
        indent=2,
        default=str,
    ),
    encoding="utf-8",
)
print(f"✅ Saved: {output_dir / 'feature_schema.json'}")

# train_config.json

(output_dir / "train_config.json").write_text(
    json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"✅ Saved: {output_dir / 'train_config.json'}")

# data_contract_check.json
(output_dir / "data_contract_check.json").write_text(
    json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"✅ Saved: {output_dir / 'data_contract_check.json'}")

print(f"\n🎉 Все артефакты сохранены в: {output_dir}")


# =============================================================================
# %% [7] Песочница предсказаний
# =============================================================================
"""
Загрузка сохранённой модели и анализ предсказаний на валидационном срезе.

Использование:
  - Загрузить model.pkl
  - Вызвать predict_proba на тестовых данных
  - Проанализировать распределение вероятностей
  - Найти топ-N кандидатов для конкретных смен
"""

print("🔍 Loading saved model...")
with (output_dir / "model.pkl").open("rb") as f:
    loaded_pipeline = pickle.load(f)
print("✅ Model loaded successfully!")

# Предсказания на тестовой выборке
test_proba = loaded_pipeline.predict_proba(x_test)[:, 1]
x_test_with_proba = x_test.copy()
x_test_with_proba["predicted_proba"] = test_proba
x_test_with_proba["actual_target"] = y_test.values

print("\n📊 Статистика предсказанных вероятностей:")
print(x_test_with_proba["predicted_proba"].describe())

print("\n📈 Распределение вероятностей по классам:")
print(x_test_with_proba.groupby("actual_target")["predicted_proba"].describe())

# Топ-10 самых уверенных предсказаний класса 1
top_positive = x_test_with_proba.nlargest(10, "predicted_proba")
print("\n🏆 Top-10 highest probability predictions:")
print(top_positive[["predicted_proba", "actual_target"]].to_string(index=False))

# Анализ конкретной смены (если есть данные)
if len(x_test) > 0:
    sample_shift_idx = 0
    sample_proba = test_proba[sample_shift_idx]
    sample_actual = y_test.iloc[sample_shift_idx]
    print(f"\n📋 Пример предсказания для строки #{sample_shift_idx}:")
    print(f"   Predicted probability: {sample_proba:.4f}")
    print(f"   Actual target: {sample_actual}")
    print(f"   Prediction {'correct' if (sample_proba > 0.5) == sample_actual else 'incorrect'}")

# 🔧 EXTENSION POINT: Интеграция с сервисом предсказаний
# Для использования в production:
#   from hackaton.service.ml_model import MLModel
#   ml_model = MLModel()
#   ml_model.load_from_pickle(output_dir / "model.pkl")
#   scores = ml_model.predict_scores(user_ids, users_cache, shift_dict)

print("\n✅ Sandbox готов к экспериментам!")
print("   Используйте loaded_pipeline для новых предсказаний.")


# =============================================================================
# %% [8] Заключение и следующие шаги
# =============================================================================
"""
Обучение завершено успешно!

Следующие шаги:
  1. Проверьте артефакты в папке: {output_dir}
  2. Проанализируйте метрики в metrics.json
  3. При необходимости замените модель (см. EXTENSION POINT в ячейке [5])
  4. Для запуска CV используйте: poetry run python -m hackaton.train.cli cv ...

Полезные команды:
  - Просмотр метрик: cat artifacts/train_debug/metrics.json
  - Запуск с другими параметрами: poetry run python -m hackaton.train.cli train --help
"""

print("=" * 60)
print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"""
Артефакты сохранены в: {output_dir}

Ключевые файлы:
  - model.pkl              : Обученная модель
  - metrics.json           : Метрики качества
  - feature_schema.json    : Описание признаков
  - train_config.json      : Конфигурация
  - data_contract_check.json: Валидация данных

Метрика: {metrics["target_metric"]:.4f}
""")


# =============================================================================
# Main entry point
# =============================================================================
if __name__ == "__main__":
    """
    При запуске как скрипт (poetry run python scripts/train_interactive.py)
    выполняются все ячейки последовательно.

    Для интерактивного режима в VS Code:
      1. Откройте этот файл в VS Code
      2. Убедитесь, что выбран правильный Python-интерпретатор (poetry)
      3. Нажимайте Shift+Enter для выполнения ячеек по порядку
      4. Используйте Variable Explorer для просмотра переменных
    """
    print(__doc__)
    print("\nЗапуск полного пайплайна...\n")
    # Все ячейки уже выполнены при импорте этого файла
    print("\n✅ Полный пайплайн выполнен!")
# %%
