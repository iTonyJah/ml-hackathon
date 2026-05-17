# ML-хакатон: ранжирование кандидатов на смены

Репозиторий содержит сервис для ранжирования работников под смены. Сервис получает пользователей, смены и события, готовит модель и по запросу `predict` возвращает список кандидатов в порядке ожидаемой релевантности.

Финальная версия решения построена вокруг:

- RPC-сервиса на `zero`;
- SQLite-хранилища входных данных;
- `PrepareManager`, который обучает модель и строит кэши;
- `MLModel` на `LightGBM`;
- быстрого in-memory inference в `predict`;
- дополнительного sleeper-rerank для работников с `FINISHED`, но без видимых `APPLY`;
- time-based CV для более надежной внутренней оценки.

## С чего начать

Если нужно понять проект как систему, лучше читать в таком порядке:

1. `docs/FINAL_REPORT.md` - общий итоговый отчет.
2. `docs/ARCHITECTURE_OVERVIEW.md` - как устроен сервис.
3. `docs/METRICS_AND_EXPERIMENTS.md` - метрика, эксперименты и результаты.
4. `docs/DEVELOPMENT_STORY.md` - что делали и как пришли к решению.
5. `docs/REPRODUCIBILITY.md` - как воспроизвести запуск.

Корневые документы:

- `DATA.md` - формат входных CSV.
- `REGLAMENT.md` - правила eval и целевой метрики.
- `TRAIN.md` - обучение, ML-путь и CV.
- `HOW-TO.md` - практические команды.
- `CHECKLIST.md` - проверка перед сдачей.

## Быстрый запуск

Установить зависимости:

```bash
make install
```

Создать базу:

```bash
make migrate
```

Запустить сервис:

```bash
make run
```

В другом терминале можно запустить тесты:

```bash
make test
```

## Подготовка validation split

Если validation-файлы уже выданы отдельно, этот шаг не нужен.

Если нужно сделать локальный split из train-данных:

```bash
poetry run python scripts/create_validation_split.py
```

Для быстрого прогона на последних двух днях:

```bash
poetry run python scripts/create_validation_split_2d.py
```

После split появятся:

```text
data/train/shift_train.csv
data/train/event_train.csv
data/validation/apply.csv
data/validation/shift.csv
data/validation/event.csv
```

## Запуск eval

Eval требует уже запущенный сервис.

Для локального split:

```bash
poetry run python -m hackaton.eval.cli run \
  --host 127.0.0.1 \
  --port 8000 \
  --user-path data/train/user.csv \
  --shift-path data/train/shift_train.csv \
  --event-path data/train/event_train.csv \
  --val-apply-path data/validation/apply.csv \
  --val-shift-path data/validation/shift.csv \
  --val-event-path data/validation/event.csv \
  --output-dir artifacts/eval_run \
  --limit 10 \
  --batch-size 1000 \
  --predict-max-concurrency 4 \
  --predict-max-rpm 200
```

Для готового train/validation набора:

```bash
poetry run python -m hackaton.eval.cli run \
  --host 127.0.0.1 \
  --port 8000 \
  --user-path data/train/user.csv \
  --shift-path data/train/shift.csv \
  --event-path data/train/event.csv \
  --val-apply-path data/validation/apply.csv \
  --val-shift-path data/validation/shift.csv \
  --val-event-path data/validation/event.csv \
  --output-dir artifacts/eval_run \
  --limit 10 \
  --batch-size 1000 \
  --predict-max-concurrency 4 \
  --predict-max-rpm 200
```

Отчет будет сохранен в:

```text
artifacts/eval_run/eval_report.md
```

## Запуск time-based CV

CV запускается без RPC-сервиса:

```bash
poetry run python -m hackaton.train.cli cv \
  --user-path data/train/user.csv \
  --shift-path data/train/shift_train.csv \
  --event-path data/train/event_train.csv \
  --output-dir artifacts/cv_run \
  --val-days 30 \
  --candidate-limit 300
```

Отчет:

```text
artifacts/cv_run/cv_report.md
```

## Где что лежит

- `hackaton/service/app.py` - RPC-методы, включая `predict`.
- `hackaton/service/prepare_manager.py` - загрузка данных, обучение модели, кэши.
- `hackaton/service/ml_model.py` - признаки, LightGBM, скоринг и rerank.
- `hackaton/service/repositories.py` - работа с SQLite.
- `hackaton/eval` - локальный eval-пайплайн и метрика.
- `hackaton/train/cv.py` - time-based cross-validation.
- `scripts/create_validation_split.py` - основной локальный validation split.
- `scripts/create_validation_split_2d.py` - быстрый validation split.
- `tests/unit`, `tests/e2e` - тесты.

## Целевая метрика

Качество считается по ранжированию TOP-10 кандидатов.

Ключевые правила:

- фиксированный пул кандидатов: `pool_size = 10`;
- ограничение FPR: `max_fpr = min(1.0, capacity / 10)`;
- агрегация: смена -> группа `capacity` внутри дня -> день -> итог.

Подробнее: `REGLAMENT.md` и `docs/METRICS_AND_EXPERIMENTS.md`.

## Проверка качества

```bash
make test
make precommit
make load-test
```

Перед сдачей важно проверить:

- сервис стартует;
- данные загружаются;
- `prepare` завершается;
- `ready` возвращает готовность;
- `predict` возвращает непустой список;
- `eval_report.md` формируется;
- тесты проходят.
