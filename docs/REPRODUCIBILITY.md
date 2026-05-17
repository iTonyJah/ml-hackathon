# Воспроизводимость результата

Этот документ описывает, как локально воспроизвести запуск решения: подготовить данные, поднять сервис, запустить eval и time-based CV.

Команды ниже рассчитаны на ветку с финальным решением:

```bash
git switch docs/project-documentation
```

Если финальная проверка выполняется от ветки `release/sleeper-rerank`, команды остаются такими же.

## Требования

- Python 3.12.
- Poetry.
- Доступ к зависимостям из `pyproject.toml` и `poetry.lock`.
- CSV-файлы с данными в ожидаемом формате.
- Свободный порт `8000` для RPC-сервиса.

## Установка

```bash
make install
```

Команда устанавливает зависимости через Poetry.

Проверить окружение можно так:

```bash
poetry run python --version
poetry run python -c "import lightgbm, pandas, sklearn; print('ok')"
```

## Структура входных данных

Для сценария с готовыми train/validation файлами ожидаются пути:

```text
data/train/user.csv
data/train/shift.csv
data/train/event.csv

data/validation/apply.csv
data/validation/shift.csv
data/validation/event.csv
```

Для сценария, где validation split создается локально из train-данных, сначала нужны:

```text
data/train/user.csv
data/train/shift.csv
data/train/event.csv
```

После запуска split-скрипта появятся:

```text
data/train/shift_train.csv
data/train/event_train.csv
data/validation/apply.csv
data/validation/shift.csv
data/validation/event.csv
data/validation/users.csv
```

## Создание validation split

Полный split на последних неделях:

```bash
poetry run python scripts/create_validation_split.py
```

Быстрый split на последних двух днях:

```bash
poetry run python scripts/create_validation_split_2d.py
```

После выполнения нужно проверить, что появились файлы:

```bash
ls -lh data/train/shift_train.csv data/train/event_train.csv
ls -lh data/validation/apply.csv data/validation/shift.csv data/validation/event.csv
```

Важные правила split:

- train-смены идут до split-даты включительно;
- validation-смены идут после split-даты;
- `apply.date` считается как `shift.start_at.date()`;
- `APPLY` после начала смены исключаются из ground truth;
- train-события не должны содержать будущую информацию после split-даты.

## Миграция базы

```bash
make migrate
```

Команда создает SQLite-схему для пользователей, смен и событий.

## Запуск сервиса

Сервис нужно запускать отдельно и держать открытым во время eval:

```bash
make run
```

По умолчанию сервис слушает `127.0.0.1:8000`.

Если порт занят, остановите старый процесс или используйте локальный helper, если он есть в ветке:

```bash
./kill_8000.sh
```

## Запуск eval для локального split

Если validation split создавался скриптом, используйте `shift_train.csv` и `event_train.csv` как train-вход:

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

Результат:

```text
artifacts/eval_run/eval_report.md
```

## Запуск eval для готового train/validation набора

Если проверяющий отдельно кладет готовые train и validation файлы, используйте исходные `shift.csv` и `event.csv`:

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

## Запуск time-based CV

CV запускается без RPC-сервиса. Он использует тот же смысл: обучиться на прошлом, проверить на будущем.

```bash
poetry run python -m hackaton.train.cli cv \
  --user-path data/train/user.csv \
  --shift-path data/train/shift_train.csv \
  --event-path data/train/event_train.csv \
  --output-dir artifacts/cv_run \
  --val-days 30 \
  --candidate-limit 300
```

Результат:

```text
artifacts/cv_run/cv_report.md
```

Если split не создавался и нужно оценить CV на полном train-наборе, можно использовать:

```bash
poetry run python -m hackaton.train.cli cv \
  --user-path data/train/user.csv \
  --shift-path data/train/shift.csv \
  --event-path data/train/event.csv \
  --output-dir artifacts/cv_run \
  --val-days 30 \
  --candidate-limit 300
```

## Проверка тестов

```bash
make test
```

Перед сдачей также полезно выполнить:

```bash
make precommit
```

## Что должно получиться

После успешного eval:

- сервис загрузит train-данные;
- `prepare` обучит модель и построит кэши;
- eval пройдет validation-дни;
- будет создан `artifacts/eval_run/eval_report.md`;
- в отчете появятся итоговая метрика, дневные метрики, latency и RPM.

После успешного CV:

- будет создан `artifacts/cv_run/cv_report.md`;
- в отчете появятся итоговая CV-метрика, число оцениваемых дней и смен.

## Частые проблемы

### Eval не подключается к сервису

Проверьте, что в отдельном терминале запущено:

```bash
make run
```

### Порт `8000` занят

Остановите старый процесс или освободите порт:

```bash
./kill_8000.sh
```

### `predict` возвращает `503`

Это значит, что сервис еще находится в `prepare`.

Что сделать:

- дождаться `ready`;
- проверить логи сервиса;
- убедиться, что train-данные загружены;
- проверить, что `prepare` не упал.

### Eval дает нулевую или очень низкую метрику

Проверьте:

- правильные ли файлы переданы в `--shift-path` и `--event-path`;
- есть ли `data/validation/apply.csv`;
- совпадает ли `apply.date` с датой начала смены;
- не попали ли в `apply.csv` события после `shift.start_at`;
- есть ли пересечение `apply.shift_id` с `validation/shift.csv`.

### CV не запускается

Проверьте:

- существуют ли переданные CSV;
- установлены ли зависимости;
- есть ли `lightgbm`;
- достаточно ли данных в выбранном validation-окне.
