# Как воспроизвести результаты (release/sleeper-rerank)

```bash
git clone https://github.com/iTonyJah/ml-hackathon.git
git branch --all
git switch release/sleeper-rerank

# Установка зависимостей
make install

# Копируем полученные файлы в data/train

# Создание валидационного сплита
# (обязательно перед eval для участника так data/validation у нас нет)
# (заказчик и проверяющий копируют файлы в data/validation самостоятельно)
poetry run python scripts/create_validation_split.py

# или для быстроты val на последних двух днях
poetry run python scripts/create_validation_split_2d.py

# Создание базы данных
make migrate

# Запуск сервиса (в отдельном терминале)
make run

# Официальный eval для участника
poetry run python -m hackaton.eval.cli run \
  --user-path data/train/user.csv \
  --shift-path data/train/shift_train.csv \
  --event-path data/train/event_train.csv \
  --val-apply-path data/validation/apply.csv \
  --val-shift-path data/validation/shift.csv \
  --val-event-path data/validation/event.csv \
  --output-dir artifacts/eval_run \
  --limit 10 \
  --batch-size 1000
```

# Официальный eval для заказчика/проверяющего
(заказчик и проверяющий копируют файлы в data/train, data/validation самостоятельно)
```bash
# Официальный eval для заказчика/проверяющего
poetry run python -m hackaton.eval.cli run \
  --user-path data/train/user.csv \
  --shift-path data/train/shift.csv \
  --event-path data/train/event.csv \
  --val-apply-path data/validation/apply.csv \
  --val-shift-path data/validation/shift.csv \
  --val-event-path data/validation/event.csv \
  --output-dir artifacts/eval_run \
  --limit 10 \
  --batch-size 1000
```

## Дополнения к инструкции

Блок выше сохранен без удаления исходного текста. Ниже добавлены уточнения, которые делают инструкцию удобнее для повторного запуска и проверки результата.

## Когда использовать этот файл

Этот how-to лучше читать как короткую боевую инструкцию: что именно нажать, чтобы получить eval-отчет.

Если нужен более полный документ с требованиями, проверками, CV и типовыми ошибками, см. `docs/REPRODUCIBILITY.md`.

## Требования перед запуском

- Python 3.12.
- Poetry.
- Свободный порт `8000`.
- Данные лежат в ожидаемых CSV-файлах.
- Команды выполняются из корня репозитория.

Проверить окружение:

```bash
poetry run python --version
poetry run python -c "import pandas, sklearn, lightgbm; print('ok')"
```

## Что должно лежать в `data/train`

Для участника перед созданием локального validation split нужны:

```text
data/train/user.csv
data/train/shift.csv
data/train/event.csv
```

После запуска:

```bash
poetry run python scripts/create_validation_split.py
```

появятся:

```text
data/train/shift_train.csv
data/train/event_train.csv
data/validation/apply.csv
data/validation/shift.csv
data/validation/event.csv
data/validation/users.csv
```

Проверить наличие файлов:

```bash
ls -lh data/train/user.csv data/train/shift.csv data/train/event.csv
ls -lh data/train/shift_train.csv data/train/event_train.csv
ls -lh data/validation/apply.csv data/validation/shift.csv data/validation/event.csv
```

## Важное различие двух eval-команд

Для участника:

- validation создается локально;
- train-часть после split лежит в `shift_train.csv` и `event_train.csv`;
- eval нужно запускать именно с этими файлами.

Для заказчика или проверяющего:

- train и validation уже подготовлены отдельно;
- можно использовать исходные `data/train/shift.csv` и `data/train/event.csv`;
- validation берется из `data/validation`.

## Более полный eval-запуск с явными сетевыми параметрами

В исходной команде не указаны `--host`, `--port`, `--predict-max-concurrency`, `--predict-max-rpm`. Это допустимо, если в CLI стоят нужные значения по умолчанию. Для воспроизводимости лучше указывать их явно.

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

Для готового train/validation:

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

## Ожидаемый результат

После успешного eval появится:

```text
artifacts/eval_run/eval_report.md
```

В отчете стоит проверить:

- `overall_target_metric`;
- дневные метрики;
- метрики по группам `capacity`;
- `predict_latency_p50_ms`;
- `predict_latency_p95_ms`;
- `predict_rpm`;
- длительность `prepare`.

## Быстрый запуск CV

CV не заменяет официальный eval, но помогает понять качество модели на более надежном time-based разбиении.

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

## Частые проблемы

### Eval не может подключиться

Скорее всего, сервис не запущен.

В отдельном терминале должен работать:

```bash
make run
```

### Порт `8000` занят

Остановите старый процесс. Если в ветке есть helper:

```bash
./kill_8000.sh
```

### `predict` возвращает `503`

Сервис еще находится в `prepare`. Нужно дождаться `ready` и проверить логи.

### Метрика неожиданно низкая или нулевая

Проверьте:

- для локального split используются `shift_train.csv` и `event_train.csv`;
- `data/validation/apply.csv` существует;
- `apply.date` соответствует дате начала смены;
- validation-смены действительно пересекаются с `apply.shift_id`;
- в validation есть оцениваемые смены с `capacity >= 2`.
