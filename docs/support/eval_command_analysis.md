# Анализ команды запуска eval

Команда:

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
  --predict-max-concurrency 4 \
  --predict-max-rpm 200
```

Эта команда запускает локальную оценку качества уже поднятого RPC-сервиса. Она не обучает модель напрямую. Она подключается к сервису на `127.0.0.1:8000`, загружает train-данные, вызывает `prepare`, `ready`, `predict`, прогоняет validation-дни, считает метрику и пишет отчет в `artifacts/eval_run/eval_report.md`.

## Где реализовано

- CLI-обертка: `hackaton/eval/cli.py`
- основная логика eval-пайплайна: `hackaton/eval/evaluator.py`
- расчет метрики: `hackaton/eval/metric.py`
- RPC-сервис, к которому обращается evaluator: `hackaton/service/app.py`

## Что означает начало команды

```bash
poetry run python -m hackaton.eval.cli run
```

`poetry run` запускает Python внутри Poetry-окружения проекта, то есть с зависимостями из `pyproject.toml` и `poetry.lock`.

`python -m hackaton.eval.cli` запускает модуль `hackaton/eval/cli.py` как программу.

`run` выбирает Click-команду `run`, объявленную в `hackaton/eval/cli.py`.

CLI собирает параметры в объект `EvalConfig` и вызывает:

```python
summary = run_evaluation(cfg)
```

## Подключение к сервису

```bash
--host 127.0.0.1
--port 8000
```

Evaluator создает `ZeroClient` и подключается к RPC-сервису:

```python
client = ZeroClient(cfg.host, cfg.port, default_timeout=cfg.rpc_timeout_ms)
```

Важно: сервис должен быть уже запущен отдельно, например:

```bash
make run
```

`make run` запускает:

```bash
poetry run python -m hackaton.service.main
```

Сам eval-сценарий сервер не поднимает.

## Используемые данные

Train-файлы:

```bash
--user-path data/train/user.csv
--shift-path data/train/shift.csv
--event-path data/train/event.csv
```

Они загружаются как стартовое состояние системы:

- пользователи;
- исторические смены;
- исторические события.

Validation-файлы:

```bash
--val-apply-path data/validation/apply.csv
--val-shift-path data/validation/shift.csv
--val-event-path data/validation/event.csv
```

Они используются для дневной симуляции оценки:

- `validation/shift.csv` - смены, для которых нужно получить рекомендации;
- `validation/apply.csv` - фактические отклики пользователей, то есть ground truth;
- `validation/event.csv` - события дня, которые после оценки дня догружаются в сервис.

При чтении CSV даты приводятся к нужным типам:

- `shift.start_at` -> `datetime`;
- `event.ts` -> `datetime`;
- `apply.date` -> `date`.

## Основной сценарий выполнения

### 1. Проверка лимитов

В начале `run_evaluation` проверяет:

```python
if cfg.predict_max_rpm > 200:
    raise ValueError("predict_max_rpm must be <= 200 to satisfy evaluation limits")

if cfg.predict_max_concurrency < 1:
    raise ValueError("predict_max_concurrency must be >= 1")
```

В переданной команде:

- `--predict-max-rpm 200` - максимально разрешенное значение;
- `--predict-max-concurrency 4` - допустимое значение.

### 2. Создание выходной директории

```python
output_dir.mkdir(parents=True, exist_ok=True)
```

Для данной команды будет создана директория:

```text
artifacts/eval_run
```

### 3. Загрузка CSV

Evaluator читает:

- `data/train/user.csv`;
- `data/train/shift.csv`;
- `data/train/event.csv`;
- `data/validation/apply.csv`;
- `data/validation/shift.csv`;
- `data/validation/event.csv`.

### 4. Загрузка train-данных в сервис

Train-данные отправляются в RPC-сервис батчами:

- users через RPC-метод `user`;
- shifts через RPC-метод `shift`;
- events через RPC-метод `event`.

По умолчанию размер батча равен `1000`, потому что параметр `--batch-size` в команде не указан.

### 5. Первичный prepare

После загрузки train-данных evaluator вызывает:

```python
client.call("prepare", None)
```

Затем в цикле опрашивает:

```python
client.call("ready", None)
```

Пока сервис не вернет готовность.

Таймаут первичного prepare по умолчанию равен `1200` секунд, то есть 20 минут.

### 6. Разбиение validation по дням

Evaluator берет все уникальные даты из `val_apply["date"]`:

```python
eval_days = sorted(d for d in val_apply["date"].dropna().unique())
```

Затем идет по этим датам по порядку.

### 7. Подготовка данных конкретного дня

Для каждого дня выбираются:

```python
day_shifts = val_shifts[val_shifts["start_at"].dt.date == day].copy()
day_apply = val_apply[val_apply["date"] == day].copy()
```

`day_shifts` - смены, для которых нужно получить рекомендации.

`day_apply` - фактические пары `user_id` и `shift_id`, которые считаются правильными ответами.

Если в конкретный день нет смен, день пропускается.

### 8. Вызовы predict

Для каждой смены evaluator вызывает RPC-метод `predict`.

Payload концептуально выглядит так:

```python
{
    "shift": {...данные смены...},
    "limit": 10,
}
```

`limit` по умолчанию равен `10`, потому что параметр `--limit` в команде не указан.

На стороне сервиса `predict`:

1. Проверяет, готова ли модель.
2. Валидирует payload.
3. Получает кандидатов.
4. Если модель обучена, скорит кандидатов моделью.
5. Возвращает список `user_ids`.

Если модель не готова, сервис может вернуть `503`. Evaluator в таком случае несколько раз ждет `ready` и повторяет запрос.

### 9. Конкурентность predict

```bash
--predict-max-concurrency 4
```

Этот параметр означает, что запросы `predict` отправляются через `ThreadPoolExecutor` максимум в 4 потока.

То есть evaluator может одновременно держать до 4 активных predict-запросов.

### 10. Ограничение RPM

```bash
--predict-max-rpm 200
```

Этот параметр ограничивает темп отправки запросов к `predict`.

Код считает минимальный интервал между отправками:

```python
min_interval_sec = 60.0 / cfg.predict_max_rpm
```

При `200 rpm` это примерно `0.3` секунды между отправками запросов.

## Как формируется таблица для метрики

Для каждого возвращенного пользователя evaluator создает строку:

```python
{
    "shift_id": "...",
    "start_at": "...",
    "capacity": ...,
    "target": 0 или 1,
    "score": ...,
}
```

`target = 1`, если пара `(user_id, shift_id)` есть в `apply.csv`.

`target = 0`, если такой пары нет.

`score` здесь не реальная вероятность модели, а искусственный скор по позиции в ранжировании:

- первый кандидат получает максимальный score;
- следующие кандидаты получают меньший score.

## Как считается метрика

Метрика реализована в `hackaton/eval/metric.py`.

Основные правила:

- фиксированный пул кандидатов: `TOP-10`;
- внутри смены берется `top capacity`;
- используется `roc_auc_score`;
- FPR ограничивается формулой:

```python
max_fpr = min(1.0, capacity / 10)
```

Дальше метрика агрегируется:

1. Сначала считается метрика по отдельным сменам.
2. Потом усредняется внутри групп `eval_date + capacity`.
3. Потом усредняется по дням.
4. Итоговая метрика - среднее по дневным метрикам.

## Что происходит после оценки дня

После того как predictions для дня получены и метрика дня рассчитана, evaluator догружает в сервис validation-данные этого дня:

- смены дня через RPC `shift`;
- события дня через RPC `event`.

Затем вызывается инкрементальный `prepare`, чтобы сервис обновил состояние перед следующим днем.

Таймаут дневного prepare по умолчанию равен `300` секунд, то есть 5 минут.

## Какие артефакты создаются

Команда создает Markdown-отчет:

```text
artifacts/eval_run/eval_report.md
```

В отчете будут:

- количество train users/shifts/events;
- `overall_target_metric`;
- `predict_latency_p50_ms`;
- `predict_latency_p80_ms`;
- `predict_latency_p95_ms`;
- `predict_rpm`;
- средняя длительность `prepare`;
- метрики по каждому validation-дню;
- метрики по группам `capacity`;
- причина остановки, обычно `completed`.

## Что выводится в терминал

После успешного завершения CLI печатает краткое резюме:

```text
Evaluation finished successfully.
Report: artifacts/eval_run/eval_report.md
Overall target metric: ...
Days evaluated: ...
predict_rpm: ...
```

## Краткий вывод

Эта команда выполняет локальную имитацию регламентной проверки решения:

1. Загружает исторические train-данные в сервис.
2. Запускает подготовку модели или кэшей через `prepare`.
3. День за днем отправляет validation-смены в `predict`.
4. Сравнивает рекомендованных пользователей с фактическими откликами из `apply.csv`.
5. Считает итоговую метрику качества.
6. Замеряет latency и RPM для `predict`.
7. Сохраняет отчет в `artifacts/eval_run/eval_report.md`.

Иными словами, команда проверяет, насколько хорошо текущий сервис ранжирует пользователей для смен на validation-наборе и укладывается ли он в ограничения по готовности, задержкам и темпу `predict`.
