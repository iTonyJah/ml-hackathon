# Регламент валидации

Этот документ описывает, как локально проверяется качество решения.

## Входные данные

Train-данные:

```text
data/train/user.csv
data/train/shift.csv
data/train/event.csv
```

Validation-данные:

```text
data/validation/apply.csv
data/validation/shift.csv
data/validation/event.csv
```

`apply.csv` содержит ground truth:

- `user_id`;
- `shift_id`;
- `date`.

Важно: `date` должна быть датой начала смены, то есть `shift.start_at.date()`.

## Запуск eval

Eval требует уже запущенный сервис.

Сначала:

```bash
make migrate
make run
```

Затем в другом терминале:

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

Если validation split создан локально, вместо `shift.csv` и `event.csv` обычно используются:

```text
data/train/shift_train.csv
data/train/event_train.csv
```

Пример команды есть в `docs/REPRODUCIBILITY.md`.

Результат:

```text
artifacts/eval_run/eval_report.md
```

## Дневной цикл eval

Eval имитирует работу сервиса по дням:

1. Загружает train-данные в сервис через RPC.
2. Вызывает `prepare`.
3. Ждет `ready`.
4. Для каждого validation-дня берет смены этого дня.
5. Для каждой смены вызывает `predict`.
6. Сравнивает возвращенных пользователей с `apply.csv`.
7. Считает дневную метрику.
8. После дня догружает validation-события и смены этого дня.
9. Снова вызывает `prepare`, чтобы сервис обновил состояние.

Таймауты:

- первичный `prepare` - до 20 минут;
- дневной `prepare` - до 5 минут.

## Правила метрики

Для каждой смены:

1. Берется список кандидатов из `predict`.
2. Для метрики используется фиксированный пул TOP-10.
3. Пользователь получает `target=1`, если пара `user_id + shift_id` есть в `apply.csv` для этого дня.
4. Иначе пользователь получает `target=0`.
5. Score берется из позиции в ранжировании: чем выше пользователь, тем выше score.

Ограничение FPR:

```text
max_fpr = min(1.0, capacity / 10)
```

Агрегация:

1. Смена.
2. Группа `eval_date + capacity`.
3. День.
4. Среднее по дням.

## Важная особенность `capacity=1`

ROC-AUC требует, чтобы в оцениваемой выборке были разные классы. Для смен с `capacity=1` часто невозможно получить одновременно и положительный, и отрицательный пример в верхней части списка.

Поэтому такие смены могут не попасть в расчет. Если в какой-то день все реальные заявки относятся к сменам с `capacity=1`, дневная метрика может быть нулевой не потому, что модель плохая, а потому что структура данных не дает оцениваемых смен.

При анализе результата всегда смотрите:

- сколько дней оценено;
- сколько групп `capacity` оценено;
- сколько смен реально вошло в метрику.

## Ограничения нагрузки

Для `predict` действует лимит:

```text
predict_max_rpm <= 200
```

Также важно следить за latency:

- `predict_latency_p50_ms`;
- `predict_latency_p80_ms`;
- `predict_latency_p95_ms`.

## Что будет в eval-отчете

`eval_report.md` содержит:

- статистику данных;
- итоговую метрику;
- дневные метрики;
- метрики по группам `capacity`;
- latency `predict`;
- фактический RPM;
- длительность `prepare`;
- причину завершения.

## Где читать подробнее

- `docs/REPRODUCIBILITY.md` - точные команды запуска.
- `docs/METRICS_AND_EXPERIMENTS.md` - объяснение метрики и экспериментов.
- `docs/FINAL_REPORT.md` - общий итоговый отчет.
