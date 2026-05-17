# HOW-TO: как работать с финальным решением

Этот документ помогает быстро запустить проект, проверить качество и понять, где находится основная логика решения.

Если нужен максимально строгий сценарий воспроизводимости, используйте `docs/REPRODUCIBILITY.md`.

## Минимальный маршрут

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

В отдельном терминале проверить тесты:

```bash
make test
```

## Подготовить validation split

Если у вас уже есть готовые файлы в `data/validation`, этот шаг можно пропустить.

Если нужно сделать локальный split из `data/train/user.csv`, `data/train/shift.csv`, `data/train/event.csv`:

```bash
poetry run python scripts/create_validation_split.py
```

Для быстрого smoke-прогона можно взять последние два дня:

```bash
poetry run python scripts/create_validation_split_2d.py
```

После этого появятся:

```text
data/train/shift_train.csv
data/train/event_train.csv
data/validation/apply.csv
data/validation/shift.csv
data/validation/event.csv
```

## Запустить eval

Eval сам не поднимает сервис. Перед командой ниже должен работать `make run`.

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

Отчет:

```text
artifacts/eval_run/eval_report.md
```

## Запустить time-based CV

CV не требует запущенного сервиса.

```bash
poetry run python -m hackaton.train.cli cv \
  --user-path data/train/user.csv \
  --shift-path data/train/shift_train.csv \
  --event-path data/train/event_train.csv \
  --output-dir artifacts/cv_run \
  --val-days 30 \
  --candidate-limit 300
```

Если split не создавался, можно передать исходные train-файлы:

```bash
poetry run python -m hackaton.train.cli cv \
  --user-path data/train/user.csv \
  --shift-path data/train/shift.csv \
  --event-path data/train/event.csv \
  --output-dir artifacts/cv_run \
  --val-days 30 \
  --candidate-limit 300
```

Отчет:

```text
artifacts/cv_run/cv_report.md
```

## Где менять решение

Основные файлы финального ML-решения:

- `hackaton/service/ml_model.py` - признаки, обучение, LightGBM, скоринг, sleeper-rerank;
- `hackaton/service/prepare_manager.py` - загрузка данных, обучение модели, построение кэшей;
- `hackaton/service/app.py` - RPC-методы и online-логика `predict`;
- `hackaton/service/repositories.py` - запись и чтение SQLite;
- `hackaton/train/cv.py` - time-based CV.

Файлы eval:

- `hackaton/eval/evaluator.py` - дневная симуляция проверки;
- `hackaton/eval/metric.py` - целевая метрика.

## Как работает финальный predict

1. Проверяет, готов ли сервис.
2. Берет до 300 кандидатов из кэша по локации.
3. Если кэш пуст, использует fallback через БД.
4. Если модель обучена, считает score для кандидатов.
5. Применяет rerank:
   - ранее подавал заявку на эту смену;
   - sleeper-пользователь;
   - остальные по ML-score.
6. Возвращает top `limit` пользователей.

## Как читать eval-отчет

В `eval_report.md` важны:

- `overall_target_metric` - итоговая метрика;
- `predict_latency_p50_ms`, `predict_latency_p80_ms`, `predict_latency_p95_ms` - задержка `predict`;
- `predict_rpm` - фактическая частота запросов;
- `prepare_duration_*` - сколько занимает подготовка;
- дневные метрики;
- метрики по группам `capacity`.

Если итоговая метрика кажется странной, сначала проверьте число оцениваемых смен и групп. Дни с `capacity=1` могут плохо отражаться в ROC-AUC.

## Частые проблемы

### Сервис не отвечает

Проверьте, что запущен:

```bash
make run
```

### Порт 8000 занят

Остановите старый процесс. Если в ветке есть helper:

```bash
./kill_8000.sh
```

### `predict` возвращает 503

Сервис еще в `prepare`. Подождите `ready` и проверьте логи.

### Eval дает нули

Проверьте:

- правильный ли `apply.csv`;
- совпадает ли `apply.date` с `shift.start_at.date()`;
- правильные ли пути к `shift_train.csv` и `event_train.csv`;
- есть ли validation-смены с `capacity >= 2`;
- не передан ли train вместо validation или наоборот.

## Перед сдачей

Минимальный чек:

```bash
make test
make precommit
```

Желательно также:

```bash
make load-test
```

И отдельно убедиться, что:

- `artifacts/eval_run/eval_report.md` создается;
- `artifacts/cv_run/cv_report.md` создается при запуске CV;
- документация по воспроизводимости соответствует командам, которые реально запускались.
