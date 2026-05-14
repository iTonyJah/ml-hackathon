# HOW-TO для участников

## Цель документа

Этот файл помогает быстро понять:

- где в проекте вносить изменения;
- как локально проверить, что решение корректное;
- как интерпретировать итог оценки.

## Минимальный маршрут участника

### Поднять сервис

```bash
make install
make migrate
make run
```

### Проверить качество кода

```bash
make test
make precommit
```

### Проверить, что CI пройдет

- Убедитесь, что локально зеленые `make test` и `make precommit`.
- Проверьте, что ваш код не ломает runtime smoke-цепочку:
  - сервис стартует;
  - данные `user/shift/event` загружаются;
  - `prepare` завершается, `ready` возвращает готовность;
  - `predict` возвращает непустой список кандидатов.
- Для проверки нагрузочного контура перед пушем можно запустить:

```bash
make load-test
```

### Обучить baseline/свою модель (вся data)

```bash
poetry run python -m hackaton.train.cli train \
  --user-path data/train/user.csv \
  --shift-path data/train/shift.csv \
  --event-path data/train/event.csv \
  --output-dir artifacts/train \
  --skip-shap
```

### Обучить свою модель (train data /new_train)

```bash
poetry run python -m hackaton.train.cli train \
  --user-path data/new_train/user.csv \
  --shift-path data/new_train/shift_train.csv \
  --event-path data/new_train/event_train.csv \
  --output-dir artifacts/new_train \
  --skip-shap
```


## Часть 14. Как воспроизвести результаты

```bash
# Установка зависимостей
make install

# Создание валидационного сплита (обязательно перед eval)
poetry run python create_validation_split.py

# Создание базы данных
make migrate

# Запуск сервиса (в отдельном терминале)
make run

# Официальный eval (нужен запущенный сервис)
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


# Официальный eval (new_train)
poetry run python -m hackaton.eval.cli run \
  --user-path data/new_train/user.csv \
  --shift-path data/new_train/shift_train.csv \
  --event-path data/new_train/event_train.csv \
  --val-apply-path data/new_validation/apply.csv \
  --val-shift-path data/new_validation/shift.csv \
  --val-event-path data/new_validation/event.csv \
  --output-dir artifacts/eval_run \
  --limit 10 \
  --batch-size 1000






# Time-based CV (запускается отдельно, без сервиса)
poetry run python -m hackaton.train.cli cv \
  --user-path data/train/user.csv \
  --shift-path data/train/shift_train.csv \
  --event-path data/train/event_train.csv \
  --output-dir artifacts/cv_run \
  --val-days 30
```

Результаты CV сохраняются в `artifacts/cv_run/cv_report.md`.





## вывод предупреждения make test:
```
itj@fedora:~/ml-hackathon$ make test
poetry run pytest
=================================================================== test session starts ====================================================================
platform linux -- Python 3.14.4, pytest-8.4.2, pluggy-1.6.0
rootdir: /home/itj/ml-hackathon
configfile: pyproject.toml
testpaths: tests
plugins: asyncio-1.3.0, cov-7.1.0
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 12 items

tests/e2e/test_rpc_api_contract_e2e.py ..                                                                                                            [ 16%]
tests/unit/test_eval_config_and_cli.py ...                                                                                                           [ 41%]
tests/unit/test_eval_metric.py ..                                                                                                                    [ 58%]
tests/unit/test_prepare_ml.py ..                                                                                                                     [ 75%]
tests/unit/test_service_smoke.py ..                                                                                                                  [ 91%]
tests/unit/test_train_smoke.py .                                                                                                                     [100%]

===================================================================== warnings summary =====================================================================
tests/unit/test_prepare_ml.py::test_predict_after_ml_prepare
  /home/itj/ml-hackathon/.venv/lib64/python3.14/site-packages/sklearn/utils/validation.py:2691: UserWarning: X does not have valid feature names, but LGBMClassifier was fitted with feature names
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
====================================================================== tests coverage ======================================================================
_____________________________________________________ coverage: platform linux, python 3.14.4-final-0 ______________________________________________________

Name                                  Stmts   Miss  Cover   Missing
-------------------------------------------------------------------
hackaton/eval/cli.py                     49      7    86%   79-81, 85-87, 122
hackaton/eval/metric.py                  50      3    94%   23, 46, 86
hackaton/service/app.py                  94      7    93%   78, 95-96, 116, 118, 158-159
hackaton/service/prepare_manager.py     111     20    82%   47, 81, 132, 147-149, 170-186, 191-193
-------------------------------------------------------------------
TOTAL                                   304     37    88%
Required test coverage of 80% reached. Total coverage: 87.83%
============================================================== 12 passed, 1 warning in 2.60s ===============================================================
```












### Запустить eval

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

### Проверить отчет

- `artifacts/eval_run/eval_report.md`

## Как добавлять новую библиотеку

Используйте только `poetry`, чтобы зависимости и lock-файл оставались согласованными.

Для runtime-зависимости:

```bash
poetry add <package-name>
```

Для dev-зависимости:

```bash
poetry add --group dev <package-name>
```

После добавления:

```bash
make test
make precommit
```

Что важно:

- не редактируйте `poetry.lock` вручную;
- если меняется состав/версии библиотек, коммитите вместе и `pyproject.toml`, и `poetry.lock`.

## Куда вносить изменения

- `hackaton/service/app.py`
  - online-логика `predict`;
  - правила фильтрации/ранжирования кандидатов;
- `hackaton/train/training.py`
  - feature engineering;
  - выбор/настройка модели;
  - train-артефакты.

## Что лучше не трогать

- Контракты RPC-методов (`user`, `event`, `shift`, `prepare`, `ready`, `predict`).
- Контракты входных CSV.
- Ограничение `predict_max_rpm <= 200`.
- Фиксированный пул: `pool_size = 10`.
- Ограничение FPR: `max_fpr = min(1.0, capacity / 10)`.
- Агрегация:
  - shift -> capacity-group/day;
  - day -> overall metric.

Подробнее про регламент можно прочитать в: `REGLAMENT.md`.

## Как читать eval-отчет

В `eval_report.md` смотрите:

- `overall_target_metric` — итоговый score решения.
- `predict_latency_p50/p80/p95` — задержки запросов `predict`.
- `predict_rpm` — фактический темп запросов.
- `prepare_duration_*` — цена подготовки модели.
- `Daily metrics` — детализация качества по дням и группам `capacity`.

## Частые проблемы и что делать

- `predict` возвращает `503 model is in prepare state`
  - дождитесь `ready` ;
  - проверьте таймауты `prepare_*_timeout_sec`.
- Eval падает на лимите RPM
  - уменьшите `--predict-max-concurrency`;
  - держите `--predict-max-rpm` не выше 200.
- Пустая/нулевая метрика
  - проверьте, что в `apply.csv` есть валидные совпадения с `predict`;
  - проверьте корректность `user_id/shift_id/date`.
- Изменили версии в `pyproject.toml` вручную, и `poetry.lock` устарел
  - перегенерируйте lock: `poetry lock`;
  - установите зависимости из lock: `poetry install`;
  - если нужно обновить конкретный пакет до новой версии: `poetry add <package-name>@<version>`;
  - если lock поврежден или конфликтный, можно пересобрать полностью:
    - `rm poetry.lock`
    - `poetry lock`
    - `poetry install`

## Чеклист перед MR

- Используйте `CHECKLIST.md` как обязательный pre-commit/pre-MR список.
- Минимум перед отправкой:
  - `make test` зеленый;
  - `make precommit` зеленый;
  - CI-цепочка из `.github/workflows/ci.yml` не должна падать на smoke-check;
  - `eval_report.md` формируется и читается.
