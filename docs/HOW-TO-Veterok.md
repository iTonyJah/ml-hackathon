# Как воспроизвести результаты (test/improve..)

```bash
# Установка зависимостей
make install

# Создание валидационного сплита (обязательно перед eval)
poetry run python create_validation_split.py

# Обучить свою модель (train /data/new_train -> artifacts/train)
poetry run python -m hackaton.train.cli train \
  --user-path data/new_train/user.csv \
  --shift-path data/new_train/shift_train.csv \
  --event-path data/new_train/event_train.csv \
  --output-dir artifacts/train \
  --skip-shap
```

(Что не ясно, как будет учиться модель. Похоже, что не на всех данных, а делит их на 80/20.)
(И вроде выпадает с ошибкой, если не учить, то выпадает с ошибкой. Пока учим всегда.)

# далее

```bash
# Создание базы данных
make migrate

# Запуск сервиса (в отдельном терминале)
make run

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
```

(без трейна не проходит, что не очень, но допустимо)

## Time-based CV (запускается отдельно, без сервиса)
```bash
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
