# Отчет о выполненной работе

## 1. Общая цель работы

В рамках доработки проекта был добавлен локальный сценарий подготовки validation-выборки из
имеющихся train-данных. Это нужно для ситуации, когда участнику доступны только исходные файлы:

- `data/train/user.csv`
- `data/train/shift.csv`
- `data/train/event.csv`

После доработки можно автоматически разделить эти данные на локальную train-часть и validation-часть,
а затем запускать существующий eval-пайплайн без ручной подготовки `apply.csv`.

Дополнительно была вынесена отдельная документация по нашим изменениям, чтобы не смешивать базовый
гайд участника с локальными доработками поверх проекта.

## 2. Добавленный сценарий

Основной новый сценарий:

1. Взять исходные train CSV из `data/train`.
2. Выбрать временную границу validation-периода.
3. Оставить более ранние смены и события в train-части.
4. Перенести последние смены и релевантные события в validation-часть.
5. Сформировать `apply.csv` в формате, который ожидает eval.
6. Запустить eval уже на раздельных директориях:
   - train: `data/train_split`
   - validation: `data/validation`

## 3. Новый скрипт `scripts/split_train_validation.py`

Добавлен файл:

```text
scripts/split_train_validation.py
```

Скрипт реализует разбиение train-данных на две части:

- `data/train_split` — новая train-часть;
- `data/validation` — validation-часть для локального eval.

### 3.1. Входные файлы

По умолчанию скрипт читает:

```text
data/train/user.csv
data/train/shift.csv
data/train/event.csv
```

Путь можно переопределить аргументом:

```bash
--input-dir
```

### 3.2. Выходные файлы

По умолчанию скрипт создает:

```text
data/train_split/user.csv
data/train_split/shift.csv
data/train_split/event.csv
data/validation/apply.csv
data/validation/shift.csv
data/validation/event.csv
```

Пути можно переопределить аргументами:

```bash
--output-train-dir
--output-validation-dir
```

### 3.3. Выбор validation-периода

Скрипт поддерживает два режима выбора границы:

1. Автоматический режим через `--validation-days`.
2. Ручной режим через `--cutoff-date`.

По умолчанию используется автоматический режим:

```bash
--validation-days 14
```

В этом режиме скрипт берет последние `N` уникальных дат смен из `shift.csv` и использует их как
validation-период.

Если передать:

```bash
--cutoff-date YYYY-MM-DD
```

то эта дата становится первой датой validation-периода. Все смены с датой `start_at >= cutoff_date`
попадают в validation.

### 3.4. Правила разбиения смен

Смены из `shift.csv` делятся по дате `start_at`:

- если дата смены меньше `cutoff_date`, смена попадает в `data/train_split/shift.csv`;
- если дата смены больше или равна `cutoff_date`, смена попадает в `data/validation/shift.csv`.

Исходный файл `data/train/shift.csv` при этом не изменяется.

### 3.5. Правила разбиения событий

События из `event.csv` делятся по дате `ts` и связи со сменами validation-периода:

- событие попадает в train-часть, если дата события меньше `cutoff_date` и оно не относится к
  validation-смене;
- событие попадает в validation-часть, если дата события больше или равна `cutoff_date`.

Это защищает train-часть от событий, которые относятся к validation-сменам.

### 3.6. Формирование `apply.csv`

Для eval нужен файл:

```text
data/validation/apply.csv
```

Он формируется автоматически из взаимодействий `APPLY` пользователей с validation-сменами.

Позитивным взаимодействием для локального eval считается только:

```text
APPLY
```

`FINISHED` не добавляется в `apply.csv` как label, потому что по официальной методике positive target
для оценки — только запись на смену через `APPLY`.

Если для пары `(user_id, shift_id, date)` есть цепочка с `SYSTEM_CANCEL`, такая пара исключается из
`apply.csv`, даже если в этой же цепочке был `APPLY`. При этом сами события `VIEW`, `USER_CANCEL`,
`SYSTEM_CANCEL` и `FINISHED` остаются в `event.csv` и могут использоваться сервисом как исторические
признаки.

Для каждого подходящего `APPLY` скрипт добавляет строку:

```text
user_id,shift_id,date
```

Дата берется не из события, а из даты самой validation-смены. Это соответствует контракту eval,
где labels группируются по дню смены.

Дубликаты удаляются через множество ключей:

```text
(user_id, shift_id, date)
```

После этого строки сортируются по:

1. `date`
2. `shift_id`
3. `user_id`

### 3.7. Защита от перезаписи

Скрипт не перезаписывает существующие выходные файлы без явного разрешения.

Если файлы уже существуют, запуск без `--force` завершится ошибкой с подсказкой использовать:

```bash
--force
```

Для обычного повторного локального запуска используется команда:

```bash
python3 scripts/split_train_validation.py --force
```

### 3.8. Вывод summary

После выполнения скрипт печатает краткую сводку:

```text
cutoff_date
train_users
train_shifts
train_events
validation_shifts
validation_events
validation_apply
```

Эта сводка помогает быстро проверить, что разбиение не получилось пустым и что validation labels
были сформированы.

## 4. Добавленный unit-тест

Подготовлен файл:

```text
tests/unit/test_split_train_validation.py
```

Тест проверяет ключевой контракт нового скрипта:

- создаются файлы, нужные eval;
- последняя дата смен попадает в validation;
- более ранняя дата остается в train;
- `apply.csv` формируется из позитивного interaction;
- событие, которое относится к validation-смене, не протекает в train-часть;
- дата в `apply.csv` соответствует дню смены.

Тест использует временную директорию `tmp_path`, поэтому не зависит от реальных файлов в `data`.

## 5. Документация по нашим изменениям

Создан файл:

```text
OUR-CHANGES.md
```

В него вынесена инструкция, которая раньше была добавлена в `HOW-TO.md`.

Документ описывает:

- зачем нужен локальный split;
- как запустить `scripts/split_train_validation.py`;
- какие директории будут созданы;
- как после split запускать eval;
- какие файлы относятся к этому изменению.

## 6. Изменения в `HOW-TO.md`

Изначально инструкция про локальное validation-разбиение была добавлена прямо в:

```text
HOW-TO.md
```

Затем по решению вынести наши изменения отдельно эта вставка была перенесена в:

```text
OUR-CHANGES.md
```

После переноса `HOW-TO.md` был возвращен к исходному содержанию без смысловых изменений.

## 7. Команды для работы с новым сценарием

### 7.1. Создать локальное разбиение

```bash
python3 scripts/split_train_validation.py --force
```

### 7.2. Запустить eval на локальном разбиении

```bash
poetry run python -m hackaton.eval.cli run \
  --host 127.0.0.1 \
  --port 8000 \
  --user-path data/train_split/user.csv \
  --shift-path data/train_split/shift.csv \
  --event-path data/train_split/event.csv \
  --val-apply-path data/validation/apply.csv \
  --val-shift-path data/validation/shift.csv \
  --val-event-path data/validation/event.csv \
  --output-dir artifacts/eval_run \
  --predict-max-concurrency 4 \
  --predict-max-rpm 200
```

### 7.3. Проверить отчет eval

```text
artifacts/eval_run/eval_report.md
```

## 8. Текущий git-статус на момент отчета

На момент подготовки отчета в git index добавлен только файл:

```text
scripts/split_train_validation.py
```

Остальные связанные файлы пока не добавлены в index:

```text
OUR-CHANGES.md
tests/unit/test_split_train_validation.py
WORK_REPORT.md
```

Также в рабочей директории есть незатреканная IDE-директория:

```text
.idea/
```

Ее не нужно добавлять в commit, если нет отдельной причины хранить настройки IDE в репозитории.

## 9. Что рекомендуется добавить в commit

Для полноценного commit по этой задаче логично добавить:

```bash
git add scripts/split_train_validation.py
git add tests/unit/test_split_train_validation.py
git add OUR-CHANGES.md
git add WORK_REPORT.md
```

`HOW-TO.md` добавлять не требуется, потому что после переноса в нем нет смыслового diff.

`.idea/` добавлять не рекомендуется.

## 10. Проверки

Для проверки нового скрипта рекомендуется запустить точечный unit-тест:

```bash
poetry run pytest tests/unit/test_split_train_validation.py
```

Для общей проверки проекта:

```bash
make test
make precommit
```

На момент создания этого отчета тесты в рамках последнего шага не запускались, потому что текущий
запрос касался подготовки документации.

## 11. Проведенные исследования baseline inference

Дополнительно был разобран вопрос, почему текущий сервис получает ненулевую метрику около `0.38`,
хотя обработчик `prepare` фактически ничего не готовит.

По коду сервиса выяснено, что текущий `prepare` в `hackaton/service/prepare_manager.py` выполняет
только роль state gate:

- переводит сервис во временное состояние not ready;
- ожидает заданное время;
- возвращает сервис в ready-состояние.

Он не строит признаки, не загружает модель и не материализует агрегаты. Это подтверждается отчетами
eval, где `prepare_duration_avg_sec` равен `0.0`.

При этом `predict` в `hackaton/service/app.py` все равно возвращает кандидатов, потому что использует
rule-based SQL-запрос из `hackaton/service/repositories.py`:

```sql
SELECT id
FROM users
WHERE location_id = ?
  AND (? = 0 OR has_mk = 1)
ORDER BY is_strict_location DESC, has_mk DESC, id ASC
LIMIT ?
```

То есть текущий online inference ранжирует пользователей не моделью, а простой эвристикой:

1. взять пользователей из той же `location_id`;
2. если смене нужен МК, оставить только пользователей с `has_mk = 1`;
3. отсортировать по `is_strict_location`, `has_mk`, затем по `id`;
4. вернуть TOP-10.

Именно эта эвристика дает baseline-качество. В отчетах:

```text
artifacts/eval_baseline/eval_report.md
artifacts/eval_strict_split/eval_report.md
```

зафиксировано:

```text
overall_target_metric: 0.3838612368024133
prepare_duration_avg_sec: 0.0
```

Вывод: метрика около `0.38` получается не за счет модели и не за счет полезной работы `prepare`, а за
счет сильных простых признаков `location_id` и `has_mk`. Для данных смен это уже дает заметный сигнал,
потому что пользователи часто откликаются на смены в своей локации, а требование МК отсекает часть
заведомо неподходящих кандидатов.

Практический вывод для дальнейшей работы:

- `prepare` сейчас является заглушкой, но его нужно использовать как основную точку роста;
- тяжелые агрегации по истории событий, пользователям и сменам нужно переносить в `prepare`;
- `predict` должен оставаться быстрым и выполнять только выбор ограниченного candidate pool,
  расчет признаков из готовых агрегатов и ранжирование;
- следующий целевой этап — заменить текущий rule-based inference на model-based scoring, сохранив
  совместимость с RPC-контрактом и ограничением `predict_max_rpm <= 200`.

## 12. Итог

В результате выполненной работы в проекте появился воспроизводимый локальный путь от исходных
train CSV до eval-ready validation-набора:

1. `scripts/split_train_validation.py` готовит `data/train_split` и `data/validation`.
2. `data/validation/apply.csv` формируется автоматически из позитивных interactions.
3. `OUR-CHANGES.md` описывает, как пользоваться новым сценарием.
4. `tests/unit/test_split_train_validation.py` фиксирует ключевой контракт split-логики.
5. `WORK_REPORT.md` фиксирует подробный отчет о выполненной работе.

## 13. Доработка online inference через prepare-time aggregates

Следующим этапом начата реализация пункта 3 рабочего плана: закрытие разрыва между offline training
и online inference.

### 13.1. Изменения в схеме БД

В `hackaton/service/db.py` добавлены materialized feature-таблицы:

```text
user_features
user_task_features
user_employer_features
user_workplace_features
```

Они предназначены для быстрых online-запросов в `predict`, чтобы не выполнять тяжелую агрегацию по
сырым `events` на каждый входящий shift.

### 13.2. Полезная работа в prepare

В `hackaton/service/repositories.py` добавлен метод:

```text
Repository.rebuild_features()
```

Он перестраивает агрегаты по истории событий:

- общая активность пользователя: `VIEW`, `APPLY`, `FINISHED`;
- отмены: `USER_CANCEL`, `SYSTEM_CANCEL`;
- история по `task_type`;
- история по `employer_id`;
- история по `workplace_id`;
- средний `reward/hour` по позитивным взаимодействиям.

В `hackaton/service/prepare_manager.py` добавлена возможность выполнять async callback во время
`prepare`. Теперь `HackatonRpcService.prepare()` передает туда `Repository.rebuild_features`, поэтому
`prepare` больше не является только sleep-заглушкой.

### 13.3. Новый online scoring в predict

В `hackaton/service/repositories.py` добавлен метод:

```text
Repository.find_scored_candidates()
```

`predict` теперь сначала использует scored ranking, а старый `location_id + has_mk` SQL оставлен как
fallback для пустых случаев.

Скоринг учитывает:

- совпадение `location_id`;
- `has_mk` при необходимости МК;
- `is_strict_location`;
- общую историю пользователя;
- историю по `task_type`;
- историю по `employer_id`;
- историю по `workplace_id`;
- штрафы за отмены;
- близость к пользовательскому `reward/hour`.

Это промежуточный rule-based ranking без подключения `model.pkl`. Такой шаг выбран намеренно: он
быстро усиливает online baseline и готовит структуру для последующего model-based scoring.

### 13.4. Добавленные проверки

В `tests/unit/test_service_smoke.py` добавлен сценарий, который проверяет:

- `prepare` создает feature-агрегаты;
- история `FINISHED`/`APPLY` по совпадающим `task_type`, `employer_id`, `workplace_id` влияет на
  порядок выдачи;
- пользователь с релевантной историей может обогнать пользователя из той же локации.

`tests/e2e/test_rpc_api_contract_e2e.py` не изменялся. Существующий контракт 503 во время подготовки
и 200 после готовности сохранен за счет реализации `PrepareManager`, где callback подготовки и
ожидание выполняются в одном background prepare-цикле.

### 13.5. Проверки после доработки

Запущена точечная проверка:

```bash
poetry run pytest tests/unit/test_service_smoke.py tests/e2e/test_rpc_api_contract_e2e.py
```

Результат:

```text
5 passed
coverage: 92.31%
```

В ограниченном sandbox `aiosqlite` зависал на подключении к SQLite, поэтому тест был запущен с
разрешенным выполнением вне sandbox. Это связано с окружением запуска, а не с логикой сервиса.

## 14. Eval после доработки prepare-time aggregates

После внедрения prepare-time агрегатов и scored ranking был запущен локальный eval на текущем
train/validation split.

### 14.1. Команда запуска сервиса

Порт `8000` был занят другим процессом, поэтому текущая версия сервиса была поднята на `8001` с
отдельной eval-БД:

```bash
APP_HOST=127.0.0.1 \
APP_PORT=8001 \
DB_PATH=./data/hackaton_eval_current_8001.db \
PREPARE_SLEEP_SECONDS=0 \
poetry run python -m hackaton.service.main
```

### 14.2. Команда eval

Eval запускался на локальном split-наборе:

```bash
poetry run python -m hackaton.eval.cli run \
  --host 127.0.0.1 \
  --port 8001 \
  --user-path data/train_split/user.csv \
  --shift-path data/train_split/shift.csv \
  --event-path data/train_split/event.csv \
  --val-apply-path data/validation/apply.csv \
  --val-shift-path data/validation/shift.csv \
  --val-event-path data/validation/event.csv \
  --output-dir artifacts/eval_scored_prepare \
  --predict-max-concurrency 4 \
  --predict-max-rpm 200
```

Первый запуск eval внутри sandbox завершился ошибкой ZeroMQ connection timeout на bootstrap upload.
Повторный запуск вне sandbox прошел успешно. Это соответствует ранее обнаруженному ограничению
локального sandbox для `aiosqlite`/ZeroMQ.

### 14.3. Результат

Итоговый отчет:

```text
artifacts/eval_scored_prepare/eval_report.md
```

Ключевые значения:

```text
overall_target_metric: 0.9171458647561589
predict_latency_p50_ms: 24.824
predict_latency_p80_ms: 28.467
predict_latency_p95_ms: 32.920
predict_rpm: 209.173
prepare_duration_avg_sec: 0.0
stop_reason: completed
days_evaluated: 13
```

Сравнение с предыдущим baseline:

```text
baseline overall_target_metric: 0.3838612368024133
current overall_target_metric: 0.9171458647561589
absolute gain: +0.5332846279537456
```

Вывод: перенос истории событий в prepare-time агрегаты и использование scored ranking дали
существенный прирост локальной validation-метрики относительно `location_id + has_mk` baseline.

### 14.4. Наблюдения и риски

- `predict_latency_p95_ms` вырос до `32.920 ms`, но остается низким для текущего локального eval.
- `predict_rpm` в отчете равен `209.173`, хотя eval запускался с `--predict-max-rpm 200`.
  Аналогичное превышение уже наблюдалось в baseline-отчете, но перед финальной сдачей это нужно
  перепроверить и при необходимости запускать eval/load-test с более консервативными параметрами.
- `prepare_duration_avg_sec` в отчете равен `0.0`, потому что evaluator измеряет prepare/ready
  контракт, а не отдельное внутреннее время SQL rebuild. По логам дневные prepare-переходы занимали
  около нескольких секунд wall-clock между post-day upload и следующим днем.

## 15. Приведение local validation к официальной методике оценки

После сверки с методикой оценки выяснилось, что локальный split формировал `apply.csv` слишком широко:
в positive labels попадали не только `APPLY`, но и `FINISHED`.

Официальная методика задает другой контракт:

- `APPLY` считается positive label;
- отсутствие `APPLY` считается negative label;
- `VIEW` и `USER_CANCEL` без последующего `APPLY` считаются negative label;
- цепочки с `SYSTEM_CANCEL` исключаются из оценки, даже если в цепочке был `APPLY`;
- `VIEW`, `USER_CANCEL`, `SYSTEM_CANCEL` можно использовать как исторические признаки.

### 15.1. Изменения в split-скрипте

В `scripts/split_train_validation.py` изменено формирование `data/validation/apply.csv`:

- positive label теперь создается только по `APPLY`;
- `FINISHED` больше не добавляется в `apply.csv`;
- пары `(user_id, shift_id, date)` с `SYSTEM_CANCEL` исключаются из `apply.csv`;
- validation `event.csv` сохраняет все события как исторические данные.

### 15.2. Тесты методики

В `tests/unit/test_split_train_validation.py` добавлен сценарий:

- `APPLY` попадает в `apply.csv`;
- `FINISHED` без `APPLY` не попадает в `apply.csv`;
- `APPLY` вместе с `SYSTEM_CANCEL` исключается из `apply.csv`.

Проверки:

```bash
poetry run pytest tests/unit/test_split_train_validation.py --no-cov
poetry run pytest tests/unit/test_service_smoke.py tests/e2e/test_rpc_api_contract_e2e.py
poetry run ruff check scripts/split_train_validation.py tests/unit/test_split_train_validation.py \
  hackaton/service/app.py hackaton/service/db.py hackaton/service/prepare_manager.py \
  hackaton/service/repositories.py tests/unit/test_service_smoke.py
```

Результат:

```text
split tests: 2 passed
service/e2e tests: 5 passed, coverage 92.31%
ruff: All checks passed
```

### 15.3. Пересборка validation

После изменения методики локальный split был пересобран:

```bash
poetry run python scripts/split_train_validation.py --force
```

Результат:

```text
cutoff_date: 2026-02-16
train_users: 5154
train_shifts: 39734
train_events: 381129
validation_shifts: 265
validation_events: 767
validation_apply: 215
```

### 15.4. Повторный eval

Повторный eval запускался на отдельном порту и отдельной БД:

```bash
APP_HOST=127.0.0.1 \
APP_PORT=8002 \
DB_PATH=./data/hackaton_eval_official_8002.db \
PREPARE_SLEEP_SECONDS=0 \
poetry run python -m hackaton.service.main
```

```bash
poetry run python -m hackaton.eval.cli run \
  --host 127.0.0.1 \
  --port 8002 \
  --user-path data/train_split/user.csv \
  --shift-path data/train_split/shift.csv \
  --event-path data/train_split/event.csv \
  --val-apply-path data/validation/apply.csv \
  --val-shift-path data/validation/shift.csv \
  --val-event-path data/validation/event.csv \
  --output-dir artifacts/eval_official_methodology \
  --predict-max-concurrency 4 \
  --predict-max-rpm 200
```

Итоговый отчет:

```text
artifacts/eval_official_methodology/eval_report.md
```

Ключевые значения:

```text
overall_target_metric: 0.9231554801407742
predict_latency_p50_ms: 28.086
predict_latency_p80_ms: 32.008
predict_latency_p95_ms: 36.151
predict_rpm: 209.129
prepare_duration_avg_sec: 0.0
stop_reason: completed
days_evaluated: 13
```

Вывод: после приведения `apply.csv` к официальной методике локальная метрика не просела. Это
подтверждает, что прирост дает не ошибочное включение `FINISHED` в labels, а history-based ranking по
событиям и сущностям смены.

### 15.5. Контрольный eval с запасом по RPM

Так как в предыдущих eval-отчетах итоговый `predict_rpm` получался немного выше `200`, был проведен
контрольный запуск с более консервативным лимитом:

```bash
APP_HOST=127.0.0.1 \
APP_PORT=8003 \
DB_PATH=./data/hackaton_eval_rpm180_8003.db \
PREPARE_SLEEP_SECONDS=0 \
poetry run python -m hackaton.service.main
```

```bash
poetry run python -m hackaton.eval.cli run \
  --host 127.0.0.1 \
  --port 8003 \
  --user-path data/train_split/user.csv \
  --shift-path data/train_split/shift.csv \
  --event-path data/train_split/event.csv \
  --val-apply-path data/validation/apply.csv \
  --val-shift-path data/validation/shift.csv \
  --val-event-path data/validation/event.csv \
  --output-dir artifacts/eval_official_rpm180 \
  --predict-max-concurrency 4 \
  --predict-max-rpm 180
```

Итоговый отчет:

```text
artifacts/eval_official_rpm180/eval_report.md
```

Ключевые значения:

```text
overall_target_metric: 0.9231554801407742
predict_latency_p50_ms: 33.580
predict_latency_p80_ms: 36.710
predict_latency_p95_ms: 39.646
predict_rpm: 188.181
prepare_duration_avg_sec: 0.0
stop_reason: completed
days_evaluated: 13
```

Вывод: при `--predict-max-rpm 180` итоговый `predict_rpm` остается ниже регламентного порога `200`, а
целевая метрика не меняется. Для финальной проверки предпочтительно использовать этот более
консервативный режим запуска eval/load-test.
