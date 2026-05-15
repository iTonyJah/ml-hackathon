# Отчет по коммитам ветки LightGBM-added

Дата анализа: 2026-05-01

Анализируемая ветка:

```text
LightGBM-added
```

Базовая ветка:

```text
main
```

Merge-base:

```text
06f2c5e28a68c5d8053574145d1bd0c8d2bf8fc5
```

## Краткий итог

Ветка `LightGBM-added` содержит 5 коммитов поверх `main`.

Первые 3 коммита совпадают с веткой `val-split` и добавляют локальные train/validation CSV, скрипт генерации validation split и helper для освобождения порта `8000`.

Последние 2 коммита добавляют ML-ранжирование в online-сервис:

- новая модель `hackaton/service/ml_model.py`;
- обучение модели во время `prepare`;
- кэш пользователей и признаков;
- замена SQL-поиска кандидатов в `predict` на in-memory cache;
- LightGBM как runtime-зависимость;
- расширение feature engineering;
- векторизованный inference;
- корректировка тестов под обязательный `prepare` перед `predict`.

Итоговый diff относительно `main`:

```text
18 files changed, 438553 insertions(+), 45 deletions(-)
```

Основной объем изменений - добавленные CSV-данные:

```text
data/train/event.csv  | 383884 строк
data/train/shift.csv  |  40000 строк
data/train/user.csv   |   5155 строк
```

## Список коммитов

```text
71041c5 Выделение валидационной выборки
dbd5ea6 Фиксы
56a2a6e Фиксы
bb7c04f Фиксы
3d3c5f8 Увеличение числа кандидатов для предсказания, переработка кэширования пользователей и модели, добавление новых фич.
```

## Итоговые измененные файлы

```text
A create_validation_split.py
A data/train/event.csv
A data/train/shift.csv
A data/train/user.csv
A data/validation/apply.csv
A data/validation/event.csv
A data/validation/shift.csv
A data/validation/users.csv
M hackaton/service/app.py
M hackaton/service/main.py
A hackaton/service/ml_model.py
M hackaton/service/prepare_manager.py
M hackaton/service/repositories.py
A kill_8000.sh
M poetry.lock
M pyproject.toml
M tests/e2e/test_rpc_api_contract_e2e.py
M tests/unit/test_service_smoke.py
```

Итоговая статистика:

| Файл | Добавлено | Удалено |
|---|---:|---:|
| `create_validation_split.py` | 175 | 0 |
| `data/train/event.csv` | 383884 | 0 |
| `data/train/shift.csv` | 40000 | 0 |
| `data/train/user.csv` | 5155 | 0 |
| `data/validation/apply.csv` | 46 | 0 |
| `data/validation/event.csv` | 3016 | 0 |
| `data/validation/shift.csv` | 286 | 0 |
| `data/validation/users.csv` | 5155 | 0 |
| `hackaton/service/app.py` | 45 | 20 |
| `hackaton/service/main.py` | 1 | 1 |
| `hackaton/service/ml_model.py` | 554 | 0 |
| `hackaton/service/prepare_manager.py` | 136 | 4 |
| `hackaton/service/repositories.py` | 51 | 7 |
| `kill_8000.sh` | 5 | 0 |
| `poetry.lock` | 29 | 3 |
| `pyproject.toml` | 1 | 0 |
| `tests/e2e/test_rpc_api_contract_e2e.py` | 1 | 1 |
| `tests/unit/test_service_smoke.py` | 13 | 9 |

## Покоммитный анализ

## Коммит 71041c5

Сообщение:

```text
Выделение валидационной выборки
```

Дата:

```text
Tue Apr 28 11:56:39 2026 +0300
```

### Что внесено

Коммит добавляет:

- `create_validation_split.py`;
- train CSV;
- validation CSV;
- `kill_8000.sh`.

Основная цель: создать локальный validation split из train-данных по дате `2026-02-15`.

Скрипт:

- читает `data/train/shift.csv`, `data/train/event.csv`, `data/train/user.csv`;
- делит смены на train/validation по `start_at`;
- делит события по времени и validation shift ids;
- строит `apply.csv` из `APPLY`-событий;
- исключает смены с `SYSTEM_CANCEL`;
- сохраняет validation-артефакты.

### Замечания

В начальной версии скрипта были проблемы:

- `apply.csv` сохранялся с колонкой `ts`, хотя evaluator ожидает `date`;
- `apply` не фильтровался явно по `ts >= SPLIT_DATE`;
- был неиспользуемый импорт `os`;
- были мелкие проблемы форматирования вывода.

## Коммит dbd5ea6

Сообщение:

```text
Фиксы
```

Дата:

```text
Tue Apr 28 12:48:17 2026 +0300
```

### Что изменено

```text
create_validation_split.py |  15 ++-
data/validation/apply.csv  | 280 ++++++++++++++++++++++-----------------------
```

### Смысл изменений

Коммит исправляет формат `apply.csv`:

Было:

```text
user_id,shift_id,ts
```

Стало:

```text
user_id,shift_id,date
```

Это приводит файл к контракту evaluator, который читает:

```python
val_apply["date"] = pd.to_datetime(val_apply["date"], errors="coerce").dt.date
```

Также убран неиспользуемый импорт `os`, исправлен вывод и заменены unicode-стрелки на `->`.

## Коммит 56a2a6e

Сообщение:

```text
Фиксы
```

Дата:

```text
Wed Apr 29 13:15:50 2026 +0300
```

### Что изменено

```text
create_validation_split.py | 45 +++++++++++++++++++++++++++++++++++++++++----
```

### Смысл изменений

Коммит дорабатывает генератор split:

- добавляет `SPLIT_DATE_PLAIN`;
- явно фильтрует `APPLY` по `events["ts"] >= SPLIT_DATE`;
- добавляет `validate_split`;
- проверяет, что `apply.date >= 2026-02-15`;
- проверяет пересечение `apply.shift_id` с validation shifts;
- проверяет, что `train_events` не содержит будущих событий.

Это исправляет ключевую проблему генератора: до этого `apply.csv` мог содержать даты до split.

## Коммит bb7c04f

Сообщение:

```text
Фиксы
```

Дата:

```text
Wed Apr 29 15:03:20 2026 +0300
```

### Что изменено

```text
data/validation/apply.csv           |  98 +---------------
hackaton/service/app.py             |  64 ++++++----
hackaton/service/main.py            |   2 +-
hackaton/service/ml_model.py        | 227 ++++++++++++++++++++++++++++++++++++
hackaton/service/prepare_manager.py | 150 +++++++++++++++++++++++-
hackaton/service/repositories.py    |  58 +++++++--
poetry.lock                         |  32 ++++-
pyproject.toml                      |   1 +
```

### Главный смысл коммита

Коммит переводит сервис от простого SQL-based ранжирования кандидатов к ML-подходу:

- добавлена новая модель `MLModel`;
- добавлен LightGBM;
- `prepare` теперь загружает данные из SQLite, строит кэши и обучает модель;
- `predict` больше не ходит напрямую в БД за кандидатами;
- кандидаты берутся из in-memory кэша `PrepareManager`;
- затем ранжируются ML-моделью.

### Новая зависимость

В `pyproject.toml` добавлено:

```toml
lightgbm = "^4.6.0"
```

`poetry.lock` обновлен под эту зависимость.

### Изменения в PrepareManager

До ветки `LightGBM-added` `PrepareManager` был почти заглушкой:

```python
ready = True
prepare = sleep(...)
```

После изменения `PrepareManager`:

- принимает `db_path`;
- хранит `MLModel`;
- загружает `users`, `shifts`, `events` из SQLite;
- строит кэш пользователей;
- строит кэш по локациям;
- строит глобальный список пользователей по активности;
- обучает модель в executor;
- выставляет `ready=True` после завершения подготовки.

Это меняет смысл `prepare`: теперь это реальная подготовка модели и inference-кэшей.

### Изменения в app.predict

До изменения `predict` работал примерно так:

1. Найти кандидатов SQL-запросом `repository.find_top_candidates`.
2. Если кандидатов нет, взять `repository.fallback_candidates`.
3. Вернуть кандидатов как есть.

После изменения:

1. Проверяется `prepare_manager.ready`.
2. Кандидаты берутся из `pm.get_candidates(...)`.
3. Если модель обучена, строится `shift_dict`.
4. Вызывается:

```python
model.predict_scores(candidates, pm._users_cache, shift_dict)
```

5. Возвращаются top `request.limit` пользователей.

### Новая MLModel

В первой версии `MLModel` добавлены:

- построение user statistics из событий;
- positive samples из `APPLY`;
- negative samples из `VIEW` без последующего `APPLY`;
- LightGBM-классификатор;
- fallback на `LogisticRegression`, если LightGBM не импортируется;
- feature engineering по пользователю, смене и связям пользователь-смена.

Признаки включают:

- user apply rate;
- user finish rate;
- user cancel rate;
- total applies;
- total views;
- active days;
- has medical book;
- shift hour;
- day of week;
- shift duration;
- reward;
- capacity;
- need medical book;
- location match;
- employer/workplace history;
- employer fill rate;
- reward relative to user average.

### Изменения в Repository

Методы `find_top_candidates` и `fallback_candidates` теперь сортируют пользователей не по `id`, а по количеству событий:

```sql
ORDER BY n_events DESC
```

Также добавлен `get_users_by_ids`, но в финальной версии `predict` он не используется, потому что данные пользователей берутся из кэша.

### Изменения в data/validation/apply.csv

Файл `apply.csv` резко сокращен: из него удалены старые строки до split-даты.

В финальном состоянии ветки:

```text
Всего записей в apply.csv: 45
Записей с date < 2026-02-15: 0
Минимальная дата: 2026-02-17
Максимальная дата: 2026-02-28
```

Это исправляет проблему, найденную в `val-split`.

## Коммит 3d3c5f8

Сообщение:

```text
Увеличение числа кандидатов для предсказания, переработка кэширования пользователей и модели, добавление новых фич.
```

Дата:

```text
Thu Apr 30 17:00:50 2026 +0300
```

### Что изменено

```text
data/validation/apply.csv              |  80 +++----
hackaton/service/app.py                |   5 +-
hackaton/service/ml_model.py           | 407 +++++++++++++++++++++++++++++----
hackaton/service/prepare_manager.py    |  74 +++---
tests/e2e/test_rpc_api_contract_e2e.py |   2 +-
tests/unit/test_service_smoke.py       |  22 +-
```

### Главный смысл коммита

Коммит усиливает ML-часть и оптимизирует online inference:

- увеличивает пул кандидатов для `predict`;
- перерабатывает candidate cache;
- добавляет векторизованный inference cache;
- добавляет новые признаки;
- меняет стратегию fallback-кандидатов;
- обновляет тесты под новый жизненный цикл `prepare -> ready -> predict`.

### Изменение пула кандидатов

В `app.predict` используется:

```python
candidates = pm.get_candidates(
    location_id=shift.location_id,
    need_mk=shift.need_mk,
    limit=5000,
)
```

Идея: скорить не только `request.limit` кандидатов, а большой пул, затем вернуть top-N после ML-ранжирования.

Это повышает шанс найти правильных пользователей, но увеличивает нагрузку на inference. Ветка компенсирует это векторизацией.

### Новая стратегия get_candidates

Финальная логика `PrepareManager.get_candidates`:

- если в локации меньше 50 пользователей, берутся пользователи локации плюс до 500 пользователей из глобального fallback;
- если в локации 50 или больше пользователей, берутся только пользователи этой локации;
- итоговый список обрезается до `limit`.

Практический смысл:

- для маленьких локаций не возвращать слишком короткий список;
- для больших локаций не добавлять cross-location шум;
- снизить риск, что глобально активные пользователи вытеснят релевантных локальных кандидатов.

### Переработка MLModel

Финальная версия `MLModel` содержит 26 признаков.

Новые или усиленные группы признаков:

- сколько раз пользователь apply-ился в этой локации;
- сколько раз пользователь apply-ился на этот `task_type`;
- сколько раз пользователь apply-ился на эту конкретную смену;
- сколько раз пользователь завершал эту конкретную смену;
- сколько раз пользователь отменял эту конкретную смену;
- recency последнего apply на эту смену;
- исправленный расчет `active_days` как количества календарных дней, а не уникальных timestamp.

Ключевые внутренние структуры:

```python
self._apply_map
self._finish_map
self._cancel_map
self._apply_ts_map
```

Они используются для recurring-shift признаков и rerank по недавнему apply.

### Векторизованный inference cache

Добавлен метод:

```python
build_inference_cache(users)
```

Он заранее строит numpy-массивы:

- apply rates;
- finish rates;
- cancel rates;
- total applies;
- total views;
- active days;
- has_mk;
- avg rewards;
- locations;
- employer/workplace sets;
- location/task-type apply counts.

После этого `predict_scores` может не строить признаки построчно через pandas/dict для каждого пользователя, а использовать `_predict_scores_vectorized`.

Это критично, потому что `predict` теперь может скорить до 5000 кандидатов на смену.

### Rerank по recency

После ML scoring применяется дополнительная сортировка:

```python
_rerank_by_apply_recency(...)
```

Смысл: если пользователь уже apply-ился на эту конкретную смену, пользователи с более свежим apply получают приоритет.

Сортировка концептуально:

1. Сначала пользователи с prior apply на эту смену.
2. Среди них - чем меньше дней с последнего apply, тем выше.
3. Затем - ML score.
4. Пользователи без prior apply идут после них.

Это сильно использует историю конкретной пары `(user_id, shift_id)`.

### Изменения тестов

Тесты обновлены из-за нового жизненного цикла сервиса.

`PrepareManager` теперь требует `db_path`:

```python
PrepareManager(db_path=db_path, sleep_seconds=0)
```

Smoke-тест теперь перед `predict` вызывает:

```python
prepare
ready
predict
```

Это отражает новое поведение: `predict` без готового prepare должен возвращать `503`.

## Итоговое состояние ключевых компонентов

## predict

Финальный `predict` работает так:

1. Если `prepare_manager.ready == False`, возвращает:

```python
{"user_ids": [], "status_code": 503, "detail": "model is in prepare state"}
```

2. Валидирует `PredictRequest`.

3. Получает кандидатов из `PrepareManager.get_candidates`.

4. Если кандидатов нет, возвращает:

```python
{"user_ids": [], "status_code": 400, "detail": "no users loaded"}
```

5. Если модель обучена:

- строит `shift_dict`;
- скорит кандидатов через `model.predict_scores`;
- возвращает top `request.limit`.

6. Если модель не обучена:

- возвращает первые `request.limit` кандидатов из кэша активности.

## prepare

Финальный `prepare`:

1. Загружает все данные из SQLite.
2. Приводит даты.
3. Строит user/location/global кэши.
4. Обучает ML-модель в executor.
5. Строит inference cache.
6. Выставляет `ready=True`.

## ML training

Позитивный класс:

```text
уникальные пары user_id/shift_id с interaction == APPLY,
исключая смены с SYSTEM_CANCEL
```

Негативный класс:

```text
уникальные VIEW пары, для которых не было APPLY
```

Количество негативов ограничено:

```python
min(len(negatives), len(applies) * 5)
```

Модель:

```python
lightgbm.LGBMClassifier
```

Основные параметры:

```python
n_estimators=500
learning_rate=0.05
num_leaves=63
max_depth=6
scale_pos_weight=pos_weight
n_jobs=2
random_state=42
```

Fallback при отсутствии LightGBM:

```python
LogisticRegression(class_weight="balanced")
```

## Исправления относительно val-split

В отличие от финального состояния `val-split`, в `LightGBM-added` файл `data/validation/apply.csv` уже согласован со split-даты:

```text
Всего записей: 45
date < 2026-02-15: 0
Минимальная дата: 2026-02-17
Максимальная дата: 2026-02-28
```

Это важное исправление: evaluator больше не будет видеть январские даты в `apply.csv`.

## Риски и замечания

## 1. Validation-данные остаются внутри train-файлов

Проверка финального состояния ветки:

```text
validation shift_id, найденных в data/train/shift.csv: 285 из 285
validation event id, найденных в data/train/event.csv: 3015 из 3015
```

Это означает, что стандартная eval-команда:

```bash
--shift-path data/train/shift.csv
--event-path data/train/event.csv
```

загрузит в сервис train-файлы, которые уже содержат validation-смены и validation-события.

Практический риск: модель может увидеть validation-период до дневной симуляции, что создает утечку и завышает качество.

## 2. Файлы shift_train.csv и event_train.csv не закоммичены

`create_validation_split.py` сохраняет корректно отрезанную train-часть сюда:

```text
data/train/shift_train.csv
data/train/event_train.csv
```

Но в итоговой ветке этих файлов нет.

Если использовать стандартные пути `data/train/shift.csv` и `data/train/event.csv`, split фактически не применяется к train-входу eval.

## 3. PrepareManager больше не использует sleep_seconds

В `main` до изменений `PrepareManager` был тестовой/асинхронной заглушкой с `sleep_seconds`.

В финальной версии `PrepareManager.__init__` принимает `sleep_seconds`, но `_background_prepare` больше не делает:

```python
await asyncio.sleep(self._sleep_seconds)
```

Это может быть нормально для production-логики, но параметр стал фактически неиспользуемым. E2E-тесты, которые передают `prepare_sleep_seconds`, больше не проверяют задержку prepare.

## 4. При ошибке prepare сервис может стать ready без рабочей модели

В `_background_prepare` при исключении:

```python
except Exception:
    self._state.ready = True
```

Это позволяет сервису отвечать на `ready=True`, даже если подготовка упала.

Если кэш кандидатов не был построен до исключения, `predict` может обратиться к `_location_cache`, который не инициализирован в `__init__`.

Практический риск: после ошибки prepare сервис выглядит готовым, но predict может падать или работать в деградированном режиме.

## 5. need_mk передается в get_candidates, но не фильтруется

Финальная сигнатура:

```python
get_candidates(location_id: str, need_mk: bool, limit: int)
```

Но внутри `get_candidates` параметр `need_mk` не используется для фильтрации кандидатов.

Совместимость по медкнижке учитывается как ML-признак `mk_compatible`, но пользователь без `has_mk` все равно может попасть в выдачу для смены `need_mk=True`, если модель/ранжирование сочтет его высоким.

Если по регламенту медкнижка должна быть жестким фильтром, это баг. Если это только признак предпочтения, поведение допустимо.

## 6. Rerank по prior apply может быть слишком сильным

После ML scoring применяется сортировка, которая поднимает пользователей с prior apply на эту конкретную смену выше пользователей без prior apply.

Это может резко улучшать метрику, если validation-смены уже есть в загруженной истории. Но при честной дневной симуляции такая информация не должна быть доступна до дня оценки.

С учетом того, что validation-события присутствуют в `data/train/event.csv`, это усиливает риск утечки.

## 7. Большие CSV в Git

Ветка добавляет сотни тысяч строк CSV прямо в репозиторий.

Для хакатона это может быть допустимо, если данные должны быть локально доступны без внешнего хранилища.

Минусы:

- тяжелый diff;
- сложный review;
- рост репозитория;
- риск случайных конфликтов при перегенерации данных.

## 8. get_users_by_ids добавлен, но не используется в финальном predict

В `Repository` добавлен метод:

```python
get_users_by_ids
```

Но финальная online-логика использует `pm._users_cache` и не обращается к этому методу.

Это не ломает поведение, но выглядит как остаток промежуточной реализации.

## Позитивные изменения

- `predict` больше не делает тяжелые SQL JOIN/агрегации на каждый запрос.
- `prepare` материализует данные и кэши заранее.
- Добавлено реальное ML-ранжирование вместо простого сортирования кандидатов.
- Добавлен LightGBM с fallback на sklearn.
- Добавлены признаки по истории пользователя, смене, работодателю, workplace, локации и recurring-shift поведению.
- Добавлен векторизованный scoring для большого пула кандидатов.
- Тесты обновлены под реальный контракт `prepare -> ready -> predict`.
- `apply.csv` исправлен: больше нет дат до split.

## Рекомендации

1. Устранить утечку validation-данных:

```text
data/train/shift.csv не должен содержать data/validation/shift.csv
data/train/event.csv не должен содержать data/validation/event.csv
```

2. Либо заменить `data/train/shift.csv` и `data/train/event.csv` на отрезанные train-файлы, либо изменить eval-команду на:

```bash
--shift-path data/train/shift_train.csv
--event-path data/train/event_train.csv
```

и закоммитить эти файлы.

3. Инициализировать `_location_cache` в `PrepareManager.__init__`:

```python
self._location_cache: dict[str, list[str]] = {}
```

4. При ошибке prepare не выставлять `ready=True` без явно обозначенного degraded state.

5. Решить, должен ли `need_mk` быть жестким фильтром. Если да, фильтровать кандидатов до ML scoring.

6. Удалить неиспользуемый `sleep_seconds` или вернуть его применение в тестовом режиме.

7. Проверить, не переусиливает ли `_rerank_by_apply_recency` leakage-сигнал при наличии validation-событий в train.

8. Добавить тесты на:

- `predict` после failed prepare;
- `need_mk=True`;
- отсутствие пересечения train/validation split;
- корректность `apply.date >= SPLIT_DATE`;
- `get_candidates` для маленькой и большой локации;
- стабильность `predict` при пустых events/shifts.

## Финальный вывод

Ветка `LightGBM-added` существенно меняет архитектуру сервиса: простой baseline-ranking заменен на prepare-time ML training, in-memory candidate cache и LightGBM-based scoring.

С инженерной точки зрения направление правильное: дорогая работа вынесена из `predict` в `prepare`, а online-часть оптимизирована через кэши и векторизацию.

Главный блокер для честной оценки - состояние данных: validation-смены и validation-события остаются в train CSV, которые использует стандартная eval-команда. Пока это не исправлено, метрика на этой ветке может быть завышена из-за временной утечки.
