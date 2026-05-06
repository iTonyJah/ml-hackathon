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

Он формируется автоматически из позитивных взаимодействий пользователей с validation-сменами.

Позитивными взаимодействиями считаются:

```text
APPLY
FINISHED
```

Для каждого такого события скрипт добавляет строку:

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

## 11. Итог

В результате выполненной работы в проекте появился воспроизводимый локальный путь от исходных
train CSV до eval-ready validation-набора:

1. `scripts/split_train_validation.py` готовит `data/train_split` и `data/validation`.
2. `data/validation/apply.csv` формируется автоматически из позитивных interactions.
3. `OUR-CHANGES.md` описывает, как пользоваться новым сценарием.
4. `tests/unit/test_split_train_validation.py` фиксирует ключевой контракт split-логики.
5. `WORK_REPORT.md` фиксирует подробный отчет о выполненной работе.
