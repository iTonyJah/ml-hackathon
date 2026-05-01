# Отчет по коммитам ветки val-split

Дата анализа: 2026-05-01

Анализируемая ветка:

```text
origin/val-split
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

Ветка `val-split` добавляет в проект локальные CSV-данные для train/validation, скрипт генерации validation split и вспомогательный shell-скрипт для освобождения порта `8000`.

Всего в ветке 3 коммита поверх `main`:

```text
71041c5 Выделение валидационной выборки
dbd5ea6 Фиксы
56a2a6e Фиксы
```

Итоговый diff относительно `main`:

```text
9 files changed, 437816 insertions(+)
```

Все итоговые изменения в ветке являются добавлением новых файлов. Удалений и модификаций файлов из `main` в итоговом diff нет.

## Итоговые добавленные файлы

```text
A create_validation_split.py
A data/train/event.csv
A data/train/shift.csv
A data/train/user.csv
A data/validation/apply.csv
A data/validation/event.csv
A data/validation/shift.csv
A data/validation/users.csv
A kill_8000.sh
```

Итоговая статистика по строкам:

| Файл | Добавлено строк |
|---|---:|
| `create_validation_split.py` | 175 |
| `data/train/event.csv` | 383884 |
| `data/train/shift.csv` | 40000 |
| `data/train/user.csv` | 5155 |
| `data/validation/apply.csv` | 140 |
| `data/validation/event.csv` | 3016 |
| `data/validation/shift.csv` | 286 |
| `data/validation/users.csv` | 5155 |
| `kill_8000.sh` | 5 |

Количество строк CSV включает строку заголовка. По данным `ConvertFrom-Csv` фактическое количество записей:

| Файл | Записей без заголовка |
|---|---:|
| `data/train/event.csv` | 383883 |
| `data/train/shift.csv` | 39999 |
| `data/train/user.csv` | 5154 |
| `data/validation/apply.csv` | 139 |
| `data/validation/event.csv` | 3015 |
| `data/validation/shift.csv` | 285 |
| `data/validation/users.csv` | 5154 |

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

Автор:

```text
Денис Богомолов <bogomolovdi@it2g.ru>
```

### Что добавлено

Коммит добавляет основную массу ветки:

```text
create_validation_split.py |    133 +
data/train/event.csv       | 383884 +
data/train/shift.csv       |  40000 +
data/train/user.csv        |   5155 +
data/validation/apply.csv  |    140 +
data/validation/event.csv  |   3016 +
data/validation/shift.csv  |    286 +
data/validation/users.csv  |   5155 +
kill_8000.sh               |      5 +
```

### Смысл изменения

Коммит впервые добавляет локальный набор данных:

- train-пользователи;
- train-смены;
- train-события;
- validation-смены;
- validation-события;
- validation apply-разметка;
- копия пользователей в `data/validation/users.csv`.

Также добавляется скрипт `create_validation_split.py`, который должен создавать validation split из train-данных по дате:

```python
SPLIT_DATE = pd.Timestamp("2026-02-15", tz="UTC")
```

Логика начальной версии:

- читает `data/train/shift.csv`, `data/train/event.csv`, `data/train/user.csv`;
- делит смены на train и validation по `start_at < 2026-02-15`;
- train-события берет по `ts < 2026-02-15`;
- validation-события берет по `shift_id` из validation-смен;
- строит `apply.csv` из событий `APPLY`;
- исключает смены, у которых есть `SYSTEM_CANCEL`;
- сохраняет validation-файлы в `data/validation`.

### Добавленный kill_8000.sh

Файл `kill_8000.sh`:

```bash
#!/bin/bash
echo "Killing process on port 8000..."
fuser -k 8000/tcp 2>/dev/null || true
sleep 1
echo "Port 8000 is free"
```

Назначение: принудительно завершить процесс, занимающий TCP-порт `8000`.

Практически это удобно перед запуском сервиса, если предыдущий процесс не был остановлен.

### Замечания по первому коммиту

В начальной версии `create_validation_split.py` есть несколько проблем, которые частично исправлены следующими коммитами:

- есть неиспользуемый импорт `os`;
- в выводе используются unicode-стрелки `→`, что не критично, но хуже для ASCII-совместимости;
- `apply.csv` сохраняется с колонкой `ts`, а evaluator проекта ожидает колонку `date`;
- в строке вывода есть опечатка `print(f"\apply.csv:")`;
- `apply.csv` не фильтруется явно по `events["ts"] >= SPLIT_DATE`, поэтому в разметку могут попасть apply-события до даты split.

## Коммит dbd5ea6

Сообщение:

```text
Фиксы
```

Дата:

```text
Tue Apr 28 12:48:17 2026 +0300
```

Автор:

```text
Денис Богомолов <bogomolovdi@it2g.ru>
```

### Что изменено

```text
create_validation_split.py |  15 ++-
data/validation/apply.csv  | 280 ++++++++++++++++++++++-----------------------
```

### Изменения в create_validation_split.py

Коммит исправляет несколько технических проблем:

- удаляет неиспользуемый импорт `os`;
- заменяет unicode-стрелку `→` на ASCII-вариант `->` в диагностическом выводе;
- исправляет вывод `apply.csv`;
- преобразует колонку `ts` в колонку `date`;
- сохраняет `apply.csv` с колонками:

```text
user_id,shift_id,date
```

Это важно, потому что evaluator в `hackaton/eval/evaluator.py` читает именно колонку `date`:

```python
val_apply["date"] = pd.to_datetime(val_apply["date"], errors="coerce").dt.date
```

### Изменения в data/validation/apply.csv

Файл `apply.csv` перегенерирован в новом формате.

До фикса:

```text
user_id,shift_id,ts
...
```

После фикса:

```text
user_id,shift_id,date
...
```

То есть исправлена схема validation-разметки под контракт evaluator.

### Оставшаяся проблема после dbd5ea6

Несмотря на замену `ts` на `date`, сам набор строк все еще содержит даты до `2026-02-15`.

Это видно уже в diff коммита: в `apply.csv` остаются значения вроде:

```text
2026-01-20
2026-01-24
2026-02-08
2026-02-14
```

Следующий коммит исправляет генератор, но не перегенерирует закоммиченный `apply.csv`.

## Коммит 56a2a6e

Сообщение:

```text
Фиксы
```

Дата:

```text
Wed Apr 29 13:15:50 2026 +0300
```

Автор:

```text
Денис Богомолов <bogomolovdi@it2g.ru>
```

### Что изменено

```text
create_validation_split.py | 45 +++++++++++++++++++++++++++++++++++++++++----
```

### Смысл изменения

Коммит улучшает финальную версию генератора validation split:

- добавляет `import datetime`;
- добавляет `SPLIT_DATE_PLAIN = datetime.date(2026, 2, 15)`;
- улучшает диагностический вывод;
- документирует правила построения `apply.csv`;
- добавляет явную фильтрацию `APPLY` по `events["ts"] >= SPLIT_DATE`;
- добавляет вывод диапазона дат в `apply`;
- добавляет функцию `validate_split`;
- вызывает `validate_split` перед сохранением файлов.

### Новая фильтрация apply

Финальная версия `build_apply` добавляет условие:

```python
& (events["ts"] >= SPLIT_DATE)
```

Это правильное исправление: `apply.csv` должен содержать только validation-период, иначе в разметке оказываются события до split-даты.

### Новая validate_split

Добавлена функция:

```python
def validate_split(
    apply: pd.DataFrame,
    val_shifts: pd.DataFrame,
    train_events: pd.DataFrame,
) -> None:
```

Она проверяет:

- все даты в `apply.csv` должны быть `>= 2026-02-15`;
- `shift_id` из `apply` должны пересекаться с `val_shifts`;
- `train_events` не должен содержать события после split-даты.

Это полезный guardrail: скрипт теперь явно диагностирует временные утечки.

### Важное несоответствие

Коммит исправляет код генератора, но не обновляет `data/validation/apply.csv`.

В итоговом состоянии ветки `origin/val-split` файл `data/validation/apply.csv` все еще содержит даты до `2026-02-15`.

Проверка показала:

```text
Всего записей в apply.csv: 139
Записей с date < 2026-02-15: 96
Минимальная дата: 2026-01-20
Максимальная дата: 2026-02-27
```

Это означает, что закоммиченный `apply.csv` не соответствует финальной логике `create_validation_split.py`.

## Итоговый анализ финального состояния ветки

## create_validation_split.py

Финальный скрипт предназначен для создания validation split из исходных train CSV.

Главные константы:

```python
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"

SPLIT_DATE = pd.Timestamp("2026-02-15", tz="UTC")
SPLIT_DATE_PLAIN = datetime.date(2026, 2, 15)
```

Основные функции:

| Функция | Назначение |
|---|---|
| `load_data` | читает `shift.csv`, `event.csv`, `user.csv` из `data/train` |
| `split_shifts` | делит смены на train/validation по `start_at` |
| `split_events` | делит события на train-events и validation-events |
| `build_apply` | строит `apply.csv` из `APPLY`-событий validation-периода |
| `validate_split` | проверяет даты и отсутствие утечек в train-events |
| `save_files` | сохраняет split-файлы |

### Что сохраняет скрипт

Финальный скрипт сохраняет:

```text
data/train/shift_train.csv
data/train/event_train.csv
data/validation/shift.csv
data/validation/event.csv
data/validation/apply.csv
data/validation/users.csv
```

## Важные риски и несоответствия

## 1. Закоммиченный apply.csv не соответствует финальному генератору

Финальный генератор требует:

```text
apply.date >= 2026-02-15
```

Но в закоммиченном `data/validation/apply.csv`:

```text
96 из 139 записей имеют date < 2026-02-15
```

Это означает, что после изменения `create_validation_split.py` файл `apply.csv`, вероятно, не был перегенерирован и закоммичен.

Практическое последствие: evaluator будет оценивать дни из `apply.csv`, включая январь и начало февраля, хотя `validation/shift.csv` содержит смены validation-периода. Для дней, где `apply` есть, а validation-смен дня нет, evaluator просто пропустит день. Это может исказить ожидаемый набор validation-дней и усложнить интерпретацию отчета.

## 2. Validation-смены и validation-события остаются внутри train-файлов

Проверка итогового состояния ветки показала:

```text
validation shift_id, найденных в data/train/shift.csv: 285 из 285
validation event id, найденных в data/train/event.csv: 3015 из 3015
```

То есть `data/train/shift.csv` и `data/train/event.csv` в ветке содержат полный исходный набор, включая validation-часть.

При этом `create_validation_split.py` сохраняет train-часть в:

```text
data/train/shift_train.csv
data/train/event_train.csv
```

Но эти файлы не добавлены в итоговый diff ветки.

### Почему это важно

Eval-команда проекта использует:

```bash
--shift-path data/train/shift.csv
--event-path data/train/event.csv
```

Если запускать eval с такими путями на состоянии ветки `val-split`, сервис получит train-файлы, которые уже содержат validation-смены и validation-события.

Это создает риск временной утечки: модель или кэши сервиса могут увидеть данные validation-периода до начала дневной симуляции.

## 3. validation/users.csv дублирует train/user.csv

`data/validation/users.csv` содержит 5154 пользователя, столько же, сколько `data/train/user.csv`.

Это не обязательно ошибка: пользователи могут быть общей справочной таблицей. Но текущий eval-пайплайн не использует `data/validation/users.csv`; он принимает только `--user-path data/train/user.csv`.

Практически это файл-дубль, который увеличивает размер ветки и может запутывать контракт данных.

## 4. Большие CSV добавлены прямо в Git

Ветка добавляет более 437 тысяч строк, почти все из которых - CSV-данные.

Самые большие файлы:

- `data/train/event.csv`: 383883 записей;
- `data/train/shift.csv`: 39999 записей.

Практические последствия:

- репозиторий становится тяжелее;
- diffs становятся шумными;
- code review сложнее;
- любые правки CSV будут выглядеть как большие текстовые изменения.

Для хакатонного проекта это может быть приемлемо, если данные специально должны лежать в репозитории. Для production-подхода лучше хранить такие данные вне Git или через отдельный механизм артефактов.

## 5. kill_8000.sh платформенно-зависим

`kill_8000.sh` использует:

```bash
fuser -k 8000/tcp
```

Это работает в Linux-like окружении, но не является переносимым решением для Windows/PowerShell.

С учетом того, что проект может запускаться локально на разных ОС, скрипт полезен, но его стоит воспринимать как Linux/macOS helper, а не универсальную команду.

## Влияние на eval-процесс

Ветка явно направлена на то, чтобы появились файлы для команды:

```bash
poetry run python -m hackaton.eval.cli run \
  --user-path data/train/user.csv \
  --shift-path data/train/shift.csv \
  --event-path data/train/event.csv \
  --val-apply-path data/validation/apply.csv \
  --val-shift-path data/validation/shift.csv \
  --val-event-path data/validation/event.csv
```

Положительное влияние:

- появляются реальные локальные CSV для запуска eval;
- появляется `apply.csv`, без которого невозможно считать целевую метрику;
- появляется скрипт, документирующий способ получения validation split;
- добавлена проверка split на утечки в финальной версии скрипта.

Проблемное влияние:

- текущие `data/train/shift.csv` и `data/train/event.csv` содержат validation-часть;
- финальный `apply.csv` не соответствует финальному коду генератора;
- generated train-split файлы `shift_train.csv` и `event_train.csv` не закоммичены, хотя именно они выглядят как корректные train-файлы после split.

## Рекомендации

1. Перегенерировать validation split финальной версией `create_validation_split.py`.

2. Закоммитить согласованный набор файлов:

```text
data/train/shift.csv
data/train/event.csv
data/validation/shift.csv
data/validation/event.csv
data/validation/apply.csv
```

или явно изменить eval-команду/документацию так, чтобы train-часть бралась из:

```text
data/train/shift_train.csv
data/train/event_train.csv
```

3. Проверить, что после split:

```text
data/train/shift.csv ∩ data/validation/shift.csv = пусто
data/train/event.csv ∩ data/validation/event.csv = пусто
```

если эти файлы используются как независимые train/validation наборы.

4. Удалить или документировать `data/validation/users.csv`, если он действительно нужен. Сейчас eval его не использует.

5. Рассмотреть перенос больших CSV из Git, если это не обязательная часть задания.

6. Добавить тест или отдельную проверку для `create_validation_split.py`, которая валидирует:

- `apply.date >= SPLIT_DATE`;
- `apply.shift_id` входит в `validation/shift.csv`;
- train-events не содержат событий после split;
- train-shifts не содержат смен после split;
- validation-shifts не пересекаются с train-shifts.

## Финальный вывод

Ветка `val-split` добавляет нужную инфраструктуру и данные для локальной validation-оценки, но финальное состояние ветки выглядит несогласованным:

- код генератора был исправлен так, чтобы не допускать apply-события до `2026-02-15`;
- при этом закоммиченный `data/validation/apply.csv` все еще содержит 96 записей до `2026-02-15`;
- validation-смены и validation-события полностью присутствуют в train-файлах, которые использует стандартная eval-команда.

Перед использованием этой ветки для честной оценки качества модели стоит перегенерировать данные и привести пути train/validation к единому контракту.
