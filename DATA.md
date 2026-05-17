# DATA: форматы train/validation данных

Этот документ описывает CSV-файлы, которые используются сервисом, локальным eval и validation split.

## Train-данные

Ожидаемые файлы:

```text
data/train/user.csv
data/train/shift.csv
data/train/event.csv
```

### `user.csv`

Колонки:

| Колонка | Тип | Описание |
|---|---|---|
| `location_id` | string | идентификатор локации пользователя |
| `is_strict_location` | bool | пользователь явно выбрал локацию |
| `id` | string | идентификатор пользователя |
| `has_mk` | bool | наличие медкнижки |

Пример:

```csv
location_id,is_strict_location,id,has_mk
loc_1,true,u_1001,true
loc_2,false,u_1002,false
```

### `shift.csv`

Колонки:

| Колонка | Тип | Описание |
|---|---|---|
| `id` | string | идентификатор смены |
| `start_at` | datetime | время начала смены |
| `location_id` | string | локация смены |
| `task_type` | string | тип задачи |
| `employer_id` | string | работодатель |
| `workplace_id` | string | рабочая точка |
| `need_mk` | bool | нужна ли медкнижка |
| `id_differential` | bool | специальный признак ставки |
| `hours` | int | длительность |
| `reward` | float | оплата |
| `capacity` | int | сколько работников нужно |

Пример:

```csv
id,start_at,location_id,task_type,employer_id,workplace_id,need_mk,id_differential,hours,reward,capacity
s_501,2026-03-24T08:00:00Z,loc_1,picker,e_10,w_77,true,false,8,1800.0,3
s_502,2026-03-24T10:00:00Z,loc_2,loader,e_12,w_90,false,false,6,1400.0,2
```

### `event.csv`

Колонки:

| Колонка | Тип | Описание |
|---|---|---|
| `id` | uuid/string | идентификатор события |
| `shift_id` | string | идентификатор смены |
| `user_id` | string | идентификатор пользователя |
| `interaction` | string | тип события |
| `ts` | datetime | время события |

Допустимые `interaction`:

- `VIEW`;
- `APPLY`;
- `FINISHED`;
- `USER_CANCEL`;
- `SYSTEM_CANCEL`.

Пример:

```csv
id,shift_id,user_id,interaction,ts
9f8f2ec9-b213-4f80-a2cb-66065a9e8cb3,s_501,u_1001,VIEW,2026-03-23T15:00:00Z
4d04eec1-3ccb-4fd8-bbd5-43e535d18ef6,s_501,u_1001,APPLY,2026-03-23T15:05:00Z
```

## Validation-данные

Ожидаемые файлы:

```text
data/validation/apply.csv
data/validation/shift.csv
data/validation/event.csv
```

### `validation/apply.csv`

Колонки:

| Колонка | Тип | Описание |
|---|---|---|
| `user_id` | string | пользователь, который реально подал заявку |
| `shift_id` | string | смена, на которую подана заявка |
| `date` | date | дата начала смены |

Пример:

```csv
user_id,shift_id,date
u_1001,s_601,2026-03-25
u_1002,s_602,2026-03-25
```

Важно: `date` - это дата начала смены `shift.start_at.date()`, а не дата события `APPLY`. Пользователь может откликнуться заранее, но eval группирует проверки по дате смены.

Также в ground truth не должны попадать `APPLY`, которые произошли после `shift.start_at`.

### `validation/shift.csv`

Схема такая же, как у `train/shift.csv`.

### `validation/event.csv`

Схема такая же, как у `train/event.csv`.

## Файлы после локального split

Если validation создается локально через:

```bash
poetry run python scripts/create_validation_split.py
```

или:

```bash
poetry run python scripts/create_validation_split_2d.py
```

то дополнительно создаются:

```text
data/train/shift_train.csv
data/train/event_train.csv
data/validation/apply.csv
data/validation/shift.csv
data/validation/event.csv
data/validation/users.csv
```

В этом сценарии для eval нужно передавать:

- `data/train/user.csv`;
- `data/train/shift_train.csv`;
- `data/train/event_train.csv`;
- validation-файлы из `data/validation`.

## Проверки корректности данных

Перед eval полезно проверить:

- все обязательные колонки есть;
- `apply.shift_id` пересекается с `validation/shift.id`;
- `apply.date` совпадает с `validation/shift.start_at.date()`;
- в `apply.csv` нет заявок после начала смены;
- train-события не содержат будущей информации после split-даты;
- даты успешно парсятся как timezone-aware datetime.

## Где читать дальше

- `REGLAMENT.md` - как считается метрика.
- `docs/REPRODUCIBILITY.md` - как подготовить данные и запустить eval.
- `docs/METRICS_AND_EXPERIMENTS.md` - почему split и `capacity` сильно влияют на результат.
