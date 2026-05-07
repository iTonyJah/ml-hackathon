# Как работать с нашими изменениями

Этот файл описывает дополнительные сценарии, которые появились поверх базового проекта.
Общий маршрут участника остается в `HOW-TO.md`.

## Локальное validation-разбиение из train

Если выдали только:

- `data/train/user.csv`
- `data/train/shift.csv`
- `data/train/event.csv`

можно сделать временное разбиение по времени:

```bash
python3 scripts/split_train_validation.py --force
```

По умолчанию скрипт оставляет исходный `data/train` без изменений и пишет:

- train-часть в `data/train_split`;
- validation-часть в `data/validation`.

`data/validation/apply.csv` формируется по официальной методике локальной оценки:

- positive label создается только по событию `APPLY`;
- `FINISHED` не считается label для `apply.csv`;
- цепочки с `SYSTEM_CANCEL` исключаются из `apply.csv`;
- остальные validation-события остаются в `event.csv` и могут использоваться как история.

## Eval после локального разбиения

После запуска `scripts/split_train_validation.py` используйте validation-файлы из
`data/validation`, а train-пути в eval замените на:

- `data/train_split/user.csv`
- `data/train_split/shift.csv`
- `data/train_split/event.csv`

Команда eval:

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

Итоговый отчет появится в `artifacts/eval_run/eval_report.md`.

## Связанные файлы

- `scripts/split_train_validation.py` — скрипт разбиения train-данных на train/validation.
- `tests/unit/test_split_train_validation.py` — unit-тесты для сценария разбиения.
