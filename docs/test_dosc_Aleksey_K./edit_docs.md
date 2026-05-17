# Что нужно отредактировать в существующих документах проекта

Этот файл создан вместо прямого редактирования существующих документов.

Ограничение: текущие документы проекта не редактируются и не удаляются. Ниже перечислены рекомендуемые правки для финализации документации.

## `README.md`

Что нужно изменить:

- Убрать ощущение, что проект сейчас является только baseline-решением.
- Добавить краткое описание финального решения:
  - `LightGBM`;
  - обучение и кэширование в `prepare`;
  - быстрый in-memory inference в `predict`;
  - sleeper-rerank;
  - time-based CV.
- Добавить ссылку на итоговый отчет, когда он будет создан:
  - `docs/FINAL_REPORT.md` или аналогичный файл в финальном месте.
- Добавить ссылку на инструкцию воспроизводимости:
  - `docs/REPRODUCIBILITY.md`.
- Обновить команды split:
  - сейчас актуальные скрипты лежат в `scripts/create_validation_split.py` и `scripts/create_validation_split_2d.py`.
- Отдельно пояснить, что `make run` нужно запускать перед eval.

## `HOW-TO.md`

Что нужно изменить:

- Синхронизировать документ с финальным ML-пайплайном.
- Указать, что основная логика решения находится в:
  - `hackaton/service/ml_model.py`;
  - `hackaton/service/prepare_manager.py`;
  - `hackaton/service/app.py`;
  - `hackaton/train/cv.py`.
- Добавить сценарий подготовки локального validation split через `scripts/create_validation_split.py`.
- Добавить быстрый сценарий через `scripts/create_validation_split_2d.py`.
- Добавить команду запуска CV:

```bash
poetry run python -m hackaton.train.cli cv \
  --user-path data/train/user.csv \
  --shift-path data/train/shift_train.csv \
  --event-path data/train/event_train.csv \
  --output-dir artifacts/cv_run \
  --val-days 30
```

- Добавить пояснение, что `predict` использует ML-модель только после успешного `prepare`, а до этого возможен fallback.

## `TRAIN.md`

Что нужно изменить:

- Сейчас документ описывает baseline train pipeline с `LogisticRegression`.
- Нужно явно разделить:
  - legacy/baseline train pipeline;
  - финальный online-пайплайн, где модель обучается в `PrepareManager` через `MLModel`.
- Добавить описание `LGBMClassifier` и AUC early stopping.
- Добавить описание формирования таргетов:
  - positive: `APPLY` до `shift.start_at`;
  - negative: `VIEW` или `USER_CANCEL` без последующего `APPLY`;
  - исключение `SYSTEM_CANCEL`;
  - фильтрация post-start `APPLY`.
- Добавить ссылку на CV-модуль `hackaton/train/cv.py`.

## `DATA.md`

Что нужно изменить:

- Добавить важное пояснение для `apply.csv`:
  - `date` должна быть датой начала смены `shift.start_at.date()`, а не датой события `APPLY`.
- Добавить пояснение, что `APPLY` после `shift.start_at` не должен попадать в ground truth.
- Добавить пояснение про локальные файлы после split:
  - `data/train/shift_train.csv`;
  - `data/train/event_train.csv`;
  - `data/validation/shift.csv`;
  - `data/validation/event.csv`;
  - `data/validation/apply.csv`.

## `REGLAMENT.md`

Что нужно изменить:

- Аккуратно дописать объяснение, почему смены с `capacity=1` часто не дают оцениваемой ROC-AUC ситуации.
- Уточнить, что evaluator использует `apply.date` как дату оцениваемого дня.
- Добавить ссылку на подробный разбор eval-команды.
- Исправить незавершенную фразу в пункте про `predict`: сейчас есть обрыв "На".

## `docs/HOW-TO-Veterok.md`

Что нужно изменить:

- Уточнить путь к папке и имя ветки, если документ останется как финальная инструкция.
- Проверить команды с учетом актуальных путей:
  - `scripts/create_validation_split.py`;
  - `scripts/create_validation_split_2d.py`.
- Разделить два сценария:
  - участник создает validation split сам;
  - проверяющий получает готовые `data/train` и `data/validation`.
- Добавить проверку ожидаемых артефактов:
  - `artifacts/eval_run/eval_report.md`;
  - `artifacts/cv_run/cv_report.md`.
- Добавить troubleshooting:
  - занят порт `8000`;
  - сервис не запущен перед eval;
  - `ready=false` или `503 model is in prepare state`;
  - неверные пути к `shift_train.csv` и `event_train.csv`.

## `docs/support/FULL_REPORT.md`

Что нужно изменить:

- Не использовать как основной документ без переработки.
- Убрать дублирующиеся разделы и скачки нумерации.
- Разделить результаты по контекстам:
  - старый локальный val;
  - исправленный split;
  - новый датасет;
  - CV.
- Проверить формулировки про "максимум" метрики:
  - максимум зависит от конкретного validation-набора и правил агрегации.
- Сократить кодовые вставки там, где они не нужны для понимания.
- Перенести подробную историю разработки в приложение, а не в главный путь чтения.

## `docs/support/REPORT.md`

Что нужно изменить:

- Пометить как краткий исторический отчет, а не как финальную версию.
- Уточнить, что метрика `0.750` относится к конкретному validation-набору и уже не является единственным финальным результатом.
- Синхронизировать выводы с более поздними исправлениями split и target construction.

## `docs/support/CV_REPORT.md`

Что нужно изменить:

- Оставить как приложение по экспериментам.
- Уточнить, что CV нужен для выбора модели, а не заменяет официальный eval.
- Проверить команды запуска и пути к данным.
- Добавить короткий вывод в начало:
  - финальный выбор: `LGBMClassifier`;
  - причина: выше и стабильнее на time-based CV.

## `docs/support/eval_command_analysis.md`

Что нужно изменить:

- Оставить как приложение к воспроизводимости.
- Синхронизировать параметры команды с финальной инструкцией.
- Явно сказать, какие параметры используются по умолчанию, если они не указаны:
  - `--limit`;
  - `--batch-size`;
  - `--predict-max-concurrency`;
  - `--predict-max-rpm`.

## `docs/support/lightgbm_added_commits_report.md`

Что нужно изменить:

- Пометить как исторический commit-report.
- Не использовать как источник финального состояния без сверки с текущим кодом.
- Уточнить, что часть выводов относится к ветке `LightGBM-added`, а финальная ветка - `release/sleeper-rerank`.

## `docs/support/val_split_commits_report.md`

Что нужно изменить:

- Пометить как исторический commit-report.
- Добавить ссылку на финальную версию split-логики.
- Уточнить, что финальный split-скрипт находится в `scripts/`, а не в корне проекта.

