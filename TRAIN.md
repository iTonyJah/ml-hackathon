# Обучение, модель и CV

В проекте есть два разных смысла слова "обучение".

Первый - legacy/baseline train pipeline из `hackaton/train/training.py`. Он оставлен как отдельный воспроизводимый pipeline.

Второй - финальный online-путь: модель обучается внутри `PrepareManager` во время вызова `prepare`. Именно этот путь используется сервисом для `predict`.

## Финальный ML-путь

Основные файлы:

- `hackaton/service/ml_model.py`;
- `hackaton/service/prepare_manager.py`;
- `hackaton/service/app.py`;
- `hackaton/train/cv.py`.

При вызове `prepare` происходит:

1. Загрузка `users`, `shifts`, `events` из SQLite.
2. Приведение дат.
3. Построение статистик пользователей.
4. Формирование обучающих пар.
5. Обучение `LGBMClassifier`.
6. AUC early stopping на отложенной части данных.
7. Построение inference cache.
8. Построение candidate cache по локациям.

## Формирование target

Позитивный пример:

- есть `APPLY` для пары `user_id + shift_id`;
- событие произошло до `shift.start_at`;
- смена не исключена из-за `SYSTEM_CANCEL`.

Негативный пример:

- был `VIEW` или `USER_CANCEL`;
- для этой пары нет последующего `APPLY`.

Чтобы не перекосить выборку, негативы семплируются относительно числа позитивов.

## Признаки модели

Модель использует несколько групп признаков.

История пользователя:

- apply rate;
- finish rate;
- cancel rate;
- число заявок;
- число просмотров;
- активные дни;
- наличие медкнижки.

Специализация:

- заявки пользователя в этой локации;
- заявки пользователя на этот тип задачи.

Параметры смены:

- час начала;
- день недели;
- длительность;
- оплата;
- capacity;
- need_mk;
- id_differential.

Совместимость:

- совпадение локации;
- совместимость по медкнижке;
- работал ли пользователь с работодателем;
- работал ли на workplace;
- средняя fill-rate статистика работодателя;
- отношение оплаты к средней оплате пользователя.

Повторяющиеся смены:

- сколько раз пользователь подавал заявку на эту смену;
- сколько раз выходил;
- сколько раз отменял;
- как давно была последняя заявка.

## Модель

Основной алгоритм:

```text
LightGBM LGBMClassifier
```

Используется:

- `n_estimators=1000`;
- `learning_rate=0.05`;
- `num_leaves=63`;
- `max_depth=6`;
- `scale_pos_weight`;
- AUC early stopping.

Если LightGBM недоступен, предусмотрен fallback на `LogisticRegression`, но основной путь решения - LightGBM.

## Sleeper-rerank

После ML-скоринга применяется дополнительный rerank.

Отдельно поднимаются пользователи, у которых:

- нет `APPLY`;
- есть `FINISHED`.

Такие пользователи могли подать заявку до начала обучающего окна, но факт выхода на смену показывает, что они не случайные. Поэтому они получают отдельный tier в ранжировании.

## Legacy train pipeline

Команда ниже запускает baseline train pipeline из `hackaton/train/training.py`:

```bash
poetry run python -m hackaton.train.cli train \
  --user-path data/train/user.csv \
  --shift-path data/train/shift.csv \
  --event-path data/train/event.csv \
  --output-dir artifacts/train \
  --skip-shap
```

Он сохраняет:

- `model.pkl`;
- `metrics.json`;
- `feature_schema.json`;
- `train_config.json`;
- `data_contract_check.json`;
- `train_report.md`.

Этот pipeline полезен как отдельный baseline и диагностический инструмент, но финальный online-сервис использует ML-путь через `PrepareManager`.

## Time-based CV

CV нужен, чтобы оценивать модель надежнее, чем на маленьком validation-наборе.

Запуск:

```bash
poetry run python -m hackaton.train.cli cv \
  --user-path data/train/user.csv \
  --shift-path data/train/shift_train.csv \
  --event-path data/train/event_train.csv \
  --output-dir artifacts/cv_run \
  --val-days 30 \
  --candidate-limit 300
```

Что делает CV:

1. Делит историю по времени.
2. Обучает модель на прошлом.
3. Проверяет на будущем.
4. Использует тот же смысл candidate generation.
5. Считает целевую метрику через `hackaton/eval/metric.py`.

Результат:

```text
artifacts/cv_run/cv_report.md
```

## Что смотреть в отчетах

В train/eval/CV отчетах важны:

- итоговая метрика;
- число оцениваемых дней;
- число оцениваемых смен;
- метрики по дням;
- метрики по `capacity`;
- latency для `predict`;
- длительность `prepare`.

Если оцениваемых смен мало, итоговая метрика может быть шумной. Именно поэтому в проект добавлен time-based CV.
