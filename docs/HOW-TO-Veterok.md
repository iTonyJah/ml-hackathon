# Как воспроизвести результаты (release/sleeper-rerank)

```bash
git clone https://github.com/iTonyJah/ml-hackathon.git
git branch --all
git switch release/sleeper-rerank

# Установка зависимостей
make install

# Копируем полученные файлы в data/train

# Создание валидационного сплита (обязательно перед eval)
poetry run python create_validation_split.py

# Создание базы данных
make migrate

# Запуск сервиса (в отдельном терминале)
make run

# Официальный eval
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
```

# Time-based CV (запускается отдельно, без сервиса)
```bash
poetry run python -m hackaton.train.cli cv \
  --user-path data/train/user.csv \
  --shift-path data/train/shift_train.csv \
  --event-path data/train/event_train.csv \
  --output-dir artifacts/cv_run \
  --val-days 30
```

Результаты CV сохраняются в `artifacts/cv_run/cv_report.md`.
