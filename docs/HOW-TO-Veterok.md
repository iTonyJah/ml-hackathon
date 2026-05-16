# Как воспроизвести результаты (release/sleeper-rerank)

```bash
git clone https://github.com/iTonyJah/ml-hackathon.git
git branch --all
git switch release/sleeper-rerank

# Установка зависимостей
make install

# Копируем полученные файлы в data/train

# Создание валидационного сплита
# (обязательно перед eval для участника так data/validation у нас нет)
# (заказчик и проверяющий копируют файлы в data/validation самостоятельно)
poetry run python scripts/create_validation_split.py

# или для быстроты val на последних двух днях
poetry run python scripts/create_validation_split_2d.py

# Создание базы данных
make migrate

# Запуск сервиса (в отдельном терминале)
make run

# Официальный eval для участника
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

# Официальный eval для заказчика/проверяющего
(заказчик и проверяющий копируют файлы в data/train, data/validation самостоятельно)
```bash
# Официальный eval для заказчика/проверяющего
poetry run python -m hackaton.eval.cli run \
  --user-path data/train/user.csv \
  --shift-path data/train/shift.csv \
  --event-path data/train/event.csv \
  --val-apply-path data/validation/apply.csv \
  --val-shift-path data/validation/shift.csv \
  --val-event-path data/validation/event.csv \
  --output-dir artifacts/eval_run \
  --limit 10 \
  --batch-size 1000
```
