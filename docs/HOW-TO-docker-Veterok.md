# Создаем директорию с данными
mkdir -p workdata/train

# Копируем файлы датасета в директорию workdata/train

# Создаем директорию для хранения артефактов
mkdir artifacts

# Клонируем решение из Git-репозитория
git clone https://github.com/iTonyJah/ml-hackathon.git
cd ml-hackathon

# Просматриваем доступные ветки
git branch --all

# Переключаемся на нужную ветку
git switch release/sleeper-rerank

# Собираем Docker image со всеми необходимыми зависимостями
docker compose build --no-cache

# Запускаем сервис
docker compose up

# В отдельном терминале:
# Получаем ID запущенного Docker-контейнера (CONTAINER ID)
docker ps

# Создание валидационного сплита
docker exec -it CONTAINER_ID poetry run python scripts/create_validation_split.py

# Или для ускоренного варианта — validation на последних двух днях
docker exec -it CONTAINER_ID poetry run python scripts/create_validation_split_2d.py

# Создание базы данных
docker exec -it CONTAINER_ID make migrate

# Официальный eval для участника
docker exec -it CONTAINER_ID poetry run python -m hackaton.eval.cli run \
  --user-path data/train/user.csv \
  --shift-path data/train/shift_train.csv \
  --event-path data/train/event_train.csv \
  --val-apply-path data/validation/apply.csv \
  --val-shift-path data/validation/shift.csv \
  --val-event-path data/validation/event.csv \
  --output-dir artifacts/eval_run \
  --limit 10 \
  --batch-size 1000

# В результате выполнения команд
# в ранее созданной директории artifacts
# будут находиться результаты работы модели
