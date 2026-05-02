# Инструкция по настройке окружения для разработки

В проекте используется `poetry` для управления зависимостями. Чтобы ноутбуки использовали те же версии библиотек, что и основной код, следуйте этой инструкции.

## 1. Настройка ядра для Jupyter (один раз)

Чтобы ваше окружение из Poetry стало доступно в Jupyter (даже если сам Jupyter запущен из Conda или системы), выполните:

```bash
# Регистрируем ядро в системе
poetry run python -m ipykernel install --user --name ml-hackathon-env --display-name "Python (ML Hackathon)"
```

## 2. Запуск ноутбуков

1. Откройте любой интерфейс (VS Code, JupyterLab из Conda и т.д.).
2. Откройте нужный ноутбук в папке `notebooks/`.
3. В меню выбора ядра (**Kernel** -> **Change Kernel**) выберите **"Python (ML Hackathon)"**.

## 3. Возможные проблемы

- **Если библиотеки не импортируются:** Убедитесь, что в верхнем углу ноутбука выбрано именно ядро `Python (ML Hackathon)`, а не стандартное системное.


# Инструкция по работе с ветками для исследования

Чтобы проводить эксперименты в ветке `feature/baseline-research`, не мешая основной разработке в `feature/eda-baseline-analysis`, используйте следующие команды:

## 0. ```git status```
Всё время проверяем статус.

## 1. Создание новой ветки
Сначала обновите локальную копию исходной ветки и создайте от неё новую:

```bash
# Переключаемся на ветку, которую берем за основу
git switch feature/eda-baseline-analysis

# Подтягиваем последние изменения из репозитория
git pull origin feature/eda-baseline-analysis

# Создаем новую ветку для исследования и сразу переходим в неё (-c = create)
git switch -c feature/baseline-research
```

## 2. Работа и сохранение результатов
После внесения изменений зафиксируйте их и отправьте на GitHub:

```bash
git add .
make precommit
git commit -m "feat: исследование работы бейзлайна"
git push origin feature/baseline-research
```

## 3. Слияние (Merge)
Когда исследование закончено и изменения одобрены:
1. Создайте **Pull Request** на GitHub из `feature/baseline-research` в `feature/eda-baseline-analysis`.
2. После проверки (review) нажмите **Merge pull request**.

## 4. Удаление временной ветки
Чтобы навести порядок после слияния:

```bash
# Возвращаемся в рабочую ветку
git switch feature/eda-baseline-analysis

# Обновляем её локально
git pull origin feature/eda-baseline-analysis

# Удаляем локальную ветку исследования
git branch -d feature/baseline-research

# Удаляем ветку в удаленном репозитории
git push origin --delete feature/baseline-research
```

> **Примечание:** Мы используем команду `git switch` вместо старой `git checkout`. Она была введена в Git специально для работы с ветками, чтобы избежать путаницы, так как `checkout` также используется для восстановления файлов. `switch` безопаснее и понятнее по смыслу.
