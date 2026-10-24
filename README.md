# Классификатор тональности: окружение

Скачайте датасет [RuTweetCorp](https://www.kaggle.com/datasets/maximsuvorov/rutweetcorp)
и положите исходные файлы в `data/positive.csv` и `data/negative.csv`.
Папка `data/` исключена из Git: проверяющий скачивает CSV отдельно.
Эта папка подключается к контейнеру вместе с проектом и доступна в Jupyter
по пути `/workspace/data/`. Данные не копируются в Docker-образ.

1. Запустите Docker Desktop.
2. В каталоге проекта выполните `docker compose up --build -d`.
3. Выполните `docker compose logs notebook` и откройте ссылку на
   `http://127.0.0.1:8888/lab?token=...` из вывода. Jupyter сохраняет
   файлы ноутбука прямо в каталоге проекта.
4. По окончании работы выполните `docker compose down`.

При изменении `requirements.txt` повторите команду сборки из шага 2.
