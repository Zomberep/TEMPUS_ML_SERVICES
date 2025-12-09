# Сборка образа
FROM python:3.12-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Установка Python-зависимостей
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Копируем код и модель
COPY . .
COPY rubert-tiny2/ /app/rubert-tiny2/

# Запуск
CMD ["python", "-u", "app.py"]