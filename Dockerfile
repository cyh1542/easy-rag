FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data ./data
COPY storage/.gitkeep ./storage/

RUN pip install --upgrade pip && pip install .

EXPOSE 5000 8000

# 默认启动 API 服务；Web 界面见 docker-compose.yml
CMD ["easy-rag-api"]
