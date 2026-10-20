FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv

COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir . && \
    useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin gateway

USER 10001
EXPOSE 8000
CMD ["uvicorn", "app.main:asgi", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
