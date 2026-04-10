FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY asne/ asne/

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "asne.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
