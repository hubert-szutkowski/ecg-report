FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY fastapi_ecg_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY fastapi_ecg_service/ ./app_service/

EXPOSE 8000

CMD ["uvicorn", "app_service.main:app", "--host", "0.0.0.0", "--port", "8000"]