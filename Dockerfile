FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.py .
COPY train_optimized.py .
COPY predict.py .
COPY scripts/ ./scripts/
COPY models/optimized/final ./models/optimized/final

RUN mkdir -p logs data

CMD ["python", "model.py", "--help"]
