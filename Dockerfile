FROM python:3.10-slim

# Prevent python buffering
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /app

# System deps (only needed ones)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
