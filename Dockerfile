# ===============================
# BASE IMAGE
# ===============================
FROM python:3.10-slim

# Prevent Python buffering
ENV PYTHONUNBUFFERED=1

# ===============================
# WORKDIR
# ===============================
WORKDIR /app

# ===============================
# SYSTEM DEPENDENCIES
# Required for:
# - PaddleOCR
# - OpenCV
# - Playwright
# - Tesseract
# ===============================
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    tesseract-ocr \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# COPY PROJECT FILES
# ===============================
COPY . /app

# ===============================
# INSTALL PYTHON DEPENDENCIES
# ===============================
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ===============================
# PLAYWRIGHT SETUP
# IMPORTANT FOR YOUR RAG FETCH
# ===============================
RUN playwright install chromium

# ===============================
# EXPOSE PORT (HF uses 7860)
# ===============================
EXPOSE 7860

# ===============================
# START FASTAPI SERVER
# ===============================
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

