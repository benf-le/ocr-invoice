# Sử dụng Python 3.11 slim image cho Linux (Debian-based)
FROM python:3.11-slim

# Thiết lập biến môi trường để tối ưu pip và Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Thiết lập working directory
WORKDIR /app

# Cài đặt các dependencies hệ thống cần thiết cho OpenCV và PaddlePaddle
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements.txt trước để tận dụng Docker cache layer
# Layer này sẽ được cache nếu requirements.txt không thay đổi
COPY requirements.txt .

# Cài đặt Python dependencies với cache
# Sử dụng --no-cache-dir để giảm kích thước image
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code vào container
# Layer này sẽ rebuild khi code thay đổi, nhưng dependencies đã được cache
COPY . .

# Tạo non-root user để chạy ứng dụng (best practice cho security)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Chuyển sang non-root user
USER appuser

# Expose port 8000
EXPOSE 8000

# Health check để đảm bảo container đang chạy tốt
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Chạy ứng dụng FastAPI với uvicorn trên port 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]