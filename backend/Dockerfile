# Multi-stage Docker build for React Frontend + FastAPI Backend

# Stage 1: Build React frontend
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY frontend/package*.json ./
RUN npm ci --only=production

# Copy frontend source
COPY frontend/ ./

# Build frontend
RUN npm run build

# Stage 2: Python backend with GPU support
FROM nvidia/cuda:11.8-devel-ubuntu22.04 AS backend-base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

USER appuser
WORKDIR /app

# Install Python dependencies
COPY --chown=appuser:appuser requirements.txt ./
RUN pip3 install --user --no-cache-dir -r requirements.txt

# Copy backend source
COPY --chown=appuser:appuser backend/ ./backend/
COPY --chown=appuser:appuser *.py ./
COPY --chown=appuser:appuser config.json ./

# Copy built frontend
COPY --from=frontend-builder --chown=appuser:appuser /app/frontend/dist ./frontend/dist

# Create necessary directories
RUN mkdir -p ./backend/outputs ./backend/logs ./backend/temp ./models ./loras

# Stage 3: Production image
FROM backend-base AS production

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV NODE_ENV=production
ENV CUDA_VISIBLE_DEVICES=0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose port
EXPOSE 8000

# Start command
CMD ["python3", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 4: Development image with additional tools
FROM backend-base AS development

# Install development dependencies
RUN pip3 install --user --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    flake8 \
    mypy

# Install Node.js for frontend development
USER root
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

USER appuser

# Copy frontend source for development
COPY --chown=appuser:appuser frontend/ ./frontend/

# Install frontend dependencies
WORKDIR /app/frontend
RUN npm install

WORKDIR /app

# Development command with hot reload
CMD ["python3", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]