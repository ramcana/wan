# Dockerfile for Wan2.2 UI Variant with Local Testing Framework
# Multi-stage build for optimized production image

# Build stage
FROM python:3.9-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA Container Toolkit (for GPU support)
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    && apt-get update

# Create non-root user
RUN useradd --create-home --shell /bin/bash wan22

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/wan22/.local

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/outputs /app/cache /app/logs /app/reports \
    && chown -R wan22:wan22 /app

# Copy configuration files
COPY local_testing_framework/examples/configurations/production_config.json /app/config.json
COPY local_testing_framework/examples/environment_templates/production.env /app/.env.template

# Install local testing framework
RUN pip install -e .

# Switch to non-root user
USER wan22

# Set environment variables
ENV PATH="/home/wan22/.local/bin:$PATH"
ENV PYTHONPATH="/app:/app/local_testing_framework"
ENV HF_HOME="/app/cache/huggingface"
ENV TRANSFORMERS_CACHE="/app/cache/transformers"
ENV TORCH_HOME="/app/cache/torch"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -m local_testing_framework validate-env --quick || exit 1

# Expose ports
EXPOSE 8080 8081 9090

# Default command
CMD ["python", "main.py", "--config", "/app/config.json", "--port", "8080"]