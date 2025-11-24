# Multi-stage Dockerfile for HuggingFace Proxy Service
# Stage 1: Builder
FROM python:3.12-slim AS builder

# Install poetry
RUN pip install --no-cache-dir poetry==1.8.3

# Set working directory
WORKDIR /build

# Copy dependency files
COPY pyproject.toml ./

# Configure poetry to not create virtual env (we're in a container)
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-dev --no-root --no-interaction --no-ansi

# Stage 2: Runtime
FROM python:3.12-slim AS runtime

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser -u 1000 -m -s /sbin/nologin appuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ /app/src/

# Create cache directory and set permissions
RUN mkdir -p /data/hf-cache && \
    chown -R appuser:appuser /app /data

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_PROXY_PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz').read()" || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
