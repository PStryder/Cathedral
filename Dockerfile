# Cathedral Dockerfile
# Multi-stage build for optimized production image

# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Optional: Install llama-cpp-python for local LLM support
# Uncomment if you need local summarization
# COPY requirements-llama.txt .
# RUN pip install --no-cache-dir -r requirements-llama.txt

# ============================================
# Stage 2: Runtime
# ============================================
FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash cathedral

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=cathedral:cathedral altar/ ./altar/
COPY --chown=cathedral:cathedral cathedral/ ./cathedral/
COPY --chown=cathedral:cathedral tests/ ./tests/
COPY --chown=cathedral:cathedral pytest.ini ./

# Create data directories
RUN mkdir -p /app/data/config \
    /app/data/personalities \
    /app/data/scripture \
    /app/data/agents \
    /app/data/backups \
    /app/data/shell_history \
    /app/models/memory \
    && chown -R cathedral:cathedral /app/data /app/models

# Switch to non-root user
USER cathedral

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    DATA_DIR=/app/data \
    MODELS_DIR=/app/models

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health/summary || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "altar.run"]
