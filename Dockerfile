# Multi-stage build for production optimization
# Stage 1: Build dependencies (needs CUDA for flash-attn compilation)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

# Install Python and minimal build dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    gcc \
    g++ \
    ninja-build \
    cmake \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Set build environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

WORKDIR /build

# Create virtual environment for better isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies with pre-built flash-attn
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/python -m pip install --no-cache-dir packaging ninja && \
    /opt/venv/bin/python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 && \
    echo "Installing flash-attn from pre-built wheel for CUDA 12.4..." && \
    /opt/venv/bin/python -m pip install --no-cache-dir \
        flash-attn --index-url https://download.pytorch.org/whl/cu124 || \
    /opt/venv/bin/python -m pip install --no-cache-dir \
        "flash-attn>=2.5.0" --find-links https://github.com/Dao-AILab/flash-attention/releases && \
    /opt/venv/bin/python -m pip install --no-cache-dir -r requirements.txt

# Stage 2: Production runtime
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS production

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Set production environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    HF_HOME=/app/cache \
    TRANSFORMERS_CACHE=/app/cache \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

WORKDIR /app

# Create cache and logs directories with proper permissions
RUN mkdir -p /app/cache /app/logs

# Create non-root user with specified UID and GID
ARG UID=1000
ARG GID=1000
RUN groupadd -f -g ${GID} appuser && \
    useradd -o -r -u ${UID} -g ${GID} -d /srv -s /bin/bash appuser 2>/dev/null || \
    echo "User/group with UID ${UID}/GID ${GID} already exists, continuing..."

# Copy Python packages from builder stage
# First, copy the entire Python installation to ensure all packages are included
COPY --from=builder /opt/venv /opt/venv

# Add virtual environment to PATH
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY main.py .

# Set proper permissions for cache and logs
RUN chown -R ${UID}:${GID} /app

# Switch to non-privileged user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command optimized for performance
CMD ["/opt/venv/bin/uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--access-log", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--no-server-header", \
     "--date-header"]
