# Multi-stage build for production optimization
# Stage 1: Install Python dependencies using lightweight Python image
FROM python:3.12-slim AS builder

# Set build environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

WORKDIR /build

# Create virtual environment for better isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 && \
    /opt/venv/bin/python -m pip install --no-cache-dir -r requirements.txt && \
    /opt/venv/bin/python -m pip install --no-cache-dir uvicorn[standard]

# Stage 2: Production runtime
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04 AS production

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Set production environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    HF_HOME=/app/cache \
    TRANSFORMERS_CACHE=/app/cache \
    CUDA_HOME=/usr/local/cuda \
    PATH="/opt/venv/bin:/usr/local/cuda/bin:$PATH" \
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
COPY --from=builder /opt/venv /opt/venv

# Ensure the virtual environment is properly activated
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# Copy application code
COPY main.py .

# Verify uvicorn installation and create symlink if needed
RUN python3 -m pip list | grep uvicorn || \
    python3 -m pip install uvicorn[standard]

# Set proper permissions for cache and logs
RUN chown -R ${UID}:${GID} /app

# Switch to non-privileged user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Override NVIDIA entrypoint to run uvicorn with the Python interpreter directly
ENTRYPOINT ["/opt/venv/bin/python","-m","uvicorn"]
# Production command optimized for performance
CMD ["main:app","--host","0.0.0.0","--port","8000","--workers","1","--log-level","info","--access-log","--loop","uvloop","--http","httptools","--no-server-header","--date-header"]
