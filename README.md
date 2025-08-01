# Qwen3 Reranker API

A high-performance GPU-accelerated FastAPI service for document reranking using Qwen3-Reranker models. Provides an API compatible with Cohere and Jina reranking services.

## 🚀 Features

- **GPU-Accelerated** with CUDA 12.6 optimization
- **Docker Containerized** for easy deployment
- **High Performance** with uvloop, httptools, and optimized PyTorch
- **API Compatible** with Cohere and Jina reranking formats
- **Secure** with non-root containers and read-only filesystem

## 📋 Requirements

- Docker Engine 24.0+ with BuildKit
- Docker Compose 2.20+
- NVIDIA Container Toolkit
- NVIDIA GPU with 12GB+ VRAM
- CUDA 12.6 compatible drivers

## 🔧 Quick Start

### 1. Setup
```bash
git clone <repository-url>
cd qwen3-reranker-api

# Copy and edit configuration
cp config.env.example .env
nano .env  # Edit as needed
```

### 2. Deploy
```bash
# Build and start the service
docker compose build
docker compose up -d

# Check status
docker compose ps
```

### 3. Verify
```bash
# Health check
curl http://localhost:8004/health

# Test API
curl -X POST http://localhost:8004/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "documents": [
      "Machine learning is a subset of AI",
      "Python is a programming language",
      "Neural networks are used in deep learning"
    ],
    "top_n": 2
  }'
```

## ⚙️ Configuration

Key settings in `.env`:

```bash
# Model Configuration
MODEL_NAME=Qwen/Qwen3-Reranker-0.6B

# GPU Performance Settings
BATCH_SIZE=32                    # Adjust based on GPU memory
TORCH_THREADS=4
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# API Limits
MAX_DOCUMENTS=100

# Uvicorn Performance Settings
UVICORN_WORKERS=1               # Keep at 1 for GPU models
UVICORN_MAX_CONNECTIONS=2000    # Max concurrent connections
UVICORN_LIMIT_CONCURRENCY=1000  # Max concurrent requests
UVICORN_BACKLOG=2048           # Connection backlog
UVICORN_TIMEOUT_KEEP_ALIVE=5   # Keep-alive timeout (seconds)

# Logging
ENABLE_LOGGING=true
LOG_LEVEL=INFO
```

### Uvicorn Performance Tuning

For high-load scenarios, adjust these settings in your `.env`:

```bash
# High-load configuration
UVICORN_MAX_CONNECTIONS=4000    # Total connections server accepts
UVICORN_LIMIT_CONCURRENCY=2000  # Max concurrent request processing
UVICORN_BACKLOG=4096           # Queue size for pending connections
UVICORN_TIMEOUT_KEEP_ALIVE=30  # Connection keep-alive time

# Memory-constrained configuration
UVICORN_MAX_CONNECTIONS=1000
UVICORN_LIMIT_CONCURRENCY=500
UVICORN_BACKLOG=1024
BATCH_SIZE=16
```

**Key Parameters:**
- `UVICORN_WORKERS`: Always keep at 1 for GPU models
- `UVICORN_MAX_CONNECTIONS`: Total TCP connections the server accepts
- `UVICORN_LIMIT_CONCURRENCY`: Maximum requests processed simultaneously
- `UVICORN_BACKLOG`: Connection queue size when at max connections
- `UVICORN_TIMEOUT_KEEP_ALIVE`: How long to keep idle connections open

## 📚 API Usage

### Health Check
```bash
GET /health
```

### Document Reranking
```bash
POST /rerank
Content-Type: application/json

{
  "query": "What is machine learning?",
  "documents": [
    "Machine learning is a subset of artificial intelligence.",
    "Python is a programming language.",
    "Deep learning uses neural networks with multiple layers."
  ],
  "top_n": 2
}
```

### Response Format
```json
{
  "results": [
    {
      "index": 0,
      "relevance_score": 0.9856,
      "document": "Machine learning is a subset of artificial intelligence."
    },
    {
      "index": 2,
      "relevance_score": 0.8742,
      "document": "Deep learning uses neural networks with multiple layers."
    }
  ]
}
```

## 🔍 Management

### View Logs
```bash
docker compose logs -f
```

### Monitor Resources
```bash
docker stats qwen3-reranker-api
nvidia-smi  # GPU monitoring
```

### Update
```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

### Cleanup
```bash
docker compose down --volumes
docker system prune -f
```

## 🛠️ Troubleshooting

### Common Issues

**429 Too Many Requests:**
```bash
# Increase connection limits in .env
UVICORN_MAX_CONNECTIONS=4000
UVICORN_LIMIT_CONCURRENCY=2000
UVICORN_BACKLOG=4096

# Then restart
docker compose restart
```

**Out of GPU Memory:**
```bash
# Keep workers at 1 for GPU models (don't increase UVICORN_WORKERS)
UVICORN_WORKERS=1
# Reduce batch size instead
BATCH_SIZE=16
```

**Connection Timeouts:**
```bash
# Increase keep-alive timeout
UVICORN_TIMEOUT_KEEP_ALIVE=30

# Increase backlog for high load
UVICORN_BACKLOG=4096
```

**Container Won't Start:**
```bash
# Check logs
docker compose logs

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.6.3-runtime-ubuntu24.04 nvidia-smi
```

**Slow Performance:**
```bash
# Check GPU utilization
nvidia-smi

# Adjust batch size and thread count in .env
BATCH_SIZE=24
TORCH_THREADS=2

# Monitor concurrent connections
docker compose logs | grep -i "connection"
```

## 📈 Performance

| GPU Model | Batch Size | Throughput | Memory Usage |
|-----------|------------|------------|--------------|
| RTX 4090  | 32         | ~1000/sec  | ~6GB VRAM   |
| RTX 3080  | 16         | ~600/sec   | ~8GB VRAM   |

## 🏗️ Architecture

The service uses a multi-stage Docker build with:
- **Builder Stage**: CUDA development environment for compiling dependencies
- **Runtime Stage**: Minimal CUDA runtime for optimized deployment
- **Security**: Non-root user, read-only filesystem, network isolation
- **Performance**: uvloop, optimized PyTorch settings

## 📝 License

MIT License - see LICENSE file for details.
