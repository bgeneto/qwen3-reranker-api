# Qwen3 Reranker API

A high-performance GPU-accelerated FastAPI service for document reranking using Qwen3-Reranker models. Provides an API compatible with Cohere and Jina reranking services.

## 🚀 Features

- **GPU-Accelerated** with CUDA 12.4 and Flash Attention
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
MAX_QUERY_LENGTH=4096
MAX_DOCUMENT_LENGTH=8192

# Logging
ENABLE_LOGGING=true
LOG_LEVEL=INFO
```

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

**Out of GPU Memory:**
```bash
# Reduce batch size in .env
BATCH_SIZE=16
```

**Container Won't Start:**
```bash
# Check logs
docker compose logs

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
```

**Slow Performance:**
```bash
# Check GPU utilization
nvidia-smi

# Adjust batch size and thread count in .env
BATCH_SIZE=24
TORCH_THREADS=2
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
- **Performance**: Flash Attention, uvloop, optimized PyTorch settings

## 📝 License

MIT License - see LICENSE file for details.
