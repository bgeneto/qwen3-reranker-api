# Qwen3 Reranker API Service

A production-ready FastAPI service for document reranking using Qwen3-Reranker models. This service provides an endpoint to rerank documents based on query relevance, returning results in a format compatible with Cohere and Jina APIs.

## 🚀 Features

- **Production-ready deployment** with Docker and Docker Compose
- **GPU acceleration** with CUDA and Flash Attention support
- **High performance** with uvloop and httptools optimizations
- **Comprehensive monitoring** with health checks and metrics
- **Robust logging** with configurable async logging
- **Input validation** with safety limits and error handling
- **Security hardening** with non-root containers and read-only filesystem

## 📋 Requirements

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (optional but recommended)
- Python 3.11+ (for local development)

## � Quick Start

### Production Deployment

```bash
# Make deployment script executable
chmod +x deploy.sh

# Copy and configure environment
cp config.env.example .env
# Edit .env with your settings

# Start production environment (GPU with flash_attn)
./deploy.sh prod

# OR start CPU-only environment (without flash_attn)
./deploy.sh cpu

# Test the API
./deploy.sh test

# View logs
./deploy.sh logs
```

### Development

```bash
# Start development environment
./deploy.sh dev

# Stop services
./deploy.sh stop
```

## 🔧 Deployment Options

### GPU Production (Recommended)
- Uses `./deploy.sh prod`
- Includes flash_attn for faster inference
- Requires NVIDIA GPU with CUDA support
- Uses `nvidia/cuda:12.1-devel-ubuntu20.04` base image
- Higher memory usage but fastest performance

### CPU-Only Production
- Uses `./deploy.sh cpu`
- No flash_attn dependency
- Works on any system
- Uses `python:3.11-slim` base image
- Lower memory usage but slower inference

### Manual Deployment
```bash
# Copy and configure environment
cp config.env.example .env
# Edit .env with your settings

# Deploy with production configuration
docker compose -f compose.prod.yaml up -d

# For CPU-only deployment (without flash_attn)
# Edit Dockerfile.prod to use requirements-prod-cpu.txt instead
```

## 📊 API Endpoints

### Health & Monitoring
- `GET /` - Basic health check
- `GET /health` - Detailed health information
- `GET /metrics` - Performance metrics

### Core Functionality
- `POST /rerank` - Document reranking

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen3-Reranker-0.6B` | HuggingFace model identifier |
| `MAX_DOCUMENTS` | `100` | Maximum documents per request |
| `MAX_QUERY_LENGTH` | `4096` | Maximum query length |
| `MAX_DOCUMENT_LENGTH` | `8192` | Maximum document length |
| `ENABLE_LOGGING` | `false` | Enable request logging |
| `LOG_METHOD` | `async` | Logging method (file/stdout/async) |
| `BATCH_SIZE` | `16` | Model inference batch size |

### Model Options
- `Qwen/Qwen3-Reranker-0.6B` (fastest, least accurate)
- `Qwen/Qwen3-Reranker-4B` (balanced)
- `Qwen/Qwen3-Reranker-8B` (best quality, requires more VRAM)

## 🧪 Testing

```bash
# Run basic tests
python test_api.py

# Test with custom query
python test_api.py --query "machine learning" --documents "AI is important" "Python is popular"

# Test different endpoint
python test_api.py --url http://your-server:8004
```

## 📈 Production Considerations

### Security
- [ ] Configure CORS properly (`CORS_ORIGINS`)
- [ ] Add API key authentication
- [ ] Use HTTPS in production
- [ ] Enable rate limiting
- [ ] Regular security updates

### Performance
- [ ] Monitor memory usage (models are large)
- [ ] Adjust `BATCH_SIZE` based on GPU memory
- [ ] Consider model sharding for larger models
- [ ] Set up load balancing for high traffic

### Monitoring
- [ ] Set up log aggregation
- [ ] Monitor `/metrics` endpoint
- [ ] Set up alerts for failures
- [ ] Track response times

### Scaling
- [ ] Use multiple GPU devices
- [ ] Horizontal scaling with load balancer
- [ ] Consider model caching strategies
- [ ] Database for request/response logging

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `BATCH_SIZE`
   - Use smaller model variant
   - Check GPU memory usage

2. **Slow Response Times**
   - Enable Flash Attention (`pip install flash-attn`)
   - Optimize batch size
   - Check GPU utilization

3. **Model Loading Fails**
   - Check HuggingFace cache permissions
   - Verify model name
   - Check internet connection for first download

4. **Flash Attention Installation Fails**
   - Error: `ModuleNotFoundError: No module named 'packaging'`
   - Solution: Build dependencies are now included in requirements
   - Alternative: Remove `flash_attn` from requirements for CPU-only deployment
   - Note: Flash Attention requires NVCC and CUDA toolkit for GPU builds

### Logs
```bash
# View container logs
docker logs qwen3-reranker-prod

# View application logs
tail -f logs/requests.log
```

## 📦 Docker Commands

```bash
# Build production image
docker build -f Dockerfile.prod -t qwen3-reranker:prod .

# Run with custom config
docker run -d \
  --name qwen3-reranker \
  --gpus device=1 \
  -p 8004:8000 \
  -v ./logs:/app/logs \
  -v ./hf_cache:/app/cache \
  --env-file .env \
  qwen3-reranker:prod

# Check container health
docker inspect qwen3-reranker --format='{{.State.Health.Status}}'
```
