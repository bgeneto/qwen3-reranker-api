# Qwen Reranker Service - Production Deployment

## Quick Start

```bash
# Make deployment script executable
chmod +x deploy.sh

# Start production environment
./deploy.sh prod

# Run tests
./deploy.sh test

# View logs
./deploy.sh logs

# Stop service
./deploy.sh stop
```

## Production Optimizations

### Why Uvicorn Only (No Gunicorn)?

✅ **Single GPU workload** - Multiple workers would duplicate model in memory
✅ **GPU-bound processing** - Not CPU-bound, so multiple workers don't help
✅ **Memory efficient** - One model instance vs multiple
✅ **Simpler deployment** - Easier debugging and monitoring

### Performance Features

- **uvloop** - High-performance event loop
- **httptools** - Fast HTTP parsing
- **Single worker** - Optimized for GPU workloads
- **Batch processing** - Prevents GPU OOM
- **Model caching** - Persistent model storage

### Production Features

- **Health checks** - Kubernetes/Docker Swarm ready
- **Metrics endpoint** - Performance monitoring
- **Structured logging** - JSON logs for analysis
- **Input validation** - Request size and content limits
- **Security hardening** - Non-root user, read-only filesystem
- **Resource limits** - Memory and GPU constraints

## Environment Variables

```bash
# Model Configuration
MODEL_NAME=Qwen/Qwen3-Reranker-0.6B

# API Limits
MAX_DOCUMENTS=100
MAX_QUERY_LENGTH=2048
MAX_DOCUMENT_LENGTH=4096

# Performance
BATCH_SIZE=16
UVICORN_WORKERS=1
UVICORN_LOOP=uvloop
UVICORN_HTTP=httptools

# Logging
ENABLE_LOGGING=true
LOG_METHOD=async
LOG_FILE_PATH=/app/logs/requests.log
```

## API Endpoints

- `GET /` - Basic health check
- `GET /health` - Detailed health with system info
- `GET /metrics` - Performance metrics
- `POST /rerank` - Document reranking

## Monitoring

```bash
# Check health
curl http://localhost:8004/health

# Get metrics
curl http://localhost:8004/metrics

# Test reranking
curl -X POST http://localhost:8004/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "documents": ["Paris is the capital of France.", "London is a city."],
    "top_k": 1
  }'
```

## Scaling

For high-traffic scenarios:

1. **Horizontal scaling** - Multiple containers behind load balancer
2. **GPU scaling** - One container per GPU
3. **Model variants** - Different model sizes for different use cases

```yaml
# Example for multiple GPUs
services:
  reranker-gpu-0:
    extends: qwen3-reranker
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - device_ids: ["0"]

  reranker-gpu-1:
    extends: qwen3-reranker
    environment:
      - CUDA_VISIBLE_DEVICES=1
    ports:
      - "8005:8000"
    deploy:
      resources:
        reservations:
          devices:
            - device_ids: ["1"]
```

## Troubleshooting

### Common Issues

1. **GPU not detected**
   ```bash
   # Check GPU availability
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
   ```

2. **Model loading fails**
   ```bash
   # Check disk space and memory
   df -h
   free -h

   # Check container logs
   docker logs qwen3-reranker-prod
   ```

3. **High memory usage**
   ```bash
   # Monitor container resources
   docker stats qwen3-reranker-prod

   # Reduce batch size in .env
   BATCH_SIZE=8
   ```

### Performance Tuning

1. **Batch size** - Adjust based on GPU memory
2. **Max documents** - Limit request size
3. **Cache directory** - Use SSD for model cache
4. **Memory limits** - Set appropriate container limits
