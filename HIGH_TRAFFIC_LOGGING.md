# High-Traffic Logging Configuration

## Overview

The Provence Pruner now supports multiple logging methods optimized for different traffic scenarios and deployment environments.

## Logging Methods

### 1. Asynchronous Logging (`LOG_METHOD=async`) - **Recommended for High Traffic**

**Best for:** High-traffic production environments with hundreds/thousands of requests per second.

**How it works:**
- Uses Python's `QueueHandler` and `QueueListener`
- Log messages are queued in memory and written by a separate background thread
- Non-blocking: API requests never wait for file I/O
- Thread-safe and highly performant

**Configuration:**
```yaml
environment:
  - LOG_METHOD=async
  - ENABLE_LOGGING=true
  - LOG_FILE_PATH=/app/logs/pruner_requests.log
```

**Pros:**
- ✅ No file locking contention
- ✅ Non-blocking API performance
- ✅ Built-in thread safety
- ✅ Automatic buffering and batching

**Cons:**
- ⚠️ Small memory overhead for queue
- ⚠️ Potential log loss if application crashes (rare)

### 2. Stdout Logging (`LOG_METHOD=stdout`) - **Best for Containers**

**Best for:** Containerized environments with external log aggregation (ELK, Fluentd, etc.)

**How it works:**
- Logs only to stdout/stderr
- No file I/O operations
- Perfect for Docker/Kubernetes with log collectors

**Configuration:**
```yaml
environment:
  - LOG_METHOD=stdout
  - ENABLE_LOGGING=true
```

**Pros:**
- ✅ Zero file I/O overhead
- ✅ Perfect for cloud-native deployments
- ✅ Works with log aggregation systems
- ✅ No disk space concerns

**Cons:**
- ⚠️ Requires external log collection
- ⚠️ Logs lost if not collected

### 3. Traditional File Logging (`LOG_METHOD=file`) - **Default**

**Best for:** Low to medium traffic, development, or when you need local file logs.

**How it works:**
- Standard Python file logging with optimizations
- Built-in buffering to reduce I/O calls
- Thread-safe but may have minor blocking

**Configuration:**
```yaml
environment:
  - LOG_METHOD=file
  - ENABLE_LOGGING=true
  - LOG_FILE_PATH=/app/logs/pruner_requests.log
```

**Pros:**
- ✅ Simple and reliable
- ✅ Local file access
- ✅ Good for debugging

**Cons:**
- ⚠️ File locking with very high traffic
- ⚠️ Disk I/O overhead

## Performance Comparison

| Method | Requests/sec | Latency Impact | Memory Usage | Disk I/O |
|--------|-------------|----------------|--------------|----------|
| `async` | 1000+ | Minimal | Low | Batched |
| `stdout` | 1000+ | Minimal | Minimal | None |
| `file` | 100-500 | Low | Minimal | Direct |

## Recommendations by Use Case

### High-Traffic Production (>500 req/s)
```yaml
environment:
  - LOG_METHOD=async
  - ENABLE_LOGGING=true
  - LOG_FILE_PATH=/app/logs/pruner_requests.log
```

### Cloud/Kubernetes Deployment
```yaml
environment:
  - LOG_METHOD=stdout
  - ENABLE_LOGGING=true
```

### Development/Testing
```yaml
environment:
  - LOG_METHOD=file
  - ENABLE_LOGGING=true
  - LOG_FILE_PATH=/app/logs/pruner_requests.log
```

### Disable Logging (Maximum Performance)
```yaml
environment:
  - ENABLE_LOGGING=false
```

## Thread Safety

All logging methods are thread-safe and designed to never block the main API execution. The application uses:

- `safe_log()` function that wraps all logging calls
- Try-catch blocks around all logging operations
- Fallback mechanisms if logging fails
- Proper resource cleanup on application shutdown

## Monitoring

You can monitor logging performance by watching:

```bash
# Check log file growth (for file/async methods)
watch -n 1 'ls -lah /app/logs/'

# Monitor stdout logs (for stdout method)
docker logs -f provence-pruner

# Check for logging warnings in application output
docker logs provence-pruner | grep "Warning: Logging failed"
```

## Troubleshooting

### High Memory Usage with Async Logging
- The queue is unlimited by default
- For extreme high traffic, consider `stdout` method instead

### File Permission Issues
- Ensure the logs directory is writable by the application user
- Check Docker volume permissions

### Missing Logs
- Verify `ENABLE_LOGGING=true`
- Check for error messages in application startup logs
- Ensure log directory exists and is writable

### Performance Issues
- Switch from `file` to `async` for high traffic
- Consider `stdout` method for containerized deployments
- Monitor disk I/O and available space
