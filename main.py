"""
Qwen3 Reranker API Service

A FastAPI-based service for document reranking using the Qwen/Qwen3-Reranker family of models.
This service provides an endpoint to rerank documents based on query relevance, returning results
in a format compatible with Cohere and Jina APIs. The API follows Cohere rerank API standards.

Author: Bernhard Enders (bgeneto)
Adapted By: Gemini/Claude
Created: July 11, 2025
Updated: August 1, 2025
Version: 1.2.0
License: MIT
Python: 3.8+

Dependencies:
    - fastapi
    - pydantic
    - torch
    - transformers
    - numpy

Environment Variables:
    MODEL_NAME: HuggingFace model identifier (default: Qwen/Qwen3-Reranker-4B)
    ENABLE_LOGGING: Enable request/response logging (default: false)
    LOG_FILE_PATH: Path to log file (default: reranker_requests.log)
    LOG_METHOD: Logging method - file, stdout, or async (default: file)

API Endpoints:
    GET /: Health check endpoint
    POST /rerank: Document reranking endpoint (Cohere API compatible)

Example Usage (Cohere API Compatible):
    POST /rerank
    {
        "model": "rerank-v3.5",
        "query": "What is the capital of France?",
        "documents": ["Paris is the capital of France.", "London is a large city.", "The Eiffel Tower is in Paris."],
        "top_n": 2
    }

Response Format:
    {
        "results": [
            {"index": 0, "relevance_score": 0.9856},
            {"index": 2, "relevance_score": 0.8742}
        ]
    }
"""

import json
import logging
import os
import queue
import sys
import threading
import time
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener
from typing import List, Dict, Any, Optional

# Performance optimizations
try:
    import uvloop

    uvloop.install()  # Use uvloop as the default event loop
except ImportError:
    pass  # Fall back to default asyncio event loop

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from pydantic import BaseModel, Field, field_validator, ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Environment Configuration ---
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-Reranker-4B")
ENABLE_LOGGING = os.getenv("ENABLE_LOGGING", "false").lower() == "true"
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "reranker_requests.log")
LOG_METHOD = os.getenv("LOG_METHOD", "async").lower()  # file, stdout, or async

# API Security Configuration
API_KEY = os.getenv("API_KEY", None)  # Set this in production!
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "true").lower() == "true"

# Production safety limits
MAX_DOCUMENTS = int(os.getenv("MAX_DOCUMENTS", "100"))
# MAX_QUERY_LENGTH and MAX_DOCUMENT_LENGTH removed - let tokenizer handle text truncation based on model's context window
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))
BATCH_SIZE = int(
    os.getenv("BATCH_SIZE", "2")
)  # Default to conservative batch size for memory safety

# --- Global metrics storage ---
METRICS = {
    "requests_total": 0,
    "requests_successful": 0,
    "requests_failed": 0,
    "total_processing_time": 0.0,
    "total_documents_processed": 0,
    "model_load_time": 0.0,
    "startup_time": time.time(),
}

# --- Logging Configuration ---
logger = None
log_queue = None
queue_listener = None

if ENABLE_LOGGING:
    try:
        log_dir = (
            os.path.dirname(LOG_FILE_PATH) if os.path.dirname(LOG_FILE_PATH) else "."
        )
        os.makedirs(log_dir, exist_ok=True)

        logger = logging.getLogger("reranker_api")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        if LOG_METHOD == "async":
            log_queue = queue.Queue(-1)
            file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
            file_handler.setFormatter(formatter)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            queue_listener = QueueListener(
                log_queue, file_handler, console_handler, respect_handler_level=True
            )
            queue_listener.start()
            logger.addHandler(QueueHandler(log_queue))
            logger.info("Async logging enabled. Log file: %s", LOG_FILE_PATH)

        elif LOG_METHOD == "stdout":
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            logger.info("Stdout logging enabled")

        else:  # default "file" method
            file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            logger.info("File logging enabled. Log file: %s", LOG_FILE_PATH)

    except Exception as e:
        print(f"Warning: Failed to configure logging: {e}")
        logger = None
else:
    logger = logging.getLogger("reranker_api")
    logger.addHandler(logging.NullHandler())


def safe_log(log_func, message, *args, **kwargs):
    """Thread-safe logging function that never fails."""
    if logger and ENABLE_LOGGING:
        try:
            log_func(message, *args, **kwargs)
        except Exception as e:
            print(f"Warning: Logging failed: {e}")


def log_gpu_memory(stage: str = ""):
    """Log current GPU memory usage for debugging."""
    if DEVICE == "cuda" and torch.cuda.is_available():
        try:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            max_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            safe_log(
                logger.info,
                f"GPU Memory {stage}: Allocated={memory_allocated:.2f}GB, Reserved={memory_reserved:.2f}GB, Max={max_memory:.2f}GB",
            )
        except Exception as e:
            safe_log(logger.warning, f"Failed to get GPU memory info: {e}")


def cleanup_logging():
    """Clean up logging resources."""
    global queue_listener
    if queue_listener:
        queue_listener.stop()


# --- Model Loading ---
model_load_start = time.time()
print(f"Loading reranker model: {MODEL_NAME}...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Model specific constants
MAX_LENGTH = 8192
PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")

    # Load model with standard attention
    reranker_model = (
        AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE)
        .to(DEVICE)
        .eval()
    )

    # Pre-tokenize prefixes and suffixes for efficiency
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)

    print(f"Reranker model loaded on {DEVICE}.")
    log_gpu_memory("after model load")
    # Update metrics
    METRICS["model_load_time"] = time.time() - model_load_start

except Exception as e:
    print(f"Fatal: Failed to load model or tokenizer: {e}")
    sys.exit(1)


# --- Reranker Helper Functions ---
def format_instruction(query: str, doc: str, instruction: Optional[str] = None) -> str:
    if instruction is None:
        # instruction = "Evaluate how relevant the following document is to the query for retrieving useful information to answer or provide context for the query"
        instruction = "Given a web search query, retrieve relevant passages that answer the query"  # using the default instruction from the Qwen3-Reranker model achieved best results
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def process_inputs(pairs: List[str]) -> Dict[str, torch.Tensor]:
    inputs = tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=MAX_LENGTH - len(prefix_tokens) - len(suffix_tokens),
    )
    for i in range(len(inputs["input_ids"])):
        inputs["input_ids"][i] = prefix_tokens + inputs["input_ids"][i] + suffix_tokens

    inputs = tokenizer.pad(
        inputs, padding=True, return_tensors="pt", max_length=MAX_LENGTH
    )
    for key in inputs:
        inputs[key] = inputs[key].to(reranker_model.device)
    return inputs


@torch.no_grad()
def compute_scores(inputs: Dict[str, torch.Tensor]) -> List[float]:
    try:
        # Enable memory efficient attention if available
        try:
            with torch.backends.cuda.sdp_kernel(
                enable_math=False, enable_flash=True, enable_mem_efficient=True
            ):
                logits = reranker_model(**inputs).logits[:, -1, :]
        except:
            # Fallback to standard attention if scaled_dot_product_attention is not available
            logits = reranker_model(**inputs).logits[:, -1, :]

        true_scores = logits[:, token_true_id]
        false_scores = logits[:, token_false_id]

        scores = torch.stack([false_scores, true_scores], dim=1)
        scores = torch.nn.functional.log_softmax(scores, dim=1)

        # Return the probability of "true" and move to CPU to free GPU memory
        result = scores[:, 1].exp().cpu().tolist()

        # Explicitly delete tensors and clear cache
        del logits, true_scores, false_scores, scores, inputs
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        return result

    except torch.cuda.OutOfMemoryError as e:
        # Clear cache and re-raise as RuntimeError for adaptive batching
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        raise RuntimeError(
            f"CUDA OOM in compute_scores. Current batch size caused OOM."
        ) from e


def process_with_adaptive_batching(
    pairs: List[str], initial_batch_size: int = None
) -> List[float]:
    """Process pairs with adaptive batch size reduction on OOM."""
    if initial_batch_size is None:
        initial_batch_size = BATCH_SIZE

    current_batch_size = initial_batch_size
    all_scores = []

    i = 0
    while i < len(pairs):
        batch_pairs = pairs[i : i + current_batch_size]

        try:
            inputs = process_inputs(batch_pairs)
            scores = compute_scores(inputs)
            all_scores.extend(scores)
            i += current_batch_size

            # Clear CUDA cache after each successful batch
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "OOM" in str(e):
                # Clear cache and reduce batch size
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

                current_batch_size = max(1, current_batch_size // 2)
                safe_log(
                    logger.warning,
                    f"CUDA OOM detected. Reducing batch size to {current_batch_size}",
                )

                if current_batch_size == 0:
                    raise RuntimeError(
                        "Cannot process even single document due to memory constraints"
                    )
                # Don't increment i, retry with smaller batch
            else:
                raise e

    return all_scores


def get_gpu_memory_info():
    """Get current GPU memory usage information."""
    if DEVICE == "cuda" and torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        memory_free = (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_reserved()
        ) / 1024**3
        return {
            "allocated_gb": round(memory_allocated, 2),
            "reserved_gb": round(memory_reserved, 2),
            "free_gb": round(memory_free, 2),
        }
    return {"allocated_gb": 0, "reserved_gb": 0, "free_gb": 0}


# --- API Security Functions ---
# Security schemes
security_bearer = HTTPBearer(auto_error=False)
security_api_key = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(
    bearer_token: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
    api_key_header: Optional[str] = Security(security_api_key),
) -> bool:
    """
    Verify API key from either Bearer token or X-API-Key header.
    Returns True if authentication is successful or not required.
    """
    if not REQUIRE_API_KEY:
        return True

    if not API_KEY:
        # If no API key is configured but required, warn and allow access
        print("Warning: REQUIRE_API_KEY is True but no API_KEY is set!")
        return True

    # Check Bearer token first
    if bearer_token and bearer_token.credentials == API_KEY:
        return True

    # Check X-API-Key header
    if api_key_header and api_key_header == API_KEY:
        return True

    # Authentication failed
    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API key. Use 'Authorization: Bearer <key>' or 'X-API-Key: <key>' header",
        headers={"WWW-Authenticate": "Bearer"},
    )


# --- API Definition ---
app = FastAPI(
    title="Qwen3 Reranker Service",
    description="Document reranking service using Qwen3-Reranker models",
    version="1.1.0",
)

# Add CORS middleware for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Debug middleware to log all requests
@app.middleware("http")
async def debug_request_middleware(request: Request, call_next):
    """Log all incoming requests for debugging 422 errors."""
    if ENABLE_LOGGING and request.url.path in ["/rerank", "/v1/rerank"]:
        try:
            # Store the body for later access (FastAPI consumes it)
            body = await request.body()

            # Create a new request with the same body for the endpoint
            async def receive():
                return {"type": "http.request", "body": body}

            request._receive = receive

        except Exception as e:
            safe_log(logger.error, "Error in debug middleware: %s", str(e))

    response = await call_next(request)
    return response


@app.on_event("shutdown")
def shutdown_event():
    cleanup_logging()


# --- Pydantic Models for API I/O ---
class RerankRequest(BaseModel):
    model: Optional[str] = Field(
        None, description="Model identifier (for API compatibility)"
    )
    query: str = Field(..., description="The search query")
    documents: List[str] = Field(
        ..., max_length=MAX_DOCUMENTS, description="List of documents to rerank"
    )
    top_n: Optional[int] = Field(
        None, ge=1, le=MAX_DOCUMENTS, description="Number of top results to return"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        # Remove length validation - let tokenizer handle truncation
        return v.strip()

    @field_validator("documents")
    @classmethod
    def validate_documents(cls, v):
        if not v:
            raise ValueError("Documents list cannot be empty")
        if len(v) > MAX_DOCUMENTS:
            raise ValueError(
                f"Too many documents. Maximum {MAX_DOCUMENTS} documents allowed, received {len(v)}"
            )

        # Remove document length validation - let tokenizer handle truncation
        for i, doc in enumerate(v):
            if not doc or not doc.strip():
                raise ValueError(f"Document at index {i} cannot be empty")

        return [doc.strip() for doc in v]

    @field_validator("top_n")
    @classmethod
    def validate_top_n(cls, v, info):
        if v is not None:
            documents = info.data.get("documents", [])
            if v > len(documents):
                raise ValueError("top_n cannot be greater than the number of documents")
        return v


class RerankResult(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    results: List[RerankResult]


# --- API Endpoints ---
@app.get("/", summary="Health Check")
def read_root(authenticated: bool = Depends(verify_api_key)):
    """A simple health check endpoint."""
    return {"status": "Reranker service API is running", "model": MODEL_NAME}


@app.get("/health", summary="Detailed Health Check")
def health_check():
    """Detailed health check with system information. This endpoint is unprotected for monitoring systems."""
    uptime = time.time() - METRICS["startup_time"]
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": DEVICE,
        "uptime_seconds": round(uptime, 2),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "authentication": {
            "required": REQUIRE_API_KEY,
            "configured": API_KEY is not None,
            "methods": (
                ["Bearer token", "X-API-Key header"] if REQUIRE_API_KEY else ["None"]
            ),
        },
    }


@app.get("/metrics", summary="Service Metrics")
def get_metrics(authenticated: bool = Depends(verify_api_key)):
    """Get service performance metrics."""
    uptime = time.time() - METRICS["startup_time"]
    avg_processing_time = (
        METRICS["total_processing_time"] / METRICS["requests_successful"]
        if METRICS["requests_successful"] > 0
        else 0
    )

    return {
        "uptime_seconds": round(uptime, 2),
        "requests_total": METRICS["requests_total"],
        "requests_successful": METRICS["requests_successful"],
        "requests_failed": METRICS["requests_failed"],
        "success_rate": (
            METRICS["requests_successful"] / METRICS["requests_total"]
            if METRICS["requests_total"] > 0
            else 0
        ),
        "total_documents_processed": METRICS["total_documents_processed"],
        "average_processing_time_ms": round(avg_processing_time * 1000, 2),
        "model_load_time_seconds": round(METRICS["model_load_time"], 2),
    }


@app.post("/v1/rerank", response_model=RerankResponse, summary="Rerank Documents")
@app.post("/rerank", response_model=RerankResponse, summary="Rerank Documents")
async def rerank(raw_request: Request, authenticated: bool = Depends(verify_api_key)):
    """
    Reranks documents based on a query and returns the top_k results
    with relevance scores.

    Authentication:
    - Bearer token: Authorization: Bearer <your-api-key>
    - API key header: X-API-Key: <your-api-key>
    """
    start_time = datetime.now()

    # Log raw request body for debugging
    try:
        raw_body = await raw_request.body()
        raw_body_str = raw_body.decode("utf-8") if raw_body else ""

        # Parse and validate the request
        if raw_body_str:
            request_data = json.loads(raw_body_str)
        else:
            request_data = {}

        # Log basic request info
        safe_log(
            logger.info,
            "REQUEST SUMMARY: query_length=%d, documents_count=%d, top_n=%s, body_size=%d bytes",
            len(request_data.get("query", "")),
            len(request_data.get("documents", [])),
            request_data.get("top_n", "None"),
            len(raw_body_str),
        )

        # Validate using Pydantic
        request = RerankRequest(**request_data)

    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON format: {str(e)}"
        safe_log(logger.error, "JSON decode error: %s", error_msg)
        METRICS["requests_total"] += 1
        METRICS["requests_failed"] += 1
        raise HTTPException(status_code=422, detail=error_msg)
    except ValidationError as e:
        # Detailed validation error logging
        error_details = []
        for error in e.errors():
            field = " -> ".join(str(x) for x in error["loc"])
            error_details.append(
                f"Field '{field}': {error['msg']} (received: {error.get('input', 'N/A')})"
            )

        error_msg = f"Request validation failed: {'; '.join(error_details)}"
        safe_log(logger.error, "Pydantic validation error: %s", error_msg)
        METRICS["requests_total"] += 1
        METRICS["requests_failed"] += 1
        raise HTTPException(status_code=422, detail=error_msg)
    except Exception as e:
        error_msg = f"Request processing error: {str(e)}"
        safe_log(logger.error, "Request processing error: %s", error_msg)
        METRICS["requests_total"] += 1
        METRICS["requests_failed"] += 1
        raise HTTPException(status_code=500, detail=error_msg)

    safe_log(
        logger.info,
        "VALIDATED REQUEST: %s",
        json.dumps(
            {
                "timestamp": start_time.isoformat(),
                "endpoint": "/rerank",
                "model": request.model,
                "query_length": len(request.query),
                "documents_count": len(request.documents),
                "top_n": request.top_n,
            },
            ensure_ascii=False,
        ),
    )

    if not request.documents:
        return {"results": []}

    log_gpu_memory("before processing")

    # Create pairs of (query, document) for the model
    # instruction = "Evaluate how relevant the following document is to the query for retrieving useful information to answer or provide context for the query"
    instruction = "Given a web search query, retrieve relevant passages that answer the query"  # using the default instruction from the Qwen3-Reranker model achieved best results
    pairs = [
        format_instruction(request.query, doc, instruction) for doc in request.documents
    ]

    # Process with adaptive batching to handle OOM errors gracefully
    try:
        all_scores = process_with_adaptive_batching(pairs)
    except RuntimeError as e:
        error_msg = f"Memory processing error: {str(e)}"
        safe_log(logger.error, "Memory processing error: %s", error_msg)
        METRICS["requests_total"] += 1
        METRICS["requests_failed"] += 1
        raise HTTPException(status_code=500, detail=error_msg)

    log_gpu_memory("after processing")

    # Associate scores with original document indices
    indexed_results = [
        {"index": i, "relevance_score": score} for i, score in enumerate(all_scores)
    ]

    # Sort results by relevance score in descending order
    sorted_results = sorted(
        indexed_results, key=lambda x: x["relevance_score"], reverse=True
    )

    # Apply top_n truncation if specified
    if request.top_n is not None and request.top_n > 0:
        final_results = sorted_results[: request.top_n]
    else:
        final_results = sorted_results

    response = {"results": final_results}
    end_time = datetime.now()

    # Update metrics
    METRICS["requests_total"] += 1
    METRICS["requests_successful"] += 1
    METRICS["total_processing_time"] += (end_time - start_time).total_seconds()
    METRICS["total_documents_processed"] += len(request.documents)

    safe_log(
        logger.info,
        "RESPONSE: %s",
        json.dumps(
            {
                "timestamp": end_time.isoformat(),
                "endpoint": "/rerank",
                "processing_time_ms": (end_time - start_time).total_seconds() * 1000,
                "results_count": len(final_results),
            },
            ensure_ascii=False,
        ),
    )

    return response
