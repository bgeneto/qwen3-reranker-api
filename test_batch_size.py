#!/usr/bin/env python3
"""
GPU Memory Test Script for Qwen3 Reranker API

This script helps you find the optimal BATCH_SIZE for your GPU by testing
different batch sizes and monitoring memory usage.
"""

import requests
import json
import time
import sys
from typing import List


def test_batch_size(batch_size: int, base_url: str = "http://localhost:8004") -> bool:
    """Test a specific batch size with a sample request."""

    # Create a test request with multiple documents
    test_documents = [
        f"This is test document number {i} about machine learning and artificial intelligence."
        for i in range(batch_size * 2)  # Test with 2x batch size worth of documents
    ]

    payload = {
        "query": "What is machine learning and artificial intelligence?",
        "documents": test_documents,
        "top_n": 10,
    }

    try:
        print(
            f"Testing batch size {batch_size} with {len(test_documents)} documents..."
        )
        start_time = time.time()

        response = requests.post(
            f"{base_url}/rerank",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )

        end_time = time.time()

        if response.status_code == 200:
            processing_time = end_time - start_time
            result = response.json()
            print(
                f"✓ Success! Batch size {batch_size}: {processing_time:.2f}s, {len(result['results'])} results"
            )
            return True
        else:
            print(f"✗ Failed! Batch size {batch_size}: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print(f"✗ Timeout! Batch size {batch_size}: Request timed out")
        return False
    except Exception as e:
        print(f"✗ Error! Batch size {batch_size}: {str(e)}")
        return False


def find_optimal_batch_size(base_url: str = "http://localhost:8004") -> int:
    """Find the optimal batch size by testing incrementally."""

    print("Finding optimal batch size for your GPU...")
    print("=" * 50)

    # Check if service is running
    try:
        health_response = requests.get(f"{base_url}/health", timeout=10)
        if health_response.status_code != 200:
            print(f"Error: Service not responding at {base_url}")
            sys.exit(1)
    except Exception as e:
        print(f"Error: Cannot connect to service at {base_url}: {e}")
        sys.exit(1)

    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 24, 32, 48, 64]
    optimal_batch_size = 1

    for batch_size in batch_sizes:
        if test_batch_size(batch_size, base_url):
            optimal_batch_size = batch_size
            time.sleep(2)  # Wait between tests
        else:
            break

    print("=" * 50)
    print(f"Recommended BATCH_SIZE: {optimal_batch_size}")
    print(f"Update your .env file with: BATCH_SIZE={optimal_batch_size}")

    return optimal_batch_size


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test optimal batch size for Qwen3 Reranker API"
    )
    parser.add_argument(
        "--url", default="http://localhost:8004", help="Base URL of the API service"
    )
    parser.add_argument("--batch-size", type=int, help="Test a specific batch size")

    args = parser.parse_args()

    if args.batch_size:
        success = test_batch_size(args.batch_size, args.url)
        sys.exit(0 if success else 1)
    else:
        find_optimal_batch_size(args.url)
