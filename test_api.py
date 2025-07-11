#!/usr/bin/env python3
"""
Test script for Qwen Reranker API
"""

import json
import requests
import time
from typing import List, Dict


class RerankerTester:
    def __init__(self, base_url: str = "http://localhost:8004"):
        self.base_url = base_url

    def test_health(self) -> bool:
        """Test health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

    def test_metrics(self) -> Dict:
        """Test metrics endpoint"""
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=10)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            print(f"Metrics check failed: {e}")
            return {}

    def test_rerank(self, query: str, documents: List[str], top_k: int = None) -> Dict:
        """Test rerank endpoint"""
        payload = {"query": query, "documents": documents, "top_k": top_k}

        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/rerank", json=payload, timeout=30
            )
            elapsed = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "elapsed_time": elapsed,
                    "results_count": len(result.get("results", [])),
                    "response": result,
                }
            else:
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "error": response.text,
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_basic_tests(self):
        """Run basic functionality tests"""
        print("🧪 Running Qwen Reranker API Tests...")
        print("=" * 50)

        # Test 1: Health check
        print("1. Testing health endpoint...")
        if self.test_health():
            print("   ✅ Health check passed")
        else:
            print("   ❌ Health check failed")
            return

        # Test 2: Metrics
        print("2. Testing metrics endpoint...")
        metrics = self.test_metrics()
        if metrics:
            print(f"   ✅ Metrics retrieved: {json.dumps(metrics, indent=2)}")
        else:
            print("   ❌ Metrics failed")

        # Test 3: Basic reranking
        print("3. Testing basic reranking...")
        test_query = "What is the capital of France?"
        test_documents = [
            "Paris is the capital of France and its largest city.",
            "London is the capital of the United Kingdom.",
            "The Eiffel Tower is located in Paris, France.",
            "Berlin is the capital of Germany.",
        ]

        result = self.test_rerank(test_query, test_documents, top_k=2)
        if result["success"]:
            print(f"   ✅ Basic reranking passed")
            print(f"   ⏱️  Response time: {result['elapsed_time']:.3f}s")
            print(f"   📊 Results count: {result['results_count']}")
            for i, res in enumerate(result["response"]["results"]):
                print(
                    f"      {i+1}. Document {res['index']}: {res['relevance_score']:.4f}"
                )
        else:
            print(
                f"   ❌ Basic reranking failed: {result.get('error', 'Unknown error')}"
            )

        # Test 4: Edge cases
        print("4. Testing edge cases...")

        # Empty documents
        empty_result = self.test_rerank("test", [])
        if empty_result["success"] and empty_result["results_count"] == 0:
            print("   ✅ Empty documents handled correctly")
        else:
            print("   ❌ Empty documents test failed")

        # Large top_k
        large_k_result = self.test_rerank(test_query, test_documents, top_k=100)
        if large_k_result["success"]:
            print("   ✅ Large top_k handled correctly")
        else:
            print("   ❌ Large top_k test failed")

        print("=" * 50)
        print("🎉 Test suite completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Qwen Reranker API")
    parser.add_argument("--url", default="http://localhost:8004", help="API base URL")
    parser.add_argument("--query", help="Custom query for testing")
    parser.add_argument("--documents", nargs="+", help="Custom documents for testing")

    args = parser.parse_args()

    tester = RerankerTester(args.url)

    if args.query and args.documents:
        print(f"Testing custom query: {args.query}")
        result = tester.test_rerank(args.query, args.documents)
        print(json.dumps(result, indent=2))
    else:
        tester.run_basic_tests()
