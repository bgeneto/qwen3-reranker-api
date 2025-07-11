#!/usr/bin/env python3
"""
Test script for Qwen Reranker API
"""

import json
import requests
import time
from typing import List, Dict


class RerankerTester:
    def __init__(self, base_url: str = "http://localhost:8004", api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = self._get_headers()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers including API key if provided"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            # Support both Bearer token and X-API-Key header
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-API-Key"] = self.api_key
        return headers

    def test_health(self) -> bool:
        """Test health endpoint (unprotected)"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

    def test_metrics(self) -> Dict:
        """Test metrics endpoint (protected)"""
        try:
            response = requests.get(
                f"{self.base_url}/metrics", headers=self.headers, timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            print(f"Metrics check failed: {e}")
            return {}

    def test_rerank(
        self, query: str, documents: List[str], top_n: int = None, model: str = None
    ) -> Dict:
        """Test rerank endpoint"""
        payload = {"query": query, "documents": documents}
        if top_n is not None:
            payload["top_n"] = top_n
        if model is not None:
            payload["model"] = model

        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/rerank",
                json=payload,
                headers=self.headers,
                timeout=30,
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

    def test_authentication_failure(self) -> bool:
        """Test that protected endpoints return 401 without proper authentication"""
        try:
            # Test root endpoint without auth
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code != 401:
                return False

            # Test metrics endpoint without auth
            response = requests.get(f"{self.base_url}/metrics", timeout=10)
            if response.status_code != 401:
                return False

            # Test rerank endpoint without auth
            response = requests.post(
                f"{self.base_url}/rerank",
                json={"query": "test", "documents": ["doc1"]},
                timeout=10,
            )
            if response.status_code != 401:
                return False

            return True
        except Exception as e:
            print(f"Authentication test failed: {e}")
            return False

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
            return  # Test 2: Authentication (if API key not provided)
        if not self.api_key:
            print("2. Testing authentication failures (no API key provided)...")
            auth_test = self.test_authentication_failure()
            if auth_test:
                print("   ✅ Authentication properly enforced - 401 responses received")
            else:
                print(
                    "   ⚠️  Authentication test failed - endpoints may not be properly protected"
                )
        else:
            print("2. Skipping authentication failure test (API key provided)")

        # Test 3: Health endpoint (unprotected)
        print("3. Testing unprotected health endpoint...")
        if self.test_health():
            print("   ✅ Health check passed (unprotected)")
        else:
            print("   ❌ Health check failed")

        # Test 4: Metrics (protected)
        print("4. Testing protected metrics endpoint...")
        metrics = self.test_metrics()
        if metrics:
            print(f"   ✅ Metrics retrieved: {json.dumps(metrics, indent=2)}")
        else:
            if self.api_key:
                print("   ❌ Metrics failed")
            else:
                print("   ✅ Metrics properly protected (401 expected without API key)")

        # Test 5: Basic reranking
        print("3. Testing basic reranking...")
        test_query = "What is the capital of France?"
        test_documents = [
            "Paris is the capital of France and its largest city.",
            "London is the capital of the United Kingdom.",
            "The Eiffel Tower is located in Paris, France.",
            "Berlin is the capital of Germany.",
        ]

        result = self.test_rerank(test_query, test_documents, top_n=2)
        if result["success"]:
            print(f"   ✅ Basic reranking passed")
            print(f"   ⏱️  Response time: {result['elapsed_time']:.3f}s")
            print(f"   📊 Results count: {result['results_count']}")
            for i, res in enumerate(result["response"]["results"]):
                print(
                    f"      {i+1}. Document {res['index']}: {res['relevance_score']:.4f}"
                )
        else:
            if self.api_key:
                print(
                    f"   ❌ Basic reranking failed: {result.get('error', 'Unknown error')}"
                )
            else:
                print(
                    "   ✅ Reranking properly protected (401 expected without API key)"
                )

        # Test 3.1: Cohere-compatible format with model field
        print("3.1. Testing Cohere-compatible format with model field...")
        cohere_result = self.test_rerank(
            test_query, test_documents, top_n=2, model="rerank-v3.5"
        )
        if cohere_result["success"]:
            print(f"   ✅ Cohere format test passed")
            print(f"   ⏱️  Response time: {cohere_result['elapsed_time']:.3f}s")
        else:
            if self.api_key:
                print(
                    f"   ❌ Cohere format test failed: {cohere_result.get('error', 'Unknown error')}"
                )
            else:
                print(
                    "   ✅ Cohere format properly protected (401 expected without API key)"
                )

        # Test 4: Edge cases
        print("4. Testing edge cases...")

        # Empty documents
        empty_result = self.test_rerank("test", [])
        if empty_result["success"] and empty_result["results_count"] == 0:
            print("   ✅ Empty documents handled correctly")
        else:
            print("   ❌ Empty documents test failed")

        # Large top_n
        large_n_result = self.test_rerank(test_query, test_documents, top_n=100)
        if large_n_result["success"]:
            print("   ✅ Large top_n handled correctly")
        else:
            print("   ❌ Large top_n test failed")

        # Test 5: Authentication failures
        print("5. Testing authentication failures...")
        if self.test_authentication_failure():
            print("   ✅ Authentication failure tests passed")
        else:
            print("   ❌ Authentication failure tests failed")

        print("=" * 50)
        print("🎉 Test suite completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Qwen Reranker API")
    parser.add_argument("--url", default="http://localhost:8004", help="API base URL")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--query", help="Custom query for testing")
    parser.add_argument("--documents", nargs="+", help="Custom documents for testing")

    args = parser.parse_args()

    tester = RerankerTester(args.url, getattr(args, "api_key", None))

    if args.query and args.documents:
        print(f"Testing custom query: {args.query}")
        result = tester.test_rerank(args.query, args.documents)
        print(json.dumps(result, indent=2))
    else:
        tester.run_basic_tests()
