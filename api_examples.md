# API Usage Examples

This document provides examples of how to use the Qwen3 Reranker API, which is compatible with Cohere's rerank API standards.

## Basic Reranking Request

```bash
curl -X POST "http://localhost:8004/rerank" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "rerank-v3.5",
    "query": "What is the capital of the United States?",
    "top_n": 3,
    "documents": [
      "Carson City is the capital city of the American state of Nevada.",
      "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
      "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
      "Capitalization or capitalisation in English grammar is the use of a capital letter at the start of a word. English usage varies from capitalization in other languages.",
      "Capital punishment has existed in the United States since before the United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states."
    ]
  }'
```

## Response Format

```json
{
  "results": [
    {
      "index": 2,
      "relevance_score": 0.9875
    },
    {
      "index": 0,
      "relevance_score": 0.2341
    },
    {
      "index": 1,
      "relevance_score": 0.1892
    }
  ]
}
```

## Alternative Authentication (X-API-Key Header)

```bash
curl -X POST "http://localhost:8004/rerank" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "query": "machine learning applications",
    "documents": [
      "Machine learning is used in healthcare for diagnosis.",
      "Basketball is a popular sport worldwide.",
      "AI algorithms can predict stock market trends.",
      "Cooking recipes require precise measurements."
    ],
    "top_n": 2
  }'
```

## Minimal Request (without model field)

```bash
curl -X POST "http://localhost:8004/rerank" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "query": "Python programming",
    "documents": [
      "Python is a popular programming language.",
      "Snakes are reptiles that can be dangerous.",
      "JavaScript is used for web development."
    ]
  }'
```

## Health Check

```bash
# Public health check (no authentication required)
curl "http://localhost:8004/health"

# Protected root endpoint (requires authentication)
curl -H "Authorization: Bearer YOUR_API_KEY" "http://localhost:8004/"
```

## Metrics

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" "http://localhost:8004/metrics"
```

## Field Descriptions

- **model**: (Optional) Model identifier for API compatibility. This field is accepted but currently ignored as the service uses the configured model.
- **query**: (Required) The search query to rank documents against.
- **documents**: (Required) Array of documents to rerank.
- **top_n**: (Optional) Number of top results to return. If not specified, all documents are returned ranked by relevance.

## Migration from Previous Version

If you were using the previous API format with `top_k`, simply rename it to `top_n`:

### Old Format (deprecated)
```json
{
  "query": "example query",
  "documents": ["doc1", "doc2"],
  "top_k": 1
}
```

### New Format (Cohere compatible)
```json
{
  "query": "example query",
  "documents": ["doc1", "doc2"],
  "top_n": 1
}
```
