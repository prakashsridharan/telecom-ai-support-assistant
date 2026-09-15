# Architecture

## Design goals

1. Separate conversational orchestration from domain tools.
2. Keep retrieval replaceable.
3. Prevent unsupported answers.
4. Make business API interactions explicit.
5. Make the application deployable as a container.
6. Keep provider-specific LLM code behind a service boundary.

## Components

### FastAPI API

Exposes:
- `GET /health`
- `POST /api/chat`

FastAPI also provides interactive OpenAPI documentation at `/docs`.

### Orchestrator

Classifies the request and decides whether the workflow is:
- Knowledge retrieval
- Order lookup
- Billing lookup
- Service-status lookup

### Retriever

The portfolio version uses a transparent lexical retriever so it has no external database dependency.

A production version could replace this component with:
- pgvector
- OpenSearch
- Pinecone
- another embedding/vector store

The orchestrator interface would remain unchanged.

### Tool layer

The tool layer simulates enterprise APIs:
- `get_order_status`
- `get_billing_status`
- `get_service_status`

These can later become secured REST clients.

### LLM service

`LLMService` supports demo mode and OpenAI-backed mode. The provider boundary makes it straightforward to add another model provider.

## Production evolution

```text
Browser
  |
CDN / WAF
  |
API Gateway
  |
FastAPI services
  |
Agent Orchestrator
  |-------- Vector DB / OpenSearch
  |-------- Customer APIs
  |-------- Order APIs
  |-------- Billing APIs
  |
LLM Provider
  |
Observability / Evaluation
```

For a production implementation, add authentication, authorization, rate limiting, secrets management, PII controls, audit logging, prompt-injection defenses, evaluation datasets, tracing, and human escalation.
