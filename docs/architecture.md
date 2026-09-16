# Architecture

## Design goals

1. Separate conversational orchestration from domain tools.
2. Keep retrieval replaceable.
3. Keep the model provider replaceable.
4. Prevent unsupported answers.
5. Make business API interactions explicit and observable.
6. Make the application deployable as a container.

## Request flow

```text
Browser chat UI
      │  POST /api/chat {message, history}
      ▼
FastAPI  (main.py → api/routes.py)              /health, /docs
      ▼
Orchestrator                     ROUTING_MODE decides who picks the tools
      │
      ├─ rules ─► classify() ─► extract identifier ─► one tool
      │                                                   │
      └─ agent ─► provider.run_agent() ◄──── tool schemas ─┤
                        │  model picks tools, loops        │
                        ▼                                  ▼
                                              services/tools.py  (ToolBox)
                                                   │
                        ┌──────────────────────────┼─────────────────┐
                        ▼                          ▼                 ▼
              search_knowledge_base        look_up_order     check_service_status
                        │                  look_up_billing
                        ▼                          │
              KnowledgeRetriever                   ▼
              knowledge_base/*.md        tools/telecom_tools.py
                                        (synthetic enterprise APIs)
      ▼
Provider  (services/providers/*)   demo | anthropic | openai
      ▼
ChatResponse {answer, intent, sources, tool_calls, mode, routing, provider, model}
```

## Components

### FastAPI layer

`GET /health` — readiness, not just liveness: reports knowledge-base load,
which provider is active, and which routing strategy will actually run.
`POST /api/chat` — the only business endpoint. `/docs` serves OpenAPI.

The route layer holds no business logic. It constructs one module-level
`Orchestrator` at import, so the knowledge base is indexed once rather than per
request, and logs a structured record per request (never the message content).

### Orchestrator

Owns the decision of *what evidence is allowed to answer a question*, and which
of two strategies makes that decision:

- **rules** — classify the message, extract the identifier, call one tool.
  Deterministic, keyless, inspectable.
- **agent** — hand the tool schemas to the provider and let the model choose,
  in a bounded loop. Real function calling.

Both return the same envelope. See
[providers-and-routing.md](providers-and-routing.md).

### Tool layer

`services/tools.py` holds provider-neutral tool specs and dispatch — one source
of truth for what the assistant can do. `tools/telecom_tools.py` holds the
synthetic business systems behind them. Each returns an explicit `found` flag,
so "no record" is distinguishable from "record with empty fields" — which is
what makes the anti-fabrication rule enforceable rather than merely prompted.

In a real deployment these become authenticated REST clients; nothing above
them changes.

### Retriever

TF-IDF cosine similarity over stopword-filtered tokens, with a relevance floor
so an off-topic question retrieves nothing and the assistant refuses. No
external service, ~40 lines of inspectable arithmetic.

Replaceable with pgvector, OpenSearch, Pinecone or any embedding store behind
`search(query, top_k) -> [{source, content, score}]`.

### Provider layer

`services/providers/` is the only place a vendor SDK is imported. `base.py`
defines the contract; `demo.py`, `anthropic_provider.py` and
`openai_provider.py` implement it. Selection is by `LLM_PROVIDER`, and any
misconfiguration degrades to the demo provider rather than failing to boot.

## Production evolution

```text
Browser
  │
CDN / WAF
  │
API Gateway          ── authN/authZ, rate limiting
  │
FastAPI services
  │
Agent Orchestrator
  ├──────── Vector DB / OpenSearch
  ├──────── Customer / Order / Billing APIs   (authenticated)
  └──────── LLM provider
  │
Observability / Evaluation / Audit log
```

A production implementation adds authentication, authorization, rate limiting,
secrets management, PII controls, audit logging, prompt-injection defenses,
conversation persistence, live evaluation, tracing, and human escalation. Those
are named here rather than half-built — see the non-goals in the README.
