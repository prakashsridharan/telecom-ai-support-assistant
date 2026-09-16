# Telecom AI Support Assistant

[![CI](https://github.com/prakashsridharan/telecom-ai-support-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/prakashsridharan/telecom-ai-support-assistant/actions/workflows/ci.yml)

A portfolio-grade AI customer support application demonstrating **RAG, agentic tool calling, conversation memory, REST APIs, evaluation, and containerized deployment**.

> This is a demo/portfolio project using synthetic telecom data. It is not a production telecom system and contains no real customer information.

## What it demonstrates

- **Pluggable model providers** — Claude, OpenAI, or a keyless demo responder,
  behind one interface no application code imports a vendor SDK through
- **Real tool calling** — the model receives tool schemas and decides what to
  call, with a bounded loop, or the application routes by rules; both drive the
  same tools and return the same response envelope
- Retrieval-Augmented Generation (RAG) over a small telecom knowledge base,
  ranked by TF-IDF cosine similarity with a relevance floor
- LLM-powered response generation, with graceful degradation when the provider fails
- Multi-turn conversation context: follow-up turns inherit intent and identifiers
- Safe fallback when the system lacks sufficient information
- A golden-set evaluation harness scoring routing, retrieval, grounding and safety
- FastAPI REST backend
- Browser chat UI that exposes the decision trail for every answer
- API documentation through Swagger/OpenAPI
- Docker containerization
- 57 unit tests, including the tool-calling loop exercised through scripted
  fake provider clients — no API key, no flakiness
- Readiness health endpoint and structured JSON logging

## Architecture

```text
Customer
   |
   v
Browser Chat UI
   |
   v
FastAPI
   |
   v
AI Orchestrator
   |----------------------|
   |                      |
   v                      v
Knowledge Retriever     Business Tools
   |                   /      |       \
   v                  v       v        v
Telecom KB          Orders  Billing  Service
   |
   v
LLM Provider
   |
   v
Grounded Response
```

## Configuring the model and the routing strategy

Two independent switches. `LLM_PROVIDER` decides **who writes the reply**;
`ROUTING_MODE` decides **who chooses the tools**.

```bash
# Default: no API key needed, fully functional
LLM_PROVIDER=demo
ROUTING_MODE=rules

# Claude writes the replies, the application still routes
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-opus-5

# Claude chooses and calls the tools itself — real function calling
ROUTING_MODE=agent
```

Swap `anthropic` for `openai` (`OPENAI_API_KEY`, `OPENAI_MODEL`) and everything
else is unchanged — that is the point of the provider boundary.

A provider named without its API key **degrades to demo mode rather than
failing to start**, and `/health` reports exactly why. Same for agent mode
without a tool-capable provider.

| | `ROUTING_MODE=rules` | `ROUTING_MODE=agent` |
|---|---|---|
| Who picks the tool | The application, by regex and keyword | The model, by function calling |
| Deterministic | Yes | No |
| Needs an API key | No | Yes |
| Handles unanticipated phrasing | Poorly | Well |

Both strategies call the same functions and return the same envelope, so the
evaluation harness can score either. Details and trade-offs:
[docs/providers-and-routing.md](docs/providers-and-routing.md).

## Project structure

```text
telecom-ai-support-assistant/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── models.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── logging_config.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── orchestrator.py
│   │   ├── retriever.py
│   │   ├── tools.py              provider-neutral tool specs + dispatch
│   │   └── providers/
│   │       ├── base.py           ChatProvider contract, prompts
│   │       ├── demo.py           keyless deterministic responder
│   │       ├── anthropic_provider.py
│   │       └── openai_provider.py
│   ├── tools/
│   │   ├── __init__.py
│   │   └── telecom_tools.py
│   └── static/
│       └── index.html
├── knowledge_base/
│   ├── plans.md
│   ├── billing.md
│   ├── orders.md
│   └── service_policies.md
├── tests/
│   ├── fakes.py                  scripted provider clients
│   ├── test_agent_tool_calling.py
│   ├── test_health.py
│   ├── test_orchestrator.py
│   ├── test_providers.py
│   ├── test_retriever.py
│   └── test_routing_modes.py
├── evals/
│   ├── golden_set.json
│   └── run_eval.py
├── docs/
│   ├── architecture.md
│   ├── conversation-flows.md
│   ├── deployment.md
│   ├── evaluation.md
│   └── providers-and-routing.md
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── render.yaml
├── requirements.txt
└── LICENSE
```

## Run locally

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Copy `.env.example` to `.env`.

The application works without an API key in demo mode.

### 4. Start the application

```bash
uvicorn app.main:app --reload
```

Open:

- Chat UI: http://localhost:8000
- Swagger: http://localhost:8000/docs
- Health: http://localhost:8000/health

## Example questions

Try:

- `What plans do you offer?`
- `How much does the Premium plan cost?`
- `What is your billing policy?`
- `Check order ORD-10234`
- `Where is my order ORD-10001?`
- `Is there an outage in Chennai?`
- `What happens if I don't know my order number?`

## Tests and evaluation

Unit tests:

```bash
pytest
```

Golden-set evaluation — scores the assistant on intent routing, tool-call
accuracy, retrieval recall, top-1 ranking, grounding and safety:

```bash
python -m evals.run_eval
python -m evals.run_eval --json --min-pass 0.95   # for CI
```

The harness exits non-zero below the pass-rate threshold, so it can gate a
pipeline. Cases live in [`evals/golden_set.json`](evals/golden_set.json); adding
a case is a JSON edit, not a code change.

## Docker

```bash
docker compose up --build
```

Then open http://localhost:8000.

The Docker setup follows the standard containerized FastAPI approach. FastAPI documents container images as a common deployment method, and Render supports FastAPI web services with a standard Uvicorn start command. See the references in `docs/deployment.md`.

## Deploy to Render

`render.yaml` is committed as a Blueprint, so the service is defined in the
repository rather than clicked together in a dashboard.

1. Sign in to [Render](https://render.com) with GitHub.
2. **New → Blueprint**, and select this repository.
3. Apply the proposed service. `autoDeploy` is on, so every push to `main`
   redeploys.

The public demo runs in **demo mode** on purpose: the endpoint is
unauthenticated and unthrottled, so a live provider key would make it an open,
billable LLM endpoint. Demo mode exercises retrieval, routing, tool calls and
refusals in full. See [docs/deployment.md](docs/deployment.md) for the details,
including free-tier cold starts.

## Portfolio positioning

This project demonstrates how I approach AI applications as an engineer:

**Business workflow → conversation design → retrieval → tool/API integration → grounded response → evaluation → deployment.**

It intentionally avoids treating an LLM as the entire solution.
