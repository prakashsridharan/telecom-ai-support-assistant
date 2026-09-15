# Deployment

## Render (public demo)

`render.yaml` is committed as a Blueprint, so the service is defined in the
repository rather than configured by hand in a dashboard.

1. Sign in at [render.com](https://render.com) with GitHub.
2. **New → Blueprint**, select the `telecom-ai-support-assistant` repository.
3. Render reads `render.yaml` and proposes one free web service. Apply it.
4. Leave `OPENAI_API_KEY` empty — see "Why the demo stays in demo mode" below.

The first build takes a few minutes. `autoDeploy` is on, so every push to
`main` redeploys.

If you prefer configuring the service manually instead of via Blueprint:

- Build: `pip install -r requirements.txt`
- Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`

### Why the demo stays in demo mode

The deployed endpoint is public, unauthenticated and unthrottled. Supplying a
live provider key would turn it into an open, billable LLM endpoint that anyone
could drive. Demo mode is fully functional — retrieval, routing, tool calls and
refusals all work — so the deployment demonstrates the architecture without that
exposure. Add a key only behind authentication and a rate limit.

### Free-tier cold starts

Free Render instances sleep after roughly 15 minutes of inactivity. The first
request to a sleeping instance waits ~30-50 seconds while it wakes. For a link
shared with a client, either warm it before the conversation or move to a paid
instance.

### Health checks

`/health` returns HTTP 200 whenever the process is up, and reports component
readiness in the body:

```json
{
  "status": "ok",
  "environment": "production",
  "mode": "demo",
  "components": {
    "retriever": { "status": "ok", "detail": "4 knowledge documents loaded" },
    "llm": { "status": "ok", "detail": "Demo mode; no LLM provider configured" }
  }
}
```

`status` is `degraded` when the app is configured for a provider it cannot use,
and `error` when the knowledge base failed to load. Render's health check only
observes the status code, so check the body after a deploy — a running service
with an empty knowledge base still returns 200.

## Docker

```bash
docker build -t telecom-ai-support .
docker run --rm -p 8000:8000 telecom-ai-support
```

Or `docker compose up --build`.

The image copies `app/`, `knowledge_base/` and `docs/` under `WORKDIR /app`. The
retriever resolves its knowledge-base path against the repository root rather
than the working directory, so the container and local runs behave identically.

## AWS and other platforms

The same image runs on any managed container platform — App Runner, ECS/Fargate,
Cloud Run. Nothing in `app/` is host-specific; the only platform coupling is the
`$PORT` binding in the start command, which is a runtime argument rather than
application code.

## Configuration

All settings come from environment variables (see `.env.example`). The ones that
matter in a deployment:

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `demo` | `openai` enables live mode — requires a non-empty key too |
| `OPENAI_API_KEY` | empty | Declared `sync: false` in `render.yaml`, never committed |
| `LLM_TIMEOUT_SECONDS` | `20` | Provider timeout before falling back to demo output |
| `LOG_FORMAT` | `json` | `text` for readable local development |
| `LOG_LEVEL` | `INFO` | |
| `HISTORY_MAX_TURNS` | `8` | Bounds prompt size and identifier lookback |

A provider failure degrades to the grounded demo responder and is labelled
`mode: "fallback"` in the response, so a broken key surfaces in the payload and
the logs rather than as a 500.
