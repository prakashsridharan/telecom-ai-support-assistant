# Deployment

## Render

The repository includes `render.yaml` for reproducible deployment.

Render's FastAPI guidance uses:
- Build: `pip install -r requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`

This project uses the equivalent module path `app.main:app`.

## Docker

Build:

```bash
docker build -t telecom-ai-support .
```

Run:

```bash
docker run --rm -p 8000:8000 telecom-ai-support
```

## AWS

The same Docker image can be deployed to a managed container platform such as AWS App Runner or ECS/Fargate. The application intentionally avoids Render-specific code.
