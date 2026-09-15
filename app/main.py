from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.config import settings
from app.logging_config import configure_logging

# Configure logging before importing the router: importing it constructs the
# Orchestrator, which loads the knowledge base and logs the result.
configure_logging()

from app.api.routes import router  # noqa: E402

app = FastAPI(
    title=settings.app_name,
    version="1.1.0",
    description=(
        "Portfolio-grade telecom customer-support assistant demonstrating "
        "RAG, agentic tools, FastAPI and LLM integration."
    ),
)

app.include_router(router)

STATIC_FILE = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_FILE)
