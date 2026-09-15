"""Structured application logging.

Emits one JSON object per line so the logs are greppable locally and ingestible
by a log platform (CloudWatch, Datadog, Loki) without a parsing rule. Any extra
fields passed through `logger.info(..., extra={...})` are merged into the object.
"""

import json
import logging
import sys
from datetime import datetime, timezone

from app.config import settings

# Attributes present on every LogRecord; anything else was supplied by the
# caller via `extra=` and belongs in the structured payload.
_RESERVED = set(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Install the configured formatter on the root logger.

    Safe to call more than once; existing handlers are replaced rather than
    stacked, so reload-driven restarts do not duplicate every line.
    """
    if settings.log_format.lower() == "json":
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s  %(message)s"
        )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level.upper())

    # Uvicorn installs its own handlers; let its records propagate to ours so
    # application and server logs share one format.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        server_logger = logging.getLogger(name)
        server_logger.handlers = []
        server_logger.propagate = True
