"""Logging utilities module."""
import logging
import json
from datetime import datetime, timezone
from typing import Dict, Any
from pathlib import Path
from config.settings import settings


class JSONFormatter(logging.Formatter):
    """JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        if hasattr(record, 'extra_fields'):
            log_obj.update(record.extra_fields)

        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


def setup_logger(name: str = __name__) -> logging.Logger:
    """Set up logger with JSON formatting."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Create logs directory if it doesn't exist
        log_path = Path(settings.LOG_FILE)
        log_path.parent.mkdir(exist_ok=True)

        # File handler with JSON formatting
        file_handler = logging.FileHandler(settings.LOG_FILE)
        file_handler.setFormatter(JSONFormatter())

        # Console handler with standard formatting
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(getattr(logging, settings.LOG_LEVEL))

    return logger


def log_with_context(logger: logging.Logger, level: str, message: str, **kwargs):
    """Log message with additional context."""
    extra = {'extra_fields': kwargs}
    getattr(logger, level.lower())(message, extra=extra)