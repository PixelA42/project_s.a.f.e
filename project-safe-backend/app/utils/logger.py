"""Application logging setup."""
import logging
from flask import Flask


def setup_logger(app: Flask) -> None:
    """Configure a simple console logger for the Flask app."""
    if app.logger.handlers:
        return

    level = logging.DEBUG if app.config.get("DEBUG", False) else logging.INFO
    app.logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)

    app.logger.addHandler(handler)
