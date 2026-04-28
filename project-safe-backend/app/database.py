"""Backend database helpers.

Re-export the shared Database.models persistence layer so the Flask app
can use it without duplicating schema definitions.
"""

from Database.models import (  # noqa: F401
    Base,
    CallLog,
    RiskLabel,
    configure_database,
    get_engine,
    get_session,
    record_call_log,
    reset_database,
)

__all__ = [
    "Base",
    "CallLog",
    "RiskLabel",
    "configure_database",
    "get_engine",
    "get_session",
    "record_call_log",
    "reset_database",
]
