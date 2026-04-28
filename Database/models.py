"""Database models and persistence helpers for Project S.A.F.E.

Requirement 5: Call Log Persistence.
"""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Iterator

from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, String, Text, create_engine
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.types import CHAR, Enum as SAEnum, TypeDecorator


class Base(DeclarativeBase):
	"""Declarative base for all ORM models."""


class GUID(TypeDecorator):
	"""Platform-independent UUID column type."""

	impl = CHAR(36)
	cache_ok = True

	def load_dialect_impl(self, dialect):
		if dialect.name == "postgresql":
			return dialect.type_descriptor(PGUUID(as_uuid=True))
		return dialect.type_descriptor(CHAR(36))

	def process_bind_param(self, value, dialect):
		if value is None:
			return value
		if not isinstance(value, uuid.UUID):
			value = uuid.UUID(str(value))
		return value if dialect.name == "postgresql" else str(value)

	def process_result_value(self, value, dialect):
		if value is None:
			return value
		if isinstance(value, uuid.UUID):
			return value
		return uuid.UUID(str(value))


class UTCDateTime(TypeDecorator):
	"""Timezone-aware UTC datetime storage with SQLite round-trip support."""

	impl = DateTime(timezone=True)
	cache_ok = True

	def process_bind_param(self, value, dialect):
		if value is None:
			return value
		if value.tzinfo is None:
			value = value.replace(tzinfo=timezone.utc)
		return value.astimezone(timezone.utc)

	def process_result_value(self, value, dialect):
		if value is None:
			return value
		if value.tzinfo is None:
			return value.replace(tzinfo=timezone.utc)
		return value.astimezone(timezone.utc)


class RiskLabel(str, Enum):
	"""Allowed risk labels persisted for call evaluations."""

	HIGH_RISK = "HIGH_RISK"
	PRANK = "PRANK"
	SAFE = "SAFE"


class CallLog(Base):
	"""Persisted risk evaluation record."""

	__tablename__ = "call_logs"
	__table_args__ = (
		CheckConstraint("spectral_score >= 0 AND spectral_score <= 100", name="ck_call_logs_spectral_score"),
		CheckConstraint("intent_score >= 0 AND intent_score <= 100", name="ck_call_logs_intent_score"),
		CheckConstraint("final_score >= 0 AND final_score <= 100", name="ck_call_logs_final_score"),
	)

	id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
	timestamp: Mapped[datetime] = mapped_column(
		UTCDateTime(),
		nullable=False,
		default=lambda: datetime.now(timezone.utc),
	)
	caller_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
	audio_file_path: Mapped[str] = mapped_column(String(512), nullable=False)
	transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
	spectral_score: Mapped[float] = mapped_column(Float, nullable=False)
	intent_score: Mapped[float] = mapped_column(Float, nullable=False)
	final_score: Mapped[float] = mapped_column(Float, nullable=False)
	risk_label: Mapped[RiskLabel] = mapped_column(
		SAEnum(RiskLabel, name="risk_label", native_enum=False, validate_strings=True),
		nullable=False,
	)
	anomaly_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="0")


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine(database_url: str | None = None) -> Engine:
	"""Create a SQLAlchemy engine for the given database URL."""

	url = database_url or os.getenv("DATABASE_URL", "sqlite:///project_safe_backend.db")
	engine_kwargs: dict[str, object] = {}

	if url.startswith("sqlite"):
		engine_kwargs["connect_args"] = {"check_same_thread": False}
		if url in {"sqlite://", "sqlite:///:memory:"} or url.endswith(":memory:"):
			engine_kwargs["poolclass"] = StaticPool

	return create_engine(url, future=True, **engine_kwargs)


def configure_database(database_url: str | None = None) -> Engine:
	"""Initialize the global engine and session factory, then create tables."""

	global _engine, _session_factory

	if _engine is not None:
		_engine.dispose()

	_engine = get_engine(database_url)
	_session_factory = sessionmaker(
		bind=_engine,
		autoflush=False,
		autocommit=False,
		future=True,
		expire_on_commit=False,
	)
	Base.metadata.create_all(bind=_engine)
	return _engine


def reset_database() -> None:
	"""Reset the global engine and session factory for tests."""

	global _engine, _session_factory

	if _engine is not None:
		_engine.dispose()
	_engine = None
	_session_factory = None


@contextmanager
def get_session() -> Iterator[Session]:
	"""Yield a transactional session.

	Commits on success and rolls back on any exception.
	"""

	global _session_factory

	if _session_factory is None:
		configure_database()

	assert _session_factory is not None
	session = _session_factory()
	try:
		yield session
		session.commit()
	except Exception:
		session.rollback()
		raise
	finally:
		session.close()


def record_call_log(
	session: Session,
	*,
	caller_id: str | None,
	audio_file_path: str,
	transcript: str | None,
	spectral_score: float,
	intent_score: float,
	final_score: float,
	risk_label: RiskLabel | str,
	anomaly_flag: bool = False,
	timestamp: datetime | None = None,
) -> CallLog:
	"""Create a persisted CallLog row and flush it for the generated UUID."""

	if not audio_file_path or not str(audio_file_path).strip():
		raise ValueError("audio_file_path is required")

	record = CallLog(
		caller_id=caller_id,
		audio_file_path=audio_file_path.strip(),
		transcript=transcript,
		spectral_score=float(spectral_score),
		intent_score=float(intent_score),
		final_score=float(final_score),
		risk_label=RiskLabel(risk_label),
		anomaly_flag=bool(anomaly_flag),
		timestamp=timestamp or datetime.now(timezone.utc),
	)
	session.add(record)
	session.flush()
	return record


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
