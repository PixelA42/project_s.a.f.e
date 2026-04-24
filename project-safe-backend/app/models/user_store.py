"""
In-memory user store with bcrypt password hashing.

This is intentionally simple — replace the `_USERS` dict with
a real database (PostgreSQL via SQLAlchemy, MongoDB, etc.) when ready.
The interface (get_by_email, create, verify_password) stays the same.
"""
import uuid
from datetime import datetime, timezone
from typing import TypedDict

import bcrypt


class UserRecord(TypedDict):
    id: str
    name: str
    email: str
    password_hash: str
    plan: str
    joined_at: str
    calls_analyzed: int
    threats_blocked: int


# ── In-memory store ────────────────────────────────────────────────────
# Key: email (lowercase), Value: UserRecord
_USERS: dict[str, UserRecord] = {}


def get_by_email(email: str) -> UserRecord | None:
    return _USERS.get(email.lower())


def get_by_id(user_id: str) -> UserRecord | None:
    return next((u for u in _USERS.values() if u["id"] == user_id), None)


def create(name: str, email: str, password: str) -> UserRecord:
    """
    Creates and stores a new user. Raises ValueError if email already exists.

    ── TO REPLACE WITH A REAL DATABASE ───────────────────────────────────
    Replace the _USERS dict operations with SQLAlchemy / psycopg2 / pymongo
    INSERT calls. The bcrypt hashing logic stays identical.

    Example (SQLAlchemy):
        user = User(id=..., name=..., email=..., password_hash=...)
        db.session.add(user)
        db.session.commit()
    ──────────────────────────────────────────────────────────────────────
    """
    email = email.lower().strip()

    if get_by_email(email):
        raise ValueError(f"Email {email} is already registered")

    salt = bcrypt.gensalt()
    password_hash = bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    record: UserRecord = {
        "id": str(uuid.uuid4()),
        "name": name.strip(),
        "email": email,
        "password_hash": password_hash,
        "plan": "free",
        "joined_at": datetime.now(timezone.utc).strftime("%b %Y"),
        "calls_analyzed": 0,
        "threats_blocked": 0,
    }

    _USERS[email] = record
    return record


def verify_password(user: UserRecord, password: str) -> bool:
    return bcrypt.checkpw(
        password.encode("utf-8"),
        user["password_hash"].encode("utf-8"),
    )


def increment_stats(user_id: str, threat_blocked: bool = False) -> None:
    """Called after each analysis to update the user's stats."""
    user = get_by_id(user_id)
    if user:
        user["calls_analyzed"] += 1
        if threat_blocked:
            user["threats_blocked"] += 1


def safe_public(user: UserRecord) -> dict:
    """Returns a user dict safe to send to the frontend (no password hash)."""
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "plan": user["plan"],
        "joinedAt": user["joined_at"],
        "callsAnalyzed": user["calls_analyzed"],
        "threatsBlocked": user["threats_blocked"],
    }