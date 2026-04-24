"""
JWT helpers — issue and verify tokens.
"""
import jwt
from datetime import datetime, timezone, timedelta
from functools import wraps
from flask import request, jsonify, current_app

from app.models import user_store


def issue_token(user_id: str) -> str:
    """
    Issues a signed JWT.
    Algorithm: HS256
    Endpoint: verified locally — no external call needed.
    """
    expiry_hours = current_app.config.get("JWT_EXPIRY_HOURS", 24)
    payload = {
        "sub": user_id,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=expiry_hours),
    }
    secret = current_app.config["JWT_SECRET_KEY"]
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_token(token: str) -> dict:
    """
    Decodes and validates a JWT. Raises jwt.InvalidTokenError on failure.
    """
    secret = current_app.config["JWT_SECRET_KEY"]
    return jwt.decode(token, secret, algorithms=["HS256"])


def _extract_token() -> str | None:
    """Pull Bearer token from Authorization header or cookie."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header.split(" ", 1)[1]
    return request.cookies.get("safe_token")


def require_auth(f):
    """
    Decorator — protects a route with JWT authentication.

    Usage:
        @analyze_bp.route("/analyze", methods=["POST"])
        @require_auth
        def analyze():
            user = g.current_user  # available inside the route
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        from flask import g

        token = _extract_token()
        if not token:
            return jsonify({"error": "Authentication required"}), 401

        try:
            payload = decode_token(token)
            user = user_store.get_by_id(payload["sub"])
            if not user:
                return jsonify({"error": "User not found"}), 401
            g.current_user = user
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired — please log in again"}), 401
        except jwt.InvalidTokenError as e:
            return jsonify({"error": f"Invalid token: {e}"}), 401

        return f(*args, **kwargs)
    return decorated


def optional_auth(f):
    """
    Like require_auth but doesn't block unauthenticated requests.
    Sets g.current_user = None if no valid token found.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        from flask import g

        g.current_user = None
        token = _extract_token()
        if token:
            try:
                payload = decode_token(token)
                g.current_user = user_store.get_by_id(payload["sub"])
            except jwt.InvalidTokenError:
                pass
        return f(*args, **kwargs)
    return decorated