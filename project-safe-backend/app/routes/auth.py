from flask import Blueprint, request, jsonify, make_response, g
from flasgger import swag_from

from app import limiter
from app.models import user_store
from app.utils.auth_helpers import issue_token, require_auth

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/signup", methods=["POST"])
@limiter.limit("10 per minute")
def signup():
    """
    Register a new user account.
    ---
    tags:
      - Auth
    summary: Create a new account
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - name
            - email
            - password
          properties:
            name:
              type: string
              example: Arjun Kumar
            email:
              type: string
              example: arjun@example.com
            password:
              type: string
              example: password123
              minLength: 8
    responses:
      201:
        description: Account created successfully
        schema:
          type: object
          properties:
            user:
              type: object
              properties:
                id:
                  type: string
                name:
                  type: string
                email:
                  type: string
                plan:
                  type: string
                  example: free
                joinedAt:
                  type: string
                callsAnalyzed:
                  type: integer
                threatsBlocked:
                  type: integer
            token:
              type: string
              description: JWT bearer token
      409:
        description: Email already registered
      422:
        description: Validation failed
    """
    body = request.get_json(force=True) or {}
    name     = str(body.get("name", "")).strip()
    email    = str(body.get("email", "")).strip().lower()
    password = str(body.get("password", ""))

    errors = {}
    if not name:
        errors["name"] = "Name is required"
    if not email or "@" not in email:
        errors["email"] = "Valid email is required"
    if len(password) < 8:
        errors["password"] = "Password must be at least 8 characters"
    if errors:
        return jsonify({"error": "Validation failed", "fields": errors}), 422

    try:
        user = user_store.create(name, email, password)
    except ValueError as e:
        return jsonify({"error": str(e)}), 409

    token = issue_token(user["id"])
    response = make_response(
        jsonify({"user": user_store.safe_public(user), "token": token}), 201
    )
    _set_token_cookie(response, token)
    return response


@auth_bp.route("/login", methods=["POST"])
@limiter.limit("10 per minute")
def login():
    """
    Log in to an existing account.
    ---
    tags:
      - Auth
    summary: Authenticate and receive a JWT
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - email
            - password
          properties:
            email:
              type: string
              example: arjun@example.com
            password:
              type: string
              example: password123
    responses:
      200:
        description: Login successful
        schema:
          type: object
          properties:
            user:
              type: object
            token:
              type: string
              description: JWT bearer token
      401:
        description: Invalid email or password
    """
    body     = request.get_json(force=True) or {}
    email    = str(body.get("email", "")).strip().lower()
    password = str(body.get("password", ""))

    user = user_store.get_by_email(email)
    if not user or not user_store.verify_password(user, password):
        return jsonify({"error": "Invalid email or password"}), 401

    token = issue_token(user["id"])
    response = make_response(
        jsonify({"user": user_store.safe_public(user), "token": token}), 200
    )
    _set_token_cookie(response, token)
    return response


@auth_bp.route("/me", methods=["GET"])
@require_auth
def me():
    """
    Get the currently authenticated user's profile.
    ---
    tags:
      - Auth
    summary: Get current user
    security:
      - BearerAuth: []
    responses:
      200:
        description: User profile returned
        schema:
          type: object
          properties:
            user:
              type: object
      401:
        description: Missing or invalid token
    """
    return jsonify({"user": user_store.safe_public(g.current_user)}), 200


@auth_bp.route("/logout", methods=["POST"])
def logout():
    """
    Log out — clears the auth cookie.
    ---
    tags:
      - Auth
    summary: Log out
    responses:
      200:
        description: Logged out successfully
        schema:
          type: object
          properties:
            message:
              type: string
              example: Logged out successfully
    """
    response = make_response(jsonify({"message": "Logged out successfully"}), 200)
    response.delete_cookie("safe_token", samesite="Lax")
    return response


def _set_token_cookie(response, token: str) -> None:
    response.set_cookie(
        "safe_token", token,
        httponly=True,
        secure=False,
        samesite="Lax",
        max_age=60 * 60 * 24,
    )