"""
Intent Engine — Layer 2
NLP / coercion and urgency detection.

Analyses the transcript of the call for:
  - Financial coercion signals ("send money now", "gift cards", "wire transfer")
  - Urgency / threat patterns ("arrest", "lawsuit", "suspend your account")
  - Authority impersonation ("IRS", "FBI", "bank security")
  - Emotional manipulation cues (fear, panic induction)

── PLUGGING IN YOUR TEAMMATE'S MODEL ─────────────────────────────────────
Option A — Trained NLP model (.pkl)
    Your teammate saves a text classifier:
        joblib.dump(pipeline, "app/models/weights/intent_model.pkl")
    Set USE_REAL_INTENT_MODEL=1 and INTENT_MODEL_PATH in .env.

Option B — OpenAI classifier (zero-shot, no training needed)
    Set USE_REAL_INTENT_MODEL=0 but set OPENAI_API_KEY in .env.
    The _run_openai_classifier() function below will be called instead.
    This is the fastest path to a working intent layer.
──────────────────────────────────────────────────────────────────────────
"""
from flask import current_app


COERCION_PROMPT = """
You are a fraud detection classifier for telephone calls.
Analyse the following call transcript for coercion signals.

Score the transcript from 0 to 100 based on ONLY these factors:
  - Financial demands or urgency (gift cards, wire transfers, crypto payments)
  - Threats of arrest, legal action, or account suspension
  - Impersonation of government, banks, or law enforcement
  - Emotional manipulation: creating panic, fear, or isolation

Return ONLY a JSON object with this exact shape and nothing else:
{"intent_score": <number 0-100>, "signals": ["signal1", "signal2"]}

Transcript:
\"\"\"{transcript}\"\"\"
"""


def analyze(transcript: str | None) -> float:
    """
    Returns an intent_score in [0, 100].
    0  = benign or harmless content
    100 = strong coercion / scam language
    """
    if not transcript or transcript.strip() == "":
        return 0.0

    model = getattr(current_app, "intent_model", None)

    if model is not None:
        return _run_local_model(model, transcript)

    # Try OpenAI if key is present
    openai_key = current_app.config.get("OPENAI_API_KEY", "")
    if openai_key and openai_key != "your_openai_api_key_here":
        return _run_openai_classifier(transcript, openai_key)

    # Pure mock fallback
    return _mock_intent_score(transcript)


def _run_local_model(model, transcript: str) -> float:
    """
    Runs your teammate's trained NLP classifier on the transcript text.

    ── EXPECTED INTERFACE ──────────────────────────────────────────────
    model.predict([transcript]) -> float or np.ndarray
    OR
    model.predict_proba([transcript]) -> np.ndarray  (probabilities)
    ────────────────────────────────────────────────────────────────────
    """
    try:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba([transcript])[0][1]  # P(coercive)
            return float(prob * 100)
        else:
            return float(model.predict([transcript])[0])
    except Exception as e:
        current_app.logger.error(f"IntentEngine.predict failed: {e}")
        return _mock_intent_score(transcript)


def _run_openai_classifier(transcript: str, api_key: str) -> float:
    """
    Zero-shot coercion classifier using the OpenAI chat API.
    No training data needed — useful for rapid prototyping.

    ── TO ENABLE ────────────────────────────────────────────────────────
    Set OPENAI_API_KEY in .env (leave USE_REAL_INTENT_MODEL=0).
    The model used is controlled by OPENAI_MODEL in .env.
    ─────────────────────────────────────────────────────────────────────
    """
    import json
    from openai import OpenAI, OpenAIError

    try:
        client = OpenAI(api_key=api_key)
        model_name = current_app.config.get("OPENAI_MODEL", "gpt-4o-mini")

        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a telephone fraud detection system. Always respond with valid JSON only.",
                },
                {
                    "role": "user",
                    "content": COERCION_PROMPT.format(transcript=transcript),
                },
            ],
        )

        raw = response.choices[0].message.content or ""
        data = json.loads(raw)
        score = float(data.get("intent_score", 0))
        signals = data.get("signals", [])

        current_app.logger.info(f"OpenAI intent score: {score} | signals: {signals}")
        return max(0.0, min(100.0, score))

    except (OpenAIError, json.JSONDecodeError, KeyError, ValueError) as e:
        current_app.logger.warning(f"OpenAI intent classifier failed: {e}. Using mock.")
        return _mock_intent_score(transcript)


def _mock_intent_score(transcript: str | None) -> float:
    """
    Keyword-based mock classifier. Not for production use.
    Provides realistic scores for frontend development without any API calls.
    """
    if not transcript:
        return 5.0

    coercion_keywords = [
        "arrest", "warrant", "lawsuit", "legal action", "irs", "fbi",
        "wire transfer", "gift card", "bitcoin", "crypto", "suspend",
        "your account", "send money", "immediately", "urgent", "deadline",
        "do not tell", "keep this secret", "bank security", "verify now",
    ]

    text_lower = transcript.lower()
    hits = sum(1 for kw in coercion_keywords if kw in text_lower)

    if hits == 0:
        return round(5.0 + (len(transcript) % 15), 1)
    elif hits <= 2:
        return round(25.0 + hits * 10, 1)
    elif hits <= 5:
        return round(55.0 + hits * 5, 1)
    else:
        return round(min(95.0, 70.0 + hits * 3), 1)