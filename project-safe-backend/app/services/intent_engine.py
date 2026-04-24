"""
Intent Engine — Layer 2
Real implementation using:
  - OpenAI Chat Completions API  (zero-shot, no training needed)
  - Teammate NLP model (.pkl)    (if provided)
  - Keyword fallback             (offline / no API key)

Endpoint: https://api.openai.com/v1/chat/completions
Docs:     https://platform.openai.com/docs/guides/text-generation
"""
import json
import re
import httpx
from flask import current_app


# ── Classifier prompt ─────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert telephone fraud detection system. "
    "Your job is to classify call transcripts for coercion and scam signals. "
    "Always respond with valid JSON only — no extra text, no markdown."
)

USER_PROMPT_TEMPLATE = """
Analyse the following call transcript for telephone fraud signals.

Score from 0 (completely benign) to 100 (extreme coercion) based ONLY on:
  1. Financial demands: requests for gift cards, wire transfers, crypto, cash
  2. Urgency / threat language: "arrest", "lawsuit", "account suspended", "last chance"
  3. Authority impersonation: claiming to be IRS, FBI, Medicare, bank security
  4. Isolation tactics: "don't tell anyone", "keep this between us"
  5. Fear induction: creating panic, threatening consequences for inaction

Return ONLY this JSON object:
{{
  "intent_score": <integer 0-100>,
  "signals_found": ["signal1", "signal2"],
  "threat_category": "none" | "financial_coercion" | "authority_impersonation" | "urgency_threat" | "mixed",
  "reasoning": "<one sentence>"
}}

Transcript:
\"\"\"
{transcript}
\"\"\"
""".strip()


# ── Public entry point ────────────────────────────────────────────────
def analyze(transcript: str | None) -> float:
    """
    Returns an intent_score in [0, 100].
    0  = benign or empty
    100 = extreme coercion / scam
    """
    if not transcript or not transcript.strip():
        return 0.0

    # Priority 1: teammate's trained NLP model
    model = getattr(current_app, "intent_model", None)
    if model is not None:
        current_app.logger.info("[Intent] Using teammate NLP model")
        return _run_teammate_model(model, transcript)

    # Priority 2: OpenAI Chat API (real, zero-shot)
    openai_key = current_app.config.get("OPENAI_API_KEY", "")
    use_openai = current_app.config.get("USE_REAL_INTENT_MODEL", False)

    if use_openai and openai_key and openai_key != "your_openai_api_key_here":
        current_app.logger.info("[Intent] Using OpenAI Chat API")
        result = _run_openai_classifier(transcript, openai_key)
        if result is not None:
            return result

    # Priority 3: local keyword classifier (no external call)
    current_app.logger.info("[Intent] Using keyword classifier")
    return _run_keyword_classifier(transcript)


# ── Teammate model ────────────────────────────────────────────────────
def _run_teammate_model(model, transcript: str) -> float:
    """
    Calls your teammate's trained text classifier.
    Expected interface:
        model.predict([text])        -> float
        model.predict_proba([text])  -> ndarray  (uses column 1 = P(coercive))
    """
    try:
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba([transcript])[0][1] * 100)
        return float(model.predict([transcript])[0])
    except Exception as e:
        current_app.logger.error(f"[Intent] Teammate model failed: {e}")
        return _run_keyword_classifier(transcript)


# ── OpenAI Chat Completions ───────────────────────────────────────────
def _run_openai_classifier(transcript: str, api_key: str) -> float | None:
    """
    Calls the OpenAI Chat Completions API directly using httpx (no SDK dependency).

    Real endpoint: POST https://api.openai.com/v1/chat/completions
    Model: gpt-4o-mini (configurable via OPENAI_MODEL in .env)

    Returns the intent_score float, or None if the call fails so the
    caller can fall back to the keyword classifier.
    """
    model_name = current_app.config.get("OPENAI_MODEL", "gpt-4o-mini")
    endpoint = current_app.config.get(
        "OPENAI_ENDPOINT", "https://api.openai.com/v1/chat/completions"
    )

    payload = {
        "model": model_name,
        "temperature": 0,
        "max_tokens": 256,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(transcript=transcript[:3000]),
            },
        ],
    }

    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

        if response.status_code == 401:
            current_app.logger.error("[Intent] OpenAI: Invalid API key")
            return None
        if response.status_code == 429:
            current_app.logger.warning("[Intent] OpenAI: Rate limited")
            return None
        if response.status_code != 200:
            current_app.logger.error(
                f"[Intent] OpenAI HTTP {response.status_code}: {response.text[:200]}"
            )
            return None

        data = response.json()
        raw_content = data["choices"][0]["message"]["content"].strip()

        # Strip markdown code fences if model adds them
        raw_content = re.sub(r"^```(?:json)?\s*", "", raw_content)
        raw_content = re.sub(r"\s*```$", "", raw_content)

        parsed = json.loads(raw_content)
        score = float(parsed.get("intent_score", 0))
        signals = parsed.get("signals_found", [])
        category = parsed.get("threat_category", "none")

        score = max(0.0, min(100.0, score))

        current_app.logger.info(
            f"[Intent] OpenAI score={score} category={category} signals={signals}"
        )
        return score

    except json.JSONDecodeError as e:
        current_app.logger.error(f"[Intent] OpenAI response JSON parse failed: {e}")
        return None
    except httpx.TimeoutException:
        current_app.logger.error("[Intent] OpenAI request timed out after 20s")
        return None
    except (KeyError, IndexError) as e:
        current_app.logger.error(f"[Intent] OpenAI unexpected response shape: {e}")
        return None
    except Exception as e:
        current_app.logger.error(f"[Intent] OpenAI unexpected error: {e}")
        return None


# ── Keyword classifier (offline fallback) ─────────────────────────────
def _run_keyword_classifier(transcript: str) -> float:
    """
    Rule-based keyword classifier.
    No API, no model — runs entirely offline.
    Used when OpenAI is disabled or unavailable.

    Organised by threat tier so the score reflects severity, not just count.
    """
    text = transcript.lower()

    # Tier 1 — extreme coercion (each = +25 points)
    tier1 = [
        "wire transfer", "gift card", "send bitcoin", "send crypto",
        "you will be arrested", "arrest warrant", "federal agents",
        "do not hang up", "this is your final warning",
    ]
    # Tier 2 — strong signals (each = +14 points)
    tier2 = [
        "irs", "social security", "medicare", "bank account suspended",
        "legal action", "lawsuit", "deportation", "criminal charges",
        "your account has been compromised", "verify your identity immediately",
    ]
    # Tier 3 — moderate signals (each = +7 points)
    tier3 = [
        "urgent", "immediately", "right now", "do not tell",
        "keep this confidential", "suspend", "frozen", "unusual activity",
        "you owe", "outstanding balance", "limited time",
    ]

    score = 0.0
    score += sum(25.0 for kw in tier1 if kw in text)
    score += sum(14.0 for kw in tier2 if kw in text)
    score += sum(7.0  for kw in tier3 if kw in text)

    final = round(min(100.0, score), 1)
    current_app.logger.info(f"[Intent] Keyword classifier score={final}")
    return final