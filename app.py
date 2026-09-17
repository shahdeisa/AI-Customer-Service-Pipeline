"""
app.py — Flask Deployment for the RAG-Based E-commerce Customer Support Chatbot

This script wires together the 4 modules built in Notebooks 1-4:
  1. Language Detection    (language_tfidf_vectorizer.joblib, language_classifier.joblib,
                             language_label_encoder.joblib)
  2. Sentiment Classifier  (sentiment_model/  — fine-tuned distilbert)
  3. Intent Classifier     (intent_tfidf_vectorizer.joblib, intent_classifier.joblib)
  4. RAG Pipeline          (rag_faiss.index, rag_documents.joblib, sentence-transformers, Groq)

--------------------------------------------------------------------------------------------
INSTALL
--------------------------------------------------------------------------------------------
    pip install flask joblib scikit-learn torch transformers sentence-transformers \
                faiss-cpu groq numpy pandas

--------------------------------------------------------------------------------------------
CONFIGURATION
--------------------------------------------------------------------------------------------
Set these environment variables before running:
    export GROQ_API_KEY="your-groq-api-key"
    export ARTIFACTS_DIR="/path/to/RAG_chatbot_project"   # folder with all saved artifacts
                                                            # (defaults to ./RAG_chatbot_project)

--------------------------------------------------------------------------------------------
RUN
--------------------------------------------------------------------------------------------
    python app.py
    # Server starts on http://0.0.0.0:5000

--------------------------------------------------------------------------------------------
API
--------------------------------------------------------------------------------------------
POST /chat
    Request body:  {"message": "customer text"}
    Response body: {
        "detected_language": "en",
        "sentiment": "negative",
        "intent": "order_status",
        "routed_action": "rag" | "greeting" | "escalate",
        "bot_response": "..."
    }

GET /health
    Simple liveness check.
"""

import os
import re
import logging
from typing import Optional

import joblib
import numpy as np
from flask import Flask, request, jsonify

# ------------------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chatbot_app")

# ------------------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------------------
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "./RAG_chatbot_project")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "3"))

if not GROQ_API_KEY:
    logger.warning(
        "GROQ_API_KEY is not set. The RAG pipeline will fall back to a safe default "
        "response instead of calling the LLM. Set GROQ_API_KEY before deploying to production."
    )

SENTIMENT_LABELS = ["negative", "neutral", "positive"]

# ------------------------------------------------------------------------------------------
# Module 1: Language Detection — load artifacts
# ------------------------------------------------------------------------------------------
try:
    lang_vectorizer = joblib.load(os.path.join(ARTIFACTS_DIR, "language_tfidf_vectorizer.joblib"))
    lang_classifier = joblib.load(os.path.join(ARTIFACTS_DIR, "language_classifier.joblib"))
    lang_label_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "language_label_encoder.joblib"))
    logger.info("Loaded language detection artifacts.")
except Exception as e:
    logger.error(f"Failed to load language detection artifacts: {e}")
    lang_vectorizer = lang_classifier = lang_label_encoder = None


def _basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def detect_language(text: str) -> dict:
    """Stage 1: detect the language of the incoming message."""
    if lang_vectorizer is None or lang_classifier is None:
        return {"language": "unknown", "confidence": 0.0}

    cleaned = _basic_clean(text)
    if not cleaned:
        return {"language": "unknown", "confidence": 0.0}

    try:
        vec = lang_vectorizer.transform([cleaned])
        proba = lang_classifier.predict_proba(vec)[0]
        pred_idx = int(np.argmax(proba))
        lang_code = lang_label_encoder.inverse_transform([pred_idx])[0]
        return {"language": lang_code, "confidence": round(float(proba[pred_idx]), 4)}
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return {"language": "unknown", "confidence": 0.0}


# ------------------------------------------------------------------------------------------
# Module 2: Sentiment Classifier — load artifacts
# ------------------------------------------------------------------------------------------
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    SENTIMENT_MODEL_DIR = os.path.join(ARTIFACTS_DIR, "sentiment_classifier")
    sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_DIR)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_DIR)
    sentiment_device = "cuda" if torch.cuda.is_available() else "cpu"
    sentiment_model.to(sentiment_device)
    sentiment_model.eval()
    logger.info(f"Loaded sentiment model on {sentiment_device}.")
except Exception as e:
    logger.error(f"Failed to load sentiment model: {e}")
    sentiment_tokenizer = sentiment_model = None
    sentiment_device = "cpu"


def classify_sentiment(text: str) -> dict:
    """Stage 2: classify sentiment into negative/neutral/positive."""
    if sentiment_model is None or sentiment_tokenizer is None:
        return {"sentiment": "neutral", "confidence": 0.0}

    if not text or not text.strip():
        return {"sentiment": "neutral", "confidence": 0.0}

    try:
        with torch.no_grad():
            inputs = sentiment_tokenizer(
                text, truncation=True, max_length=128, return_tensors="pt"
            ).to(sentiment_device)
            logits = sentiment_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        return {
            "sentiment": SENTIMENT_LABELS[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
        }
    except Exception as e:
        logger.error(f"Sentiment classification failed: {e}")
        return {"sentiment": "neutral", "confidence": 0.0}


# ------------------------------------------------------------------------------------------
# Module 3: Intent Classifier — load artifacts
# ------------------------------------------------------------------------------------------
try:
    intent_vectorizer = joblib.load(os.path.join(ARTIFACTS_DIR, "intent_tfidf_vectorizer.joblib"))
    intent_classifier = joblib.load(os.path.join(ARTIFACTS_DIR, "intent_classifier.joblib"))
    logger.info("Loaded intent classifier artifacts.")
except Exception as e:
    logger.error(f"Failed to load intent classifier artifacts: {e}")
    intent_vectorizer = intent_classifier = None

# See Notebook 3 for why this pre-filter exists: the Bitext training data contains
# no greeting/goodbye/gratitude examples, so the trained model can never predict that class.
GREETING_PATTERNS = re.compile(
    r"^\s*("
    r"hi|hello|hey|good (morning|afternoon|evening)|"
    r"bye|goodbye|see you|take care|"
    r"thanks|thank you|thx|appreciate it|much appreciated"
    r")\b",
    re.IGNORECASE,
)


def classify_intent(text: str) -> dict:
    """Stage 3: route the message into one of 7 macro-intent categories."""
    if not text or not text.strip():
        return {"intent": "out_of_scope", "confidence": 0.0, "source": "rule"}

    stripped = text.strip()
    if GREETING_PATTERNS.match(stripped) and len(stripped.split()) <= 6:
        return {"intent": "greeting/goodbye/gratitude", "confidence": 1.0, "source": "rule"}

    if intent_vectorizer is None or intent_classifier is None:
        return {"intent": "out_of_scope", "confidence": 0.0, "source": "fallback"}

    try:
        cleaned = _basic_clean(text).lower()
        vec = intent_vectorizer.transform([cleaned])
        proba = intent_classifier.predict_proba(vec)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = intent_classifier.classes_[pred_idx]
        return {
            "intent": pred_label,
            "confidence": round(float(proba[pred_idx]), 4),
            "source": "model",
        }
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return {"intent": "out_of_scope", "confidence": 0.0, "source": "fallback"}


# ------------------------------------------------------------------------------------------
# Module 4: RAG Pipeline — load artifacts
# ------------------------------------------------------------------------------------------
try:
    import faiss
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from groq import Groq

    rag_index = faiss.read_index(os.path.join(ARTIFACTS_DIR, "rag_faiss.index"))
    rag_documents = joblib.load(os.path.join(ARTIFACTS_DIR, "rag_documents.joblib"))
    rag_embedder = SentenceTransformer("all-MiniLM-L6-v2")
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
    logger.info(f"Loaded RAG artifacts. Index size: {rag_index.ntotal}")
except Exception as e:
    logger.error(f"Failed to load RAG artifacts: {e}")
    rag_index = rag_documents = rag_embedder = groq_client = None

SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful, professional customer support assistant for an online retailer. "
    "Answer the customer's question using ONLY the information in the retrieved support "
    "responses below. If the customer sounds frustrated ({detected_sentiment}), acknowledge "
    "that before answering. If the retrieved context does not cover the question, say so "
    "honestly and offer to escalate to a human agent rather than guessing."
)

SAFE_FALLBACK_ANSWER = (
    "I'm sorry, I'm having trouble reaching our answer service right now. "
    "Let me connect you with a human agent who can help."
)


def _retrieve(query: str, k: int = RAG_TOP_K) -> list:
    if rag_index is None or rag_embedder is None or rag_documents is None:
        return []
    try:
        query_emb = rag_embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = rag_index.search(query_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            row = rag_documents.iloc[idx]
            results.append({
                "instruction": row["instruction"],
                "response": row["response"],
                "score": float(score),
            })
        return results
    except Exception as e:
        logger.error(f"RAG retrieval failed: {e}")
        return []


def _format_context(chunks: list) -> str:
    if not chunks:
        return "(no relevant context found)"
    return "\n".join(
        f"{i}. Q: {c['instruction']}\n   A: {c['response']}"
        for i, c in enumerate(chunks, start=1)
    )


def run_rag(user_message: str, detected_sentiment: str) -> dict:
    """Stage 4: retrieve grounding context and query the LLM."""
    retrieved_chunks = _retrieve(user_message, k=RAG_TOP_K)

    if groq_client is None:
        logger.warning("Groq client unavailable (missing API key or failed init); using fallback answer.")
        return {"answer": SAFE_FALLBACK_ANSWER, "retrieved_chunks": retrieved_chunks}

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(detected_sentiment=detected_sentiment)
    user_prompt = (
        f"Context:\n{_format_context(retrieved_chunks)}\n\n"
        f'Customer question: "{user_message}"'
    )

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        answer = SAFE_FALLBACK_ANSWER

    return {"answer": answer, "retrieved_chunks": retrieved_chunks}


# ------------------------------------------------------------------------------------------
# Canned responses for direct (non-RAG) routes
# ------------------------------------------------------------------------------------------
GREETING_RESPONSE = "Hello! Thanks for reaching out — how can I help you with your order today?"
GOODBYE_THANKS_RESPONSE = "You're very welcome! Have a great day, and don't hesitate to reach out again."

COMPLAINT_APOLOGY_PREFIX = (
    "I'm really sorry to hear about this experience — that's not what we want for our "
    "customers, and I understand the frustration. "
)
ESCALATION_NOTICE = (
    " I've flagged this conversation for a human agent to follow up with you directly."
)


def build_greeting_response(text: str) -> str:
    """Distinguish a greeting/hello from a goodbye/thanks for a slightly more natural reply."""
    lowered = text.strip().lower()
    if re.match(r"^(bye|goodbye|see you|take care|thanks|thank you|thx|appreciate)", lowered):
        return GOODBYE_THANKS_RESPONSE
    return GREETING_RESPONSE


# ------------------------------------------------------------------------------------------
# Flask app + routing logic
# ------------------------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True)
    if not payload or "message" not in payload:
        return jsonify({"error": "Request body must be JSON with a 'message' field."}), 400

    user_message = payload["message"]
    if not isinstance(user_message, str) or not user_message.strip():
        return jsonify({"error": "'message' must be a non-empty string."}), 400

    # --- Stage 1: Language detection ---
    lang_result = detect_language(user_message)

    # --- Stage 2: Sentiment classification ---
    sentiment_result = classify_sentiment(user_message)

    # --- Stage 3: Intent classification ---
    intent_result = classify_intent(user_message)

    intent = intent_result["intent"]
    sentiment = sentiment_result["sentiment"]

    # --- Stage 4: Conditional routing ---
    try:
        if intent == "greeting/goodbye/gratitude":
            routed_action = "greeting"
            bot_response = build_greeting_response(user_message)

        elif intent == "complaint" or sentiment == "negative":
            # Design choice: acknowledge + escalate rather than let the LLM freehand a
            # response to an upset customer. Still runs RAG so the escalation includes
            # any directly relevant info the agent/customer can use in the meantime.
            routed_action = "escalate"
            rag_result = run_rag(user_message, detected_sentiment=sentiment)
            bot_response = (
                COMPLAINT_APOLOGY_PREFIX + rag_result["answer"] + ESCALATION_NOTICE
            )

        else:
            routed_action = "rag"
            rag_result = run_rag(user_message, detected_sentiment=sentiment)
            bot_response = rag_result["answer"]

    except Exception as e:
        logger.error(f"Unhandled error during routing: {e}")
        routed_action = "escalate"
        bot_response = (
            "I'm sorry, something went wrong on our end. "
            "I've flagged this for a human agent to follow up with you."
        )

    response_body = {
        "detected_language": lang_result["language"],
        "sentiment": sentiment,
        "intent": intent,
        "routed_action": routed_action,
        "bot_response": bot_response,
    }

    logger.info(f"Processed message. intent={intent}, sentiment={sentiment}, routed_action={routed_action}")
    return jsonify(response_body), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
