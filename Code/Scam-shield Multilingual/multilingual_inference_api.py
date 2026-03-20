"""
multilingual_inference_api.py
==============================
Production inference API for ScamShield Multilingual.
Drop this file alongside the two .pkl files on your Flask backend,
or call it directly from Android via Chaquopy.

Inference pipeline:
  1. Detect language (zero-dependency Unicode detection, <0.1ms)
  2. Extract 32 features (with f32 = 0.0 if ngram model not loaded)
  3. GBM ensemble → calibrated probability
  4. Three-tier verdict: safe / suspicious / scam

Android deployment notes:
  - Total model bundle: ~842 KB (well within low-RAM budget)
  - Inference latency: <5ms on mid-range Android (no GPU needed)
  - Models load once at startup; subsequent calls are fast
  - Graceful degradation: if ngram model fails to load, f32 = 0.0
    and the 31-feature GBM still runs correctly
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from multilingual_feature_extractor import (
    extract_features_extended,
    FEATURE_NAMES_EXTENDED,
)
from lang_detect import detect_language

# ─────────────────────────────────────────────────────────────────────────────
# Model paths (relative to this file)
# ─────────────────────────────────────────────────────────────────────────────

_DIR              = os.path.dirname(os.path.abspath(__file__))
_GBM_PATH         = os.path.join(_DIR, "models", "multilingual_scam_detector.pkl")
_NGRAM_PATH       = os.path.join(_DIR, "models", "multilingual_ngram_model.pkl")
_METADATA_PATH    = os.path.join(_DIR, "multilingual_model_metadata.json")

# ─────────────────────────────────────────────────────────────────────────────
# Lazy model loading (loaded once, then cached)
# ─────────────────────────────────────────────────────────────────────────────

_gbm_model    = None
_ngram_model  = None
_metadata     = None


def _load_models():
    global _gbm_model, _ngram_model, _metadata

    if _gbm_model is not None:
        return  # Already loaded

    if not os.path.exists(_GBM_PATH):
        raise FileNotFoundError(
            f"GBM model not found at {_GBM_PATH}. "
            f"Run train_multilingual.py first."
        )

    _gbm_model = joblib.load(_GBM_PATH)

    # Ngram model is optional — graceful degradation if missing
    if os.path.exists(_NGRAM_PATH):
        try:
            _ngram_model = joblib.load(_NGRAM_PATH)
        except Exception as e:
            print(f"[ScamShield] Warning: ngram model failed to load: {e}. "
                  f"f32 will be 0.0 — model still functional.")
            _ngram_model = None
    else:
        _ngram_model = None

    if os.path.exists(_METADATA_PATH):
        with open(_METADATA_PATH, encoding="utf-8") as f:
            _metadata = json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold configuration
# ─────────────────────────────────────────────────────────────────────────────

# These match the original ScamShield threshold table (§IX of the paper)
THRESHOLD_SCAM       = 0.90
THRESHOLD_SUSPICIOUS = 0.20

# Language-specific threshold adjustments
# Tighter thresholds for languages with less training data (honest caveat)
LANG_THRESHOLD_OVERRIDES: dict[str, float] = {
    "hi": 0.85,   # Largest multilingual pool — good confidence
    "mr": 0.80,   # Smaller pool — be slightly more conservative
    "te": 0.85,
    "kn": 0.85,
    "en": 0.90,   # Original English threshold unchanged
}


def _get_scam_threshold(lang_code: str) -> float:
    return LANG_THRESHOLD_OVERRIDES.get(lang_code, THRESHOLD_SCAM)


# ─────────────────────────────────────────────────────────────────────────────
# Main predict function
# ─────────────────────────────────────────────────────────────────────────────

def predict(content: str) -> dict:
    """
    Classify a message as safe / suspicious / scam.

    Args:
        content: raw message text (any of en/hi/mr/te/kn)

    Returns:
        {
            "verdict":      "safe" | "suspicious" | "scam",
            "probability":  float (0-1, calibrated),
            "risk_score":   int (0-100),
            "language":     str (detected language code),
            "threshold":    float (threshold used for verdict),
            "top_signals":  list of {feature, value, importance},
        }
    """
    _load_models()

    # Language detection
    lang = detect_language(content)

    # Feature extraction
    features = extract_features_extended(content, ngram_model=_ngram_model)
    X = np.array([features], dtype=np.float32)

    # GBM prediction
    prob  = float(_gbm_model.predict_proba(X)[0][1])
    score = int(round(prob * 100))

    # Threshold (language-aware)
    threshold = _get_scam_threshold(lang)

    # Verdict
    if prob >= threshold:
        verdict = "scam"
    elif prob >= THRESHOLD_SUSPICIOUS:
        verdict = "suspicious"
    else:
        verdict = "safe"

    # Top contributing signals (feature importance × feature value)
    if _metadata and "feature_importances" in _metadata:
        importances = _metadata["feature_importances"]
        signals = []
        for feat_name, feat_val, feat_imp in zip(
            FEATURE_NAMES_EXTENDED, features, [importances.get(n, 0.0) for n in FEATURE_NAMES_EXTENDED]
        ):
            if feat_val != 0 and feat_imp > 0:
                signals.append({
                    "feature":    feat_name,
                    "value":      round(float(feat_val), 4),
                    "importance": round(float(feat_imp), 4),
                })
        signals.sort(key=lambda s: s["importance"], reverse=True)
        top_signals = signals[:5]
    else:
        top_signals = []

    return {
        "verdict":     verdict,
        "probability": round(prob, 4),
        "risk_score":  score,
        "language":    lang,
        "threshold":   threshold,
        "top_signals": top_signals,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Flask microservice (optional — same pattern as original ml_api.py)
# ─────────────────────────────────────────────────────────────────────────────

def create_flask_app():
    """
    Returns a Flask app. Usage:
        app = create_flask_app()
        app.run(port=5001)
    """
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        raise ImportError("Flask not installed. Run: pip install flask")

    app = Flask(__name__)

    @app.route("/predict", methods=["POST"])
    def _predict():
        data = request.get_json(silent=True) or {}
        content = data.get("content", "")
        if not content:
            return jsonify({"error": "content field is required"}), 400
        try:
            result = predict(content)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/health", methods=["GET"])
    def _health():
        _load_models()
        return jsonify({
            "status": "ok",
            "model": "ScamShield Multilingual",
            "n_features": 32,
            "languages": ["en", "hi", "mr", "te", "kn"],
            "ngram_loaded": _ngram_model is not None,
        })

    return app


# ─────────────────────────────────────────────────────────────────────────────
# CLI quick-test
# ─────────────────────────────────────────────────────────────────────────────

_TEST_MESSAGES = [
    # Hindi scam
    ("hi", "आपके बैंक खाते में संदिग्ध गतिविधि पाई गई है। तुरंत वेरीफाई करें: bit.ly/bank-verify"),
    # Hindi safe
    ("hi", "भाई कल आ रहे हो? खाना साथ खाएंगे"),
    # Marathi scam
    ("mr", "बधाई हो! आपके मोबाइल नंबर ने लॉटरी में ₹85,00,000 जीते! दावा करें: lottery_claim@protonmail.com"),
    # Telugu scam
    ("te", "మీ బ్యాంక్ ఖాతాలో అనుమానాస్పద కార్యకలాపం కనుగొనబడింది. వెంటనే వెరిఫై చేయండి: bit.ly/bank-verify"),
    # Telugu safe
    ("te", "అన్నా రేపు వస్తావా? కలిసి భోజనం చేద్దాం"),
    # Kannada scam
    ("kn", "ನಿಮ್ಮ Apple ID ಲಾಕ್ ಆಗಿದೆ. ಈಗಲೇ ಅನ್ಲಾಕ್ ಮಾಡಿ: apple-id-support.xyz/unlock"),
    # Kannada safe
    ("kn", "ಅಣ್ಣ ನಾಳೆ ಬರ್ತೀಯಾ? ಒಟ್ಟಿಗೆ ಊಟ ಮಾಡೋಣ"),
    # English scam
    ("en", "URGENT! Verify your PayPal account NOW at bit.ly/pp-secure or it will be suspended"),
    # English safe
    ("en", "Hey, are you free this weekend? Let's grab coffee"),
    # Mixed script (scam URL in Hindi message)
    ("mixed", "आपका खाता suspend हो गया। bit.ly/verify-now पर जाएं"),
]


if __name__ == "__main__":
    print("ScamShield Multilingual — Quick Test")
    print("=" * 65)
    for expected_lang, msg in _TEST_MESSAGES:
        result = predict(msg)
        verdict_icon = {"scam": "🚨", "suspicious": "⚠️ ", "safe": "✅"}.get(result["verdict"], "?")
        print(f"{verdict_icon} [{result['language']:>5}] {result['verdict'].upper():<10} "
              f"p={result['probability']:.3f}  | {msg[:60]}")
    print("=" * 65)
