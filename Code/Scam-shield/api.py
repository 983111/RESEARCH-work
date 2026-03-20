from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from make_scam_urls import extract_features

app = FastAPI(title="Stremini AI - Scam Detector API")

MODEL_PATH = "../models/scam_detector.pkl"

# ── Load model ───────────────────────────────────────────────────────────────
model        = None
feature_cols = None

DEFAULT_COLS = [
    "url_length", "domain_length", "path_length", "query_length",
    "num_dots", "num_hyphens", "num_digits", "num_special_chars",
    "has_ip", "is_https", "has_php", "has_html", "has_exe",
]

try:
    payload = joblib.load(MODEL_PATH)
    if isinstance(payload, dict):
        model        = payload["model"]
        feature_cols = payload["feature_cols"]
    else:
        model        = payload          # old bare-model format
        feature_cols = DEFAULT_COLS
    print(f"✅ Model loaded  |  features: {feature_cols}")
except Exception as e:
    print(f"❌ Model not loaded: {e}")


class URLRequest(BaseModel):
    url: str


@app.get("/")
def home():
    return {"status": "online", "system": "Stremini Scam Detector"}


@app.post("/scan")
def scan_url(request: URLRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    features = extract_features(request.url)
    if not features:
        raise HTTPException(status_code=400, detail="Invalid URL format")

    df = pd.DataFrame([features])

    # Fill missing cols (handles Kaggle-extra columns) with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    X           = df[feature_cols]
    is_scam     = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    return {
        "url"             : request.url,
        "is_scam"         : bool(is_scam),
        "confidence_score": round(probability, 4),
        "risk_level"      : "CRITICAL" if probability > 0.8
                            else "SAFE" if probability < 0.5
                            else "SUSPICIOUS",
    }

# Run with: uvicorn api:app --reload