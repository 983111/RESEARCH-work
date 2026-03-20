"""
ML API v2.0 — Upgraded inference service
  - Uses calibrated probability + configurable threshold
  - Returns explainable feature contributions
  - No manual_score dependency
"""
import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from feature_extractor import extract_features, FEATURE_NAMES

app = Flask(__name__)

model = joblib.load('scam_detector_model.pkl')

with open('model_metadata.json') as f:
    metadata = json.load(f)

THRESHOLD = metadata.get('threshold', 0.4)
IMPORTANCES = metadata.get('feature_importances', {})


def get_top_signals(features, n=5):
    """Return top N features that are active (non-zero) and most important."""
    signals = []
    for name, val in zip(FEATURE_NAMES, features):
        if val != 0 and val != 0.0:
            importance = IMPORTANCES.get(name, 0)
            signals.append({'feature': name, 'value': round(float(val), 4),
                            'importance': round(importance, 4)})
    signals.sort(key=lambda x: x['importance'], reverse=True)
    return signals[:n]


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    content = data.get('content', '')
    if not content:
        return jsonify({'error': 'content field is required'}), 400

    features = extract_features(content)
    X = pd.DataFrame([features], columns=FEATURE_NAMES)
    prob = model.predict_proba(X)[0][1]
    is_scam = prob >= THRESHOLD

    if prob >= 0.75:
        verdict = 'scam'
    elif prob >= THRESHOLD:
        verdict = 'suspicious'
    elif prob >= 0.25:
        verdict = 'suspicious'
    else:
        verdict = 'safe'

    return jsonify({
        'verdict': verdict,
        'probability': round(float(prob), 4),
        'risk_score': round(float(prob) * 100),
        'threshold_used': THRESHOLD,
        'top_signals': get_top_signals(features),
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': metadata.get('model_type'),
                    'n_features': metadata.get('n_features'),
                    'threshold': THRESHOLD})


if __name__ == '__main__':
    app.run(port=5000, debug=False)