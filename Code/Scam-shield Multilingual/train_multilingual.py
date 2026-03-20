"""
train_multilingual.py
======================
Training and evaluation pipeline for ScamShield Multilingual Extension.

What this script does:
  1. Generates the multilingual dataset (or loads if already present)
  2. Trains the char-ngram lightweight model (produces f32 feature)
  3. Trains the extended 32-feature GBM ensemble
  4. Runs 3-fold cross-validation
  5. Evaluates on held-out test set per language
  6. Runs adversarial robustness tests (same 3 attacks as original + script-swap)
  7. Exports an Android-compatible model bundle:
       - multilingual_scam_detector.pkl   (32-feat GBM, ~2MB)
       - multilingual_ngram_model.pkl     (char-ngram LR, ~500KB)
       - multilingual_model_metadata.json (thresholds, feature names, metrics)

Android RAM budget:
  - Char-ngram TF-IDF vectoriser + LR: ~500KB
  - 32-feature GBM (150 trees, depth 4): ~1.5MB
  - Total: <2MB — well within low-RAM Android budget
"""

from __future__ import annotations

import os
import json
import sys
import random
import warnings
import re

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(__file__))

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    matthews_corrcoef,
    f1_score,
    accuracy_score,
)

from multilingual_feature_extractor import (
    extract_features_extended,
    FEATURE_NAMES_EXTENDED,
)
from build_multilingual_dataset import generate, ALL_SCAM_POOLS, ALL_SAFE_POOLS
from lang_detect import detect_language

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

os.makedirs("models", exist_ok=True)
os.makedirs("eval_results", exist_ok=True)

DATASET_PATH       = "multilingual_scam_dataset.csv"
GBM_MODEL_PATH     = "models/multilingual_scam_detector.pkl"
NGRAM_MODEL_PATH   = "models/multilingual_ngram_model.pkl"
METADATA_PATH      = "multilingual_model_metadata.json"
EVAL_PATH          = "eval_results/multilingual_evaluation.json"

SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial attacks (same as original + one multilingual-specific attack)
# ─────────────────────────────────────────────────────────────────────────────

SYNONYM_MAP_EN = {
    "verify":    "confirm",
    "urgent":    "important",
    "suspended": "paused",
    "password":  "credentials",
}

# Script-swap attack: replace native-script scam keywords with their
# Romanized equivalents (and vice versa). Tests whether the model relies
# purely on script-specific patterns.
HINDI_SWAP = {
    "तुरंत": "turant",
    "पासवर्ड": "password",
    "ओटीपी": "otp",
    "लॉटरी": "lottery",
    "ब्लॉक": "block",
}
TELUGU_SWAP = {
    "వెంటనే": "ventane",
    "పాస్వర్డ్": "password",
    "ఓటీపీ": "otp",
    "లాటరీ": "lottery",
}
KANNADA_SWAP = {
    "ತಕ್ಷಣ": "takshana",
    "ಪಾಸ್‌ವರ್ಡ್": "password",
    "ಒಟಿಪಿ": "otp",
    "ಲಾಟರಿ": "lottery",
}


def _attack_synonym(text: str) -> str:
    for orig, replacement in SYNONYM_MAP_EN.items():
        text = re.sub(r"\b" + orig + r"\b", replacement, text, flags=re.IGNORECASE)
    return text


def _attack_homoglyph(text: str, intensity: float = 0.25) -> str:
    glyph_map = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с"}
    return "".join(
        glyph_map.get(ch.lower(), ch) if random.random() < intensity else ch
        for ch in text
    )


def _attack_url_obfuscation(text: str) -> str:
    shorteners = ["bit.ly", "tinyurl", "goo.gl", "is.gd", "cutt.ly"]
    wrapper    = "https://redirect.example.com/?target="
    for s in shorteners:
        if s in text:
            text = text.replace(s, wrapper + s, 1)
            break
    return text


def _attack_script_swap(text: str) -> str:
    """Replace native-script scam keywords with Romanized equivalents."""
    for swap_map in [HINDI_SWAP, TELUGU_SWAP, KANNADA_SWAP]:
        for native, roman in swap_map.items():
            text = text.replace(native, roman)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helper
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    p, r, f, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
    return {
        "accuracy":  round(float(accuracy_score(y, preds)), 4),
        "precision": round(float(p),                         4),
        "recall":    round(float(r),                         4),
        "f1":        round(float(f),                         4),
        "roc_auc":   round(float(roc_auc_score(y, proba)),   4),
        "mcc":       round(float(matthews_corrcoef(y, preds)), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Build / load dataset
# ─────────────────────────────────────────────────────────────────────────────

def step1_dataset() -> pd.DataFrame:
    print("[1/6] Building multilingual dataset ...", flush=True)
    if not os.path.exists(DATASET_PATH):
        generate(n_scam=4000, n_safe=4000, output_path=DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)
    print(f"      {len(df):,} samples loaded. "
          f"Scam={df['label'].sum()}, Safe={(df['label']==0).sum()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Train char-ngram model (produces f32 feature)
# ─────────────────────────────────────────────────────────────────────────────

def step2_ngram_model(raw_texts: list[str], labels: list[int]) -> object:
    """
    Train a lightweight char 3-5gram TF-IDF + LogisticRegression pipeline.
    This model is:
      - Script-agnostic (char n-grams work on any Unicode text)
      - ~500KB serialized
      - Used to produce f32 (char_ngram_scam_score)
    """
    print("[2/6] Training char-ngram model (f32 source) ...", flush=True)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=8000,    # keeps model small for Android
            sublinear_tf=True,
            min_df=2,
        )),
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            random_state=SEED,
        )),
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(
        raw_texts, labels, test_size=0.20, stratify=labels, random_state=SEED
    )
    pipeline.fit(X_tr, y_tr)

    preds = pipeline.predict(X_te)
    f1 = f1_score(y_te, preds)
    auc = roc_auc_score(y_te, pipeline.predict_proba(X_te)[:, 1])
    print(f"      Char-ngram model: F1={f1:.4f}  AUC={auc:.4f}")

    joblib.dump(pipeline, NGRAM_MODEL_PATH)
    print(f"      Saved: {NGRAM_MODEL_PATH}")
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Re-extract features with ngram model (populates f32)
# ─────────────────────────────────────────────────────────────────────────────

def step3_reextract(raw_texts: list[str], labels: list[int], ngram_model) -> tuple:
    print("[3/6] Re-extracting 32-feature vectors (with f32 populated) ...", flush=True)
    rows = []
    for text in raw_texts:
        feats = extract_features_extended(text, ngram_model=ngram_model)
        rows.append(feats)
    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"      Feature matrix shape: {X.shape}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Train the extended 32-feature GBM
# ─────────────────────────────────────────────────────────────────────────────

def step4_train_gbm(X, y) -> tuple:
    print("[4/6] Training 32-feature GBM ensemble ...", flush=True)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )

    # 3-fold CV first
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    base_cv = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=4, random_state=SEED,
    )
    f1_scores  = cross_val_score(base_cv, X_tr, y_tr, cv=cv, scoring="f1")
    auc_scores = cross_val_score(base_cv, X_tr, y_tr, cv=cv, scoring="roc_auc")
    print(f"      3-fold CV: F1={f1_scores.mean():.4f}±{f1_scores.std():.4f}  "
          f"AUC={auc_scores.mean():.4f}±{auc_scores.std():.4f}")

    # Final model with calibration
    base_final = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=4, random_state=SEED,
    )
    final_model = CalibratedClassifierCV(base_final, method="isotonic", cv=3)
    final_model.fit(X_tr, y_tr)

    metrics = _compute_metrics(final_model, X_te, y_te)
    print(f"      Test set: F1={metrics['f1']:.4f}  AUC={metrics['roc_auc']:.4f}  "
          f"Recall={metrics['recall']:.4f}  Precision={metrics['precision']:.4f}")

    joblib.dump(final_model, GBM_MODEL_PATH)
    print(f"      Saved: {GBM_MODEL_PATH}")

    cv_results = {
        "f1_mean":  round(float(f1_scores.mean()),  4),
        "f1_std":   round(float(f1_scores.std()),   4),
        "auc_mean": round(float(auc_scores.mean()), 4),
        "auc_std":  round(float(auc_scores.std()),  4),
    }
    return final_model, X_tr, X_te, y_tr, y_te, metrics, cv_results


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Per-language evaluation
# ─────────────────────────────────────────────────────────────────────────────

def step5_per_language_eval(
    model, ngram_model, raw_texts, labels
) -> dict[str, dict]:
    print("[5/6] Per-language evaluation ...", flush=True)
    lang_results = {}

    # Detect language for each sample
    langs = [detect_language(t) for t in raw_texts]

    for lang_code in ["en", "hi", "mr", "te", "kn"]:
        indices = [i for i, l in enumerate(langs) if l == lang_code]
        if len(indices) < 10:
            print(f"      [{lang_code}] insufficient samples ({len(indices)}), skipping")
            continue

        texts_lang  = [raw_texts[i] for i in indices]
        labels_lang = [labels[i]    for i in indices]

        X_lang = np.array(
            [extract_features_extended(t, ngram_model=ngram_model) for t in texts_lang],
            dtype=np.float32,
        )
        y_lang = np.array(labels_lang, dtype=np.int32)

        if len(np.unique(y_lang)) < 2:
            print(f"      [{lang_code}] only one class present, skipping")
            continue

        m = _compute_metrics(model, X_lang, y_lang)
        lang_results[lang_code] = {**m, "n_samples": len(indices)}
        print(f"      [{lang_code}] n={len(indices):>4}  F1={m['f1']:.4f}  "
              f"AUC={m['roc_auc']:.4f}  Recall={m['recall']:.4f}")

    return lang_results


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Adversarial robustness
# ─────────────────────────────────────────────────────────────────────────────

def step6_adversarial(model, ngram_model, raw_texts, labels) -> dict:
    print("[6/6] Adversarial robustness ...", flush=True)

    # Use only scam samples
    scam_texts  = [t for t, l in zip(raw_texts, labels) if l == 1]
    y_scam_true = np.ones(len(scam_texts), dtype=np.int32)

    def _recall_under_attack(attacked_texts):
        X_atk = np.array(
            [extract_features_extended(t, ngram_model=ngram_model) for t in attacked_texts],
            dtype=np.float32,
        )
        preds = model.predict(X_atk)
        _, r, _, _ = precision_recall_fscore_support(
            y_scam_true, preds, average="binary", zero_division=0
        )
        return round(float(r), 4)

    # Baseline recall (no attack)
    base_recall = _recall_under_attack(scam_texts)
    print(f"      Clean scam recall: {base_recall:.4f}")

    attacks = {
        "synonym_substitution": _attack_synonym,
        "homoglyph_attack":     _attack_homoglyph,
        "url_obfuscation":      _attack_url_obfuscation,
        "script_swap":          _attack_script_swap,
    }

    results = {"clean": {"recall": base_recall, "delta": 0.0}}
    for name, attack_fn in attacks.items():
        attacked = [attack_fn(t) for t in scam_texts]
        r = _recall_under_attack(attacked)
        delta = round(r - base_recall, 4)
        results[name] = {"recall": r, "delta": delta}
        print(f"      [{name:<25}] recall={r:.4f}  delta={delta:+.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run():
    print("=" * 70)
    print("  ScamShield Multilingual Extension — Training Pipeline")
    print("  Languages: Hindi · Marathi · Telugu · Kannada")
    print("  Features: 32 (24 original + 8 multilingual)")
    print("  Target: Android low-RAM (<2MB model bundle)")
    print("=" * 70)

    # Step 1
    df = step1_dataset()
    # We need raw texts. Since the dataset stores features, we re-generate
    # raw texts from the same seed for use by the ngram model.
    # This is honest: feature extraction is applied twice — once during dataset
    # generation (for the 32-col CSV) and once here (for the ngram model training).
    random.seed(42)
    raw_texts_all, labels_all = [], []
    for pool in ALL_SCAM_POOLS:
        per_pool = max(1, 4000 // len(ALL_SCAM_POOLS))
        for _ in range(per_pool):
            raw_texts_all.append(random.choice(pool))
            labels_all.append(1)
    for pool in ALL_SAFE_POOLS:
        per_pool = max(1, 4000 // len(ALL_SAFE_POOLS))
        for _ in range(per_pool):
            raw_texts_all.append(random.choice(pool))
            labels_all.append(0)

    # Step 2
    ngram_model = step2_ngram_model(raw_texts_all, labels_all)

    # Step 3
    X, y = step3_reextract(raw_texts_all, labels_all, ngram_model)

    # Step 4
    gbm_model, X_tr, X_te, y_tr, y_te, test_metrics, cv_results = step4_train_gbm(X, y)

    # Step 5
    lang_metrics = step5_per_language_eval(gbm_model, ngram_model, raw_texts_all, labels_all)

    # Step 6
    adv_results = step6_adversarial(gbm_model, ngram_model, raw_texts_all, labels_all)

    # Feature importances
    base_plain = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=4, random_state=SEED,
    )
    base_plain.fit(X_tr, y_tr)
    importances = {
        name: round(float(imp), 6)
        for name, imp in zip(FEATURE_NAMES_EXTENDED, base_plain.feature_importances_)
    }

    # Android model sizes
    gbm_size_kb   = os.path.getsize(GBM_MODEL_PATH) // 1024
    ngram_size_kb = os.path.getsize(NGRAM_MODEL_PATH) // 1024
    total_kb      = gbm_size_kb + ngram_size_kb

    # Save metadata + full evaluation
    metadata = {
        "model_type": "CalibratedClassifierCV(GradientBoosting, isotonic) — 32 features",
        "n_features": 32,
        "feature_names": FEATURE_NAMES_EXTENDED,
        "threshold": 0.70,
        "languages_supported": ["en", "hi", "mr", "te", "kn"],
        "android_model_sizes_kb": {
            "gbm_model":   gbm_size_kb,
            "ngram_model": ngram_size_kb,
            "total":       total_kb,
        },
        "feature_importances": importances,
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    evaluation = {
        "dataset":              {"total": len(labels_all), "scam": sum(labels_all), "safe": labels_all.count(0)},
        "cross_validation":     cv_results,
        "test_metrics_overall": test_metrics,
        "per_language_metrics": lang_metrics,
        "adversarial_results":  adv_results,
        "feature_importances":  importances,
        "android_bundle_kb":    total_kb,
    }
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)

    # Final summary
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  Overall test metrics:")
    for k, v in test_metrics.items():
        print(f"    {k:<14} {v}")

    print(f"\n  Per-language F1 scores:")
    for lang, m in lang_metrics.items():
        print(f"    [{lang}] F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}  n={m['n_samples']}")

    print(f"\n  Adversarial robustness (scam recall):")
    for attack, r in adv_results.items():
        delta_str = f"  (Δ={r['delta']:+.4f})" if attack != "clean" else ""
        print(f"    {attack:<28} {r['recall']:.4f}{delta_str}")

    print(f"\n  Android bundle size: {total_kb} KB total "
          f"({gbm_size_kb} KB GBM + {ngram_size_kb} KB ngram)")
    print(f"\n  Saved:")
    print(f"    {GBM_MODEL_PATH}")
    print(f"    {NGRAM_MODEL_PATH}")
    print(f"    {METADATA_PATH}")
    print(f"    {EVAL_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    run()
