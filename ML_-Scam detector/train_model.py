"""
train_model.py
──────────────
Trains a Random Forest classifier on the StreminiAI balanced dataset.
Automatically finds the CSV whether you run from the project root or src/.
"""

import pathlib
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ── Auto-locate the dataset CSV ──────────────────────────────────────────────
_CANDIDATES = [
    "dataset_balanced.csv",                    # project root  ← your case
    "data/processed/dataset_balanced.csv",     # root/data/processed/
    "../data/processed/dataset_balanced.csv",  # src/
]

DATA_PATH = next(
    (p for p in _CANDIDATES if pathlib.Path(p).exists()),
    _CANDIDATES[0]   # fallback so error message shows a real path
)

# Save model next to wherever the dataset lives
_data_dir  = pathlib.Path(DATA_PATH).resolve().parent
MODEL_DIR  = _data_dir.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = str(MODEL_DIR / "scam_detector.pkl")

# ── All possible feature columns ─────────────────────────────────────────────
ALL_FEATURE_COLS = [
    "url_length", "domain_length", "path_length", "query_length",
    "num_dots", "num_hyphens", "num_digits", "num_special_chars",
    "has_ip", "is_https", "has_php", "has_html", "has_exe",
]

LABEL_CANDIDATES = ["label", "is_malicious", "class", "target", "y"]


def find_label_col(df: pd.DataFrame) -> str:
    for col in LABEL_CANDIDATES:
        if col in df.columns:
            return col
    for col in df.columns:
        unique = set(df[col].dropna().unique())
        if unique.issubset({0, 1, "0", "1"}):
            print(f"   ℹ️  Using '{col}' as label column (binary values detected).")
            return col
    raise ValueError(
        f"Could not find a label column. Tried: {LABEL_CANDIDATES}\n"
        f"Available columns: {list(df.columns)}"
    )


def train():
    print("🚀 Loading dataset …")
    print(f"   Path: {pathlib.Path(DATA_PATH).resolve()}")

    if not pathlib.Path(DATA_PATH).exists():
        print(
            f"\n❌ Dataset not found.\n"
            f"   Looked in: {[str(pathlib.Path(p).resolve()) for p in _CANDIDATES]}\n\n"
            f"   Option A — generate locally:\n"
            f"       python make_scam_urls.py && python add_benign_data.py\n\n"
            f"   Option B — use the Kaggle dataset:\n"
            f"       python download_kaggle_dataset.py\n\n"
            f"   Option C — place your CSV in the project root and rename it:\n"
            f"       dataset_balanced.csv"
        )
        return

    df = pd.read_csv(DATA_PATH)
    print(f"   Columns   : {list(df.columns)}")
    print(f"   Total rows: {len(df):,}")

    # ── Label column ──────────────────────────────────────────────────────────
    label_col = find_label_col(df)
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[label_col])
    df[label_col] = df[label_col].astype(int)

    print(f"   Label col : '{label_col}'")
    print(f"   Malicious : {(df[label_col] == 1).sum():,}")
    print(f"   Benign    : {(df[label_col] == 0).sum():,}")

    # ── Feature columns ───────────────────────────────────────────────────────
    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]

    # Include any extra numeric columns the Kaggle dataset might have
    non_feature = {label_col, "url", "threat", "tags", "urlhaus_link",
                   "reporter", "id", "dateadded", "url_status", "last_online"}
    extra = [
        c for c in df.columns
        if c not in non_feature and c not in feature_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if extra:
        print(f"   ➕ Extra numeric columns (included): {extra}")
        feature_cols += extra

    if not feature_cols:
        print("❌ No usable numeric feature columns found.")
        return

    print(f"\n📐 Features ({len(feature_cols)}): {feature_cols}")
    df[feature_cols] = df[feature_cols].fillna(0)

    X = df[feature_cols]
    y = df[label_col]

    # ── Split ─────────────────────────────────────────────────────────────────
    print("\n📊 Splitting data …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train):,}   Test: {len(X_test):,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n🧠 Training Random Forest (n_estimators=200, n_jobs=-1) …")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    print("\n" + "=" * 40)
    print(f"🏆  MODEL ACCURACY: {acc:.4%}")
    print("=" * 40)
    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, target_names=["Benign (0)", "Malicious (1)"]
    ))

    print("📌 Feature Importances:")
    for feat, score in sorted(zip(feature_cols, model.feature_importances_),
                               key=lambda x: x[1], reverse=True):
        bar = "█" * int(score * 50)
        print(f"   {feat:<22} {score:.4f}  {bar}")

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_PATH)
    print(f"\n💾 Model saved to {MODEL_PATH}")
    print("   ✅ Done! Run  python predict_url.py  to test it.")


if __name__ == "__main__":
    train()