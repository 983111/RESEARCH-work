code = '''
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, classification_report, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from feature_extractor import FEATURE_NAMES

SEED = 42
np.random.seed(SEED)

def load_data(path="scam_dataset.csv"):
    df = pd.read_csv(path)
    X = df.drop("label", axis=1)
    y = df["label"]
    print("Dataset: {} samples | Scam: {} | Safe: {}".format(len(df), int(y.sum()), int((y==0).sum())))
    return X, y

def cross_validate_models(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced", random_state=SEED))
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=SEED, n_jobs=-1))
        ]),
        "Gradient Boosting": Pipeline([
            ("clf", GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, min_samples_leaf=5, subsample=0.8, random_state=SEED))
        ]),
    }
    print("\\n-- 5-Fold Cross-Validation --")
    results = {}
    for name, pipeline in models.items():
        f1  = cross_val_score(pipeline, X, y, cv=cv, scoring="f1", n_jobs=-1)
        rec = cross_val_score(pipeline, X, y, cv=cv, scoring="recall", n_jobs=-1)
        auc = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        results[name] = {"f1_mean": round(f1.mean(),4), "recall_mean": round(rec.mean(),4), "auc_mean": round(auc.mean(),4)}
        print("  {:<22}  F1={:.3f}+/-{:.3f}  Recall={:.3f}+/-{:.3f}  AUC={:.3f}+/-{:.3f}".format(
            name, f1.mean(), f1.std(), rec.mean(), rec.std(), auc.mean(), auc.std()))
    return results

def train_best_model(X_train, y_train):
    print("\\n-- Training Final Model --")
    base = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, min_samples_leaf=4, subsample=0.8, random_state=SEED)
    model = CalibratedClassifierCV(base, method="isotonic", cv=5)
    model.fit(X_train, y_train)
    print("   Done.")
    return model

def find_optimal_threshold(model, X_test, y_test, min_recall=0.90):
    y_prob = model.predict_proba(X_test)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
    best_f1, best_thresh = 0, 0.5
    for p, r, t in zip(prec, rec, thresholds):
        if r >= min_recall:
            f1 = 2 * p * r / (p + r + 1e-9)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
    print("\\n-- Optimal Threshold: {:.3f} --".format(best_thresh))
    return round(best_thresh, 3)

def evaluate(model, X_test, y_test, threshold):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    print("\\n-- Test Results (threshold={}) --".format(threshold))
    print("  Accuracy:  {:.4f}".format(acc))
    print("  Precision: {:.4f}".format(p))
    print("  Recall:    {:.4f}  <- security priority".format(r))
    print("  F1-Score:  {:.4f}".format(f))
    print("  ROC-AUC:   {:.4f}".format(auc))
    print("  Confusion Matrix: TN={}  FP={}  FN={}  TP={}".format(cm[0,0], cm[0,1], cm[1,0], cm[1,1]))
    print(classification_report(y_test, y_pred, target_names=["Safe","Scam"]))
    return {"precision": round(p,4), "recall": round(r,4), "f1": round(f,4), "accuracy": round(acc,4), "roc_auc": round(auc,4)}

def feature_importance_analysis(X_train, y_train):
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, min_samples_leaf=4, subsample=0.8, random_state=SEED)
    gb.fit(X_train, y_train)
    importances = dict(zip(FEATURE_NAMES, gb.feature_importances_.tolist()))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print("\\n-- Feature Importances --")
    for name, imp in sorted_imp[:12]:
        print("  {:<25} {:.4f}".format(name, imp))
    return importances

def main():
    os.makedirs("models", exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    print("Train: {} | Test: {}".format(len(X_train), len(X_test)))
    cv_results = cross_validate_models(X, y)
    final_model = train_best_model(X_train, y_train)
    threshold = find_optimal_threshold(final_model, X_test, y_test, min_recall=0.90)
    metrics = evaluate(final_model, X_test, y_test, threshold)
    importances = feature_importance_analysis(X_train, y_train)
    joblib.dump(final_model, "scam_detector_model.pkl")
    metadata = {"threshold": threshold, "features": FEATURE_NAMES, "n_features": len(FEATURE_NAMES),
                "test_metrics": metrics, "feature_importances": importances, "cv_results": cv_results,
                "model_type": "CalibratedClassifierCV(GradientBoosting, isotonic)"}
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("\\nDone! Model saved -> scam_detector_model.pkl")
    print("Threshold for inference: {}".format(threshold))

if __name__ == "__main__":
    main()
'''

with open('train_model.py', 'w', encoding='utf-8') as f:
    f.write(code.strip())

print("train_model.py created successfully!")