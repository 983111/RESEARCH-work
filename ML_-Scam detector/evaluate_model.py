"""
Model Evaluation Script for Scam Detection System
Generates all metrics, plots, and data needed for the evaluation report.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score,
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    classification_report, matthews_corrcoef, brier_score_loss
)
import joblib

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
os.makedirs("eval_figures", exist_ok=True)

# ── Load & split ───────────────────────────────────────────────────────────────
df = pd.read_csv("scam_dataset.csv")
X = df.drop("label", axis=1)
y = df["label"]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# ── Train ──────────────────────────────────────────────────────────────────────
base = LogisticRegression(max_iter=1000, random_state=SEED)
model = CalibratedClassifierCV(base, method="sigmoid", cv=5)
model.fit(X_train, y_train)
joblib.dump(model, "scam_detector_model.pkl")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ══════════════════════════════════════════════════════════════════════════════
# 1. CORE METRICS
# ══════════════════════════════════════════════════════════════════════════════
p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
ap  = average_precision_score(y_test, y_prob)
mcc = matthews_corrcoef(y_test, y_pred)
brier = brier_score_loss(y_test, y_prob)

print("=" * 50)
print("CORE EVALUATION METRICS")
print("=" * 50)
print(f"Accuracy:          {acc:.4f}")
print(f"Precision:         {p:.4f}")
print(f"Recall:            {r:.4f}")
print(f"F1-Score:          {f:.4f}")
print(f"ROC-AUC:           {auc:.4f}")
print(f"Avg Precision:     {ap:.4f}")
print(f"MCC:               {mcc:.4f}")
print(f"Brier Score:       {brier:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=["Safe", "Scam"]))

# Per-class metrics
p_per, r_per, f_per, sup = precision_recall_fscore_support(y_test, y_pred)
print("\nPer-Class Metrics:")
for i, cls in enumerate(["Safe", "Scam"]):
    print(f"  {cls}: P={p_per[i]:.3f}  R={r_per[i]:.3f}  F1={f_per[i]:.3f}  Support={sup[i]}")

# ── Cross-validation ────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
base2 = LogisticRegression(max_iter=1000, random_state=SEED)
cv_acc   = cross_val_score(base2, X, y, cv=cv, scoring='accuracy')
cv_f1    = cross_val_score(base2, X, y, cv=cv, scoring='f1')
cv_auc   = cross_val_score(base2, X, y, cv=cv, scoring='roc_auc')
cv_prec  = cross_val_score(base2, X, y, cv=cv, scoring='precision')
cv_rec   = cross_val_score(base2, X, y, cv=cv, scoring='recall')

print("\n5-Fold Cross-Validation:")
for name, scores in [("Accuracy", cv_acc), ("F1", cv_f1), ("AUC", cv_auc),
                      ("Precision", cv_prec), ("Recall", cv_rec)]:
    print(f"  {name}: {scores.mean():.4f} ± {scores.std():.4f}  (folds: {np.round(scores,3)})")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Safe', 'Scam'], yticklabels=['Safe', 'Scam'],
            linewidths=1, linecolor='white', ax=ax,
            annot_kws={"size": 18, "weight": "bold"})
ax.set_xlabel('Predicted Label', fontsize=13, labelpad=10)
ax.set_ylabel('True Label', fontsize=13, labelpad=10)
ax.set_title('Confusion Matrix', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("eval_figures/fig1_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Figure 1: Confusion Matrix saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Per-Class Metrics Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
metrics_labels = ['Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics_labels))
w = 0.3
bars1 = ax.bar(x - w/2, [p_per[0], r_per[0], f_per[0]], w, label='Safe', color='#4CAF50', alpha=0.85)
bars2 = ax.bar(x + w/2, [p_per[1], r_per[1], f_per[1]], w, label='Scam', color='#F44336', alpha=0.85)
for b in list(bars1) + list(bars2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
            f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylim(0, 1.15)
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels, fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Per-Class Classification Metrics', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.4)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("eval_figures/fig2_per_class_metrics.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure 2: Per-Class Metrics saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – ROC Curve
# ══════════════════════════════════════════════════════════════════════════════
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color='#1565C0', lw=2.5, label=f'ROC Curve (AUC = {auc:.4f})')
ax.fill_between(fpr, tpr, alpha=0.08, color='#1565C0')
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("eval_figures/fig3_roc_curve.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure 3: ROC Curve saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Precision-Recall Curve
# ══════════════════════════════════════════════════════════════════════════════
prec_curve, rec_curve, thresholds = precision_recall_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec_curve, prec_curve, color='#7B1FA2', lw=2.5,
        label=f'PR Curve (AP = {ap:.4f})')
ax.fill_between(rec_curve, prec_curve, alpha=0.08, color='#7B1FA2')

# Mark current operating point
ax.scatter([r], [p], color='red', zorder=5, s=120,
           label=f'Operating Point\nP={p:.3f}, R={r:.3f}')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("eval_figures/fig4_pr_curve.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure 4: Precision-Recall Curve saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 – Score Distribution
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
safe_probs = y_prob[y_test == 0]
scam_probs = y_prob[y_test == 1]
ax.hist(safe_probs, bins=30, alpha=0.65, color='#43A047', label='Safe (True Label)')
ax.hist(scam_probs, bins=30, alpha=0.65, color='#E53935', label='Scam (True Label)')
ax.axvline(0.5, color='gray', linestyle='--', lw=2, label='Decision Threshold (0.5)')
ax.set_xlabel('Predicted Scam Probability', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Predicted Score Distribution by True Class', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("eval_figures/fig5_score_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure 5: Score Distribution saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 – Feature Importance (from base LR coefficients)
# ══════════════════════════════════════════════════════════════════════════════
# Re-train plain LR to get interpretable coefficients
lr = LogisticRegression(max_iter=1000, random_state=SEED)
lr.fit(X_train, y_train)
coefs = lr.coef_[0]

feature_labels = [
<<<<<<< HEAD
    'has_urgency', 'has_money', 'has_sensitive', 'has_off_platform',
    'has_threat', 'has_legitimacy_marker',
    'text_length', 'exclamation_count', 'question_count',
    'uppercase_ratio', 'digit_ratio', 'char_entropy', 'avg_word_length',
    'punctuation_density',
    'urgency_density', 'money_density', 'sensitive_density',
    'num_urls', 'url_density',
    'ip_url', 'url_shortener', 'risky_tld', 'domain_spoof', 'verified_domain'
=======
    'Urgency (f1)', 'Money (f2)', 'Sensitive (f3)', 'Off-Platform (f4)',
    'Text Length (f5)', 'Exclamations (f6)', 'Uppercase Ratio (f7)', 'Digit Ratio (f8)',
    'Num URLs (f9)', 'URL Ratio (f10)', 'IP URL (f11)', 'URL Shortener (f12)',
    'Risky TLD (f13)', 'Domain Spoof (f14)', 'Verified Domain (f15)', 'Manual Score (f16)'
>>>>>>> fe065bb089ed369fb5a44cc368a6ed27630f21da
]

sorted_idx = np.argsort(coefs)
sorted_coefs = coefs[sorted_idx]
sorted_labels = [feature_labels[i] for i in sorted_idx]
colors = ['#E53935' if c > 0 else '#43A047' for c in sorted_coefs]

fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(range(len(sorted_labels)), sorted_coefs, color=colors, alpha=0.85, edgecolor='white')
ax.set_yticks(range(len(sorted_labels)))
ax.set_yticklabels(sorted_labels, fontsize=10)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Logistic Regression Coefficient', fontsize=12)
ax.set_title('Feature Importance (LR Coefficients)', fontsize=14, fontweight='bold')
red_patch  = mpatches.Patch(color='#E53935', alpha=0.85, label='Scam indicator (positive)')
green_patch = mpatches.Patch(color='#43A047', alpha=0.85, label='Safe indicator (negative)')
ax.legend(handles=[red_patch, green_patch], fontsize=10, loc='lower right')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("eval_figures/fig6_feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure 6: Feature Importance saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 – Cross-Validation Results
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
cv_metrics = {'Accuracy': cv_acc, 'F1': cv_f1, 'AUC': cv_auc, 'Precision': cv_prec, 'Recall': cv_rec}
positions = np.arange(len(cv_metrics))
bp = ax.boxplot([v for v in cv_metrics.values()], positions=positions,
                patch_artist=True, widths=0.45, notch=False)
colors_box = ['#1565C0', '#7B1FA2', '#E65100', '#2E7D32', '#880E4F']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for median in bp['medians']:
    median.set_color('white')
    median.set_linewidth(2)
ax.set_xticks(positions)
ax.set_xticklabels(cv_metrics.keys(), fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('5-Fold Cross-Validation Performance', fontsize=14, fontweight='bold')
ax.set_ylim(0.6, 1.05)
ax.axhline(1.0, color='gray', linestyle='--', alpha=0.4)
ax.grid(axis='y', alpha=0.3)
for i, (name, scores) in enumerate(cv_metrics.items()):
    ax.text(i, scores.min() - 0.02, f'μ={scores.mean():.3f}', ha='center',
            fontsize=9, color='black')
plt.tight_layout()
plt.savefig("eval_figures/fig7_cross_validation.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure 7: Cross-Validation saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 – Calibration Curve
# ══════════════════════════════════════════════════════════════════════════════
fraction_of_pos, mean_predicted = calibration_curve(y_test, y_prob, n_bins=10)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0,1],[0,1], 'k--', label='Perfect Calibration', lw=1.5)
ax.plot(mean_predicted, fraction_of_pos, 's-', color='#1565C0', lw=2.5, ms=8,
        label=f'Model (Brier={brier:.4f})')
ax.fill_between(mean_predicted, fraction_of_pos, mean_predicted,
                alpha=0.1, color='#1565C0')
ax.set_xlabel('Mean Predicted Probability', fontsize=12)
ax.set_ylabel('Fraction of Positives', fontsize=12)
ax.set_title('Calibration Curve (Reliability Diagram)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("eval_figures/fig8_calibration.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure 8: Calibration Curve saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 – Threshold Analysis
# ══════════════════════════════════════════════════════════════════════════════
thresholds_arr = np.linspace(0.01, 0.99, 200)
prec_t, rec_t, f1_t, acc_t = [], [], [], []
for t in thresholds_arr:
    preds = (y_prob >= t).astype(int)
    if preds.sum() == 0:
        prec_t.append(0); rec_t.append(0); f1_t.append(0); acc_t.append(0)
        continue
    pi, ri, fi, _ = precision_recall_fscore_support(y_test, preds, average='binary', zero_division=0)
    prec_t.append(pi); rec_t.append(ri); f1_t.append(fi)
    acc_t.append(accuracy_score(y_test, preds))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(thresholds_arr, prec_t, label='Precision', color='#1565C0', lw=2)
ax.plot(thresholds_arr, rec_t, label='Recall', color='#E53935', lw=2)
ax.plot(thresholds_arr, f1_t, label='F1-Score', color='#7B1FA2', lw=2)
ax.plot(thresholds_arr, acc_t, label='Accuracy', color='#43A047', lw=2, linestyle='--')
ax.axvline(0.5, color='gray', linestyle=':', lw=1.5, label='Default Threshold (0.5)')
ax.set_xlabel('Decision Threshold', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Metric vs. Decision Threshold', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='center right')
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("eval_figures/fig9_threshold_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure 9: Threshold Analysis saved")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE NUMERIC RESULTS FOR REPORT
# ══════════════════════════════════════════════════════════════════════════════
results = {
    "dataset_total": len(df),
    "dataset_scam": int(y.sum()),
    "dataset_safe": int((y == 0).sum()),
    "train_size": len(X_train),
    "test_size": len(X_test),
    "accuracy": round(acc, 4),
    "precision": round(p, 4),
    "recall": round(r, 4),
    "f1": round(f, 4),
    "roc_auc": round(auc, 4),
    "avg_precision": round(ap, 4),
    "mcc": round(mcc, 4),
    "brier": round(brier, 4),
    "cv_acc_mean": round(cv_acc.mean(), 4),
    "cv_acc_std": round(cv_acc.std(), 4),
    "cv_f1_mean": round(cv_f1.mean(), 4),
    "cv_f1_std": round(cv_f1.std(), 4),
    "cv_auc_mean": round(cv_auc.mean(), 4),
    "cv_auc_std": round(cv_auc.std(), 4),
    "cv_prec_mean": round(cv_prec.mean(), 4),
    "cv_prec_std": round(cv_prec.std(), 4),
    "cv_rec_mean": round(cv_rec.mean(), 4),
    "cv_rec_std": round(cv_rec.std(), 4),
    "cm_tn": int(cm[0,0]), "cm_fp": int(cm[0,1]),
    "cm_fn": int(cm[1,0]), "cm_tp": int(cm[1,1]),
    "safe_precision": round(p_per[0], 4), "safe_recall": round(r_per[0], 4), "safe_f1": round(f_per[0], 4),
    "scam_precision": round(p_per[1], 4), "scam_recall": round(r_per[1], 4), "scam_f1": round(f_per[1], 4),
    "coefs": dict(zip(feature_labels, [round(c, 4) for c in coefs]))
}

import json
with open("eval_results.json", "w") as f_out:
    json.dump(results, f_out, indent=2)

print("\n✓ Results saved to eval_results.json")
print("\n=== EVALUATION COMPLETE ===")
