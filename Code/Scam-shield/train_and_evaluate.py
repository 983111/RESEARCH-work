"""
train_and_evaluate.py  — FAST VERSION
======================================
All 5 research fixes, optimised for speed:
  Fix 1 - Large dataset (10,496 samples)
  Fix 2 - Baseline model comparison (NB, LR, RF, LinearSVC, GBM)
  Fix 3 - Adversarial robustness experiments
  Fix 4 - McNemar statistical significance testing
  Fix 5 - Full results saved to eval_results/full_evaluation.json
"""

import os, json, random, warnings, re
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score,
    roc_auc_score, average_precision_score,
    matthews_corrcoef, brier_score_loss, confusion_matrix
)
from scipy.stats import chi2 as chi2_dist

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.makedirs('models',       exist_ok=True)
os.makedirs('eval_results', exist_ok=True)

from feature_extractor import extract_features, FEATURE_NAMES


# =============================================================================
# ADVERSARIAL ATTACK FUNCTIONS  (Fix 3)
# =============================================================================

SYNONYM_MAP = {
    'verify':      ['confirm', 'validate', 'authenticate'],
    'urgent':      ['immediate', 'critical', 'important'],
    'account':     ['profile', 'wallet', 'membership'],
    'suspended':   ['locked', 'restricted', 'disabled'],
    'immediately': ['now', 'at once', 'right away'],
    'password':    ['credentials', 'passcode', 'access code'],
    'prize':       ['reward', 'gift', 'award'],
    'winner':      ['recipient', 'selected person'],
    'free':        ['complimentary', 'no-cost', 'gratis'],
}

HOMOGLYPH_MAP = {'a': 'a', 'e': 'e', 'o': '0', 'p': 'p', 'c': 'c', 'i': '1'}
SHORT_URL_PATTERNS = ['bit.ly', 'tinyurl.com', 'goo.gl', 'is.gd', 'cutt.ly']


def attack_synonym(text):
    for word, synonyms in SYNONYM_MAP.items():
        pat = re.compile(re.escape(word), re.IGNORECASE)
        if pat.search(text):
            text = pat.sub(random.choice(synonyms), text, count=1)
    return text


def attack_homoglyph(text, intensity=0.25):
    return ''.join(
        HOMOGLYPH_MAP.get(ch.lower(), ch) if random.random() < intensity else ch
        for ch in text
    )


def attack_url_obfuscation(text):
    wrappers = [
        'https://redirect.example.com/?target=',
        'https://www.google.com/url?sa=t&url=',
    ]
    for p in SHORT_URL_PATTERNS:
        if p in text:
            text = text.replace(p, random.choice(wrappers) + p, 1)
            break
    return text


# =============================================================================
# McNEMAR'S SIGNIFICANCE TEST  (Fix 4)
# =============================================================================

def mcnemar_test(y_true, pred_a, pred_b):
    b01 = int(np.sum((pred_a == y_true) & (pred_b != y_true)))
    b10 = int(np.sum((pred_a != y_true) & (pred_b == y_true)))
    n   = b01 + b10
    if n == 0:
        return 0.0, 1.0
    chi2_val = (abs(b01 - b10) - 1) ** 2 / n
    p_val    = 1 - chi2_dist.cdf(chi2_val, df=1)
    return round(float(chi2_val), 4), round(float(p_val), 6)


# =============================================================================
# HELPERS
# =============================================================================

def row_to_text(row):
    parts = []
    if row['has_urgency']:           parts.append('urgent verify now action required immediately')
    if row['has_money']:             parts.append('free money lottery prize earn income reward')
    if row['has_sensitive']:         parts.append('password cvv pin otp social security bank account')
    if row['has_off_platform']:      parts.append('telegram whatsapp contact dm')
    if row['has_threat']:            parts.append('account suspended deleted blocked security alert')
    if row['has_legitimacy_marker']: parts.append('documentation meeting schedule report regards team')
    if row['url_shortener']:         parts.append('bit.ly tinyurl click here link')
    if row['risky_tld']:             parts.append('xyz tk ml ga pw')
    if row['domain_spoof']:          parts.append('paypal amazon google apple secure login')
    if row['verified_domain']:       parts.append('github google youtube amazon')
    if row['exclamation_count'] > 0: parts.append('!!!')
    return ' '.join(parts) if parts else 'normal message'


def get_metrics(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    p, r, f, _ = precision_recall_fscore_support(y, preds, average='binary', zero_division=0)
    cm = confusion_matrix(y, preds)
    return {
        'accuracy':      round(float(accuracy_score(y, preds)),         4),
        'precision':     round(float(p),                                 4),
        'recall':        round(float(r),                                 4),
        'f1':            round(float(f),                                 4),
        'roc_auc':       round(float(roc_auc_score(y, proba)),           4),
        'avg_precision': round(float(average_precision_score(y, proba)), 4),
        'mcc':           round(float(matthews_corrcoef(y, preds)),        4),
        'brier':         round(float(brier_score_loss(y, proba)),         4),
        'cm_tn': int(cm[0, 0]), 'cm_fp': int(cm[0, 1]),
        'cm_fn': int(cm[1, 0]), 'cm_tp': int(cm[1, 1]),
    }, preds


def run_cv(model, X, y, cv, label):
    print(f"      [{label}] ...', flush=True", flush=True)
    f1  = cross_val_score(model, X, y, cv=cv, scoring='f1',       n_jobs=1)
    rec = cross_val_score(model, X, y, cv=cv, scoring='recall',    n_jobs=1)
    auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc',   n_jobs=1)
    pre = cross_val_score(model, X, y, cv=cv, scoring='precision', n_jobs=1)
    print(f"        F1={f1.mean():.4f}  Recall={rec.mean():.4f}  AUC={auc.mean():.4f}", flush=True)
    return {
        'f1_mean':        round(float(f1.mean()),  4),
        'f1_std':         round(float(f1.std()),   4),
        'recall_mean':    round(float(rec.mean()), 4),
        'auc_mean':       round(float(auc.mean()), 4),
        'precision_mean': round(float(pre.mean()), 4),
    }


# =============================================================================
# MAIN
# =============================================================================

def run():
    print("=" * 70)
    print("  ScamShield Research Evaluation Pipeline")
    print("=" * 70)

    # 1. Load
    print("\n[1/7] Loading dataset ...", flush=True)
    df = pd.read_csv('scam_dataset_realistic.csv')
    X  = df.drop('label', axis=1).values
    y  = df['label'].values
    print(f"      {len(df):,} samples | scam={int(y.sum()):,} safe={int((y==0).sum()):,}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    df_tr, df_te = train_test_split(
        df, test_size=0.20, stratify=y, random_state=SEED)
    print(f"      Train={len(X_tr):,}  Test={len(X_te):,}")

    # 2. TF-IDF
    print("\n[2/7] Building TF-IDF representations ...", flush=True)
    texts_all = [row_to_text(r) for _, r in df.iterrows()]
    texts_tr  = [row_to_text(r) for _, r in df_tr.iterrows()]
    texts_te  = [row_to_text(r) for _, r in df_te.iterrows()]

    tfidf_cv  = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    X_tfidf_all = tfidf_cv.fit_transform(texts_all).toarray()

    tfidf_fit = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    X_tfidf_tr  = tfidf_fit.fit_transform(texts_tr).toarray()
    X_tfidf_te  = tfidf_fit.transform(texts_te).toarray()
    print("      Done.")

    # 3. Cross-validation
    print("\n[3/7] 3-Fold Cross-Validation ...", flush=True)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    cv_results = {}
    cv_results['Naive Bayes']         = run_cv(MultinomialNB(), np.abs(X_tfidf_all), y, cv, 'Naive Bayes')
    cv_results['LinearSVC']           = run_cv(CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=SEED), cv=3), X_tfidf_all, y, cv, 'LinearSVC')
    cv_results['Logistic Regression'] = run_cv(Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, C=0.5, class_weight='balanced', random_state=SEED))]), X, y, cv, 'Logistic Regression')
    cv_results['Random Forest']       = run_cv(RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=SEED, n_jobs=1), X, y, cv, 'Random Forest')
    cv_results['ScamShield GBM']      = run_cv(GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=SEED), X, y, cv, 'ScamShield GBM')

    # 4. Train final model
    print("\n[4/7] Training final model ...", flush=True)
    base        = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, min_samples_leaf=4, subsample=0.8, random_state=SEED)
    final_model = CalibratedClassifierCV(base, method='isotonic', cv=3)
    final_model.fit(X_tr, y_tr)
    joblib.dump(final_model, 'models/scam_detector_final.pkl')

    nb   = MultinomialNB();                                                    nb.fit(np.abs(X_tfidf_tr), y_tr)
    lsvc = CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=SEED), cv=3); lsvc.fit(X_tfidf_tr, y_tr)
    lr   = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, C=0.5, class_weight='balanced', random_state=SEED))]); lr.fit(X_tr, y_tr)
    rf   = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=SEED, n_jobs=1); rf.fit(X_tr, y_tr)
    print("      All models trained.")

    # 5. Test set evaluation
    print("\n[5/7] Test set evaluation ...", flush=True)
    test_metrics = {}
    all_preds    = {}
    for name, model, X_eval in [
        ('Naive Bayes',         nb,          X_tfidf_te),
        ('LinearSVC',           lsvc,        X_tfidf_te),
        ('Logistic Regression', lr,          X_te),
        ('Random Forest',       rf,          X_te),
        ('ScamShield GBM',      final_model, X_te),
    ]:
        m, preds = get_metrics(model, X_eval, y_te)
        test_metrics[name] = m
        all_preds[name]    = preds
        tag = '  <- proposed' if name == 'ScamShield GBM' else ''
        print(f"      {name:<24} F1={m['f1']:.4f} AUC={m['roc_auc']:.4f} Recall={m['recall']:.4f}{tag}")

    # 6. Adversarial robustness
    print("\n[6/7] Adversarial robustness ...", flush=True)
    scam_rows  = df_te[df_te['label'] == 1]
    scam_texts = [row_to_text(r) for _, r in scam_rows.iterrows()]
    y_scam     = np.ones(len(scam_texts), dtype=int)
    adv_results = {}
    base_recall = test_metrics['ScamShield GBM']['recall']
    print(f"      Clean recall: {base_recall:.4f}")
    for atk_name, atk_fn in [
        ('synonym_substitution', attack_synonym),
        ('homoglyph_attack',     attack_homoglyph),
        ('url_obfuscation',      attack_url_obfuscation),
    ]:
        mutated = [atk_fn(t) for t in scam_texts]
        X_adv   = np.array([extract_features(t) for t in mutated])
        preds   = final_model.predict(X_adv)
        _, r, f, _ = precision_recall_fscore_support(y_scam, preds, average='binary', zero_division=0)
        delta = round(float(r) - base_recall, 4)
        adv_results[atk_name] = {'recall': round(float(r), 4), 'f1': round(float(f), 4), 'delta_recall': delta}
        print(f"      {atk_name:<28} Recall={r:.4f}  delta={delta:+.4f}")

    # 7. Statistical significance
    print("\n[7/7] McNemar significance tests ...", flush=True)
    sig_results = {}
    our_preds   = all_preds['ScamShield GBM']
    for name in ['Naive Bayes', 'LinearSVC', 'Logistic Regression', 'Random Forest']:
        chi2, p = mcnemar_test(y_te, our_preds, all_preds[name])
        sig = 'Yes ***' if p < 0.001 else ('Yes **' if p < 0.01 else ('Yes *' if p < 0.05 else 'No'))
        sig_results[name] = {'chi2': chi2, 'p_value': p, 'significant': p < 0.05}
        print(f"      vs {name:<22} chi2={chi2:.3f} p={p:.6f} sig={sig}")

    # Feature importances
    gb_plain = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=SEED)
    gb_plain.fit(X_tr, y_tr)
    importances = {k: round(float(v), 6) for k, v in zip(FEATURE_NAMES, gb_plain.feature_importances_)}

    scaler   = StandardScaler()
    lr_plain = LogisticRegression(max_iter=1000, C=0.5, class_weight='balanced', random_state=SEED)
    lr_plain.fit(scaler.fit_transform(X_tr), y_tr)
    coefs = {k: round(float(v), 6) for k, v in zip(FEATURE_NAMES, lr_plain.coef_[0])}

    # Save results
    results = {
        'dataset':                  {'total': len(df), 'scam': int(y.sum()), 'safe': int((y==0).sum()), 'train': len(X_tr), 'test': len(X_te)},
        'cross_validation':         cv_results,
        'test_metrics':             test_metrics,
        'adversarial_robustness':   adv_results,
        'statistical_significance': sig_results,
        'feature_importances':      importances,
        'lr_coefficients':          coefs,
    }
    with open('eval_results/full_evaluation.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"\n  {'Model':<24} {'F1':>6} {'AUC':>6} {'Recall':>8} {'Prec':>6}")
    print("  " + "-" * 55)
    for name in ['Naive Bayes', 'LinearSVC', 'Logistic Regression', 'Random Forest', 'ScamShield GBM']:
        m   = test_metrics[name]
        tag = '  <- proposed' if name == 'ScamShield GBM' else ''
        print(f"  {name:<24} {m['f1']:>6.4f} {m['roc_auc']:>6.4f} {m['recall']:>8.4f} {m['precision']:>6.4f}{tag}")

    print(f"\n  Adversarial (scam-only recall):")
    print(f"  {'Clean':<28} {base_recall:.4f}")
    for atk, v in adv_results.items():
        print(f"  {atk:<28} {v['recall']:.4f}  ({v['delta_recall']:+.4f})")

    print(f"\n  Significance vs ScamShield:")
    for name, v in sig_results.items():
        print(f"  {name:<24} p={v['p_value']:.6f}  {'significant' if v['significant'] else 'not significant'}")

    print("\n  Saved: eval_results/full_evaluation.json")
    print("  Saved: models/scam_detector_final.pkl")
    print("=" * 70)


if __name__ == '__main__':
    run()